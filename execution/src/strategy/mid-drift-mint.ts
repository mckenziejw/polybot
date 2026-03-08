import type { MarketState, TradeAction, DataMode } from "../types.ts";
import type { Strategy } from "./strategy.ts";
import { registerStrategy } from "./strategy.ts";
import { loadConfig } from "../config.ts";

/**
 * Mid-price drift strategy — mint variant.
 *
 * 1. At market open, split USDC into YES+NO token pairs (on-chain via CTF Core).
 *    This is done immediately so settlement completes before the signal window.
 * 2. At t=60s, check mid-price drift. If drift > 0.10, sell the LOSER side
 *    as a maker (post at the loser's ask price).
 * 3. Hold the WINNER tokens to expiry. Any unsold pairs are fully hedged
 *    (YES+NO = $1) so there's no risk holding the remainder.
 *
 * Advantages:
 *   - Earns the spread instead of paying it (~1.5c/token savings)
 *   - Maker rebates instead of taker fees
 *   - 92.4% fill rate on maker sells (anti-predictive orderflow buys the loser)
 *   - No settlement delay at signal time (split already confirmed)
 *
 * Backtest: +9,328 PnL vs +4,495 direct buy, Sharpe 2.91 vs 1.43
 */
class MidDriftMintStrategy implements Strategy {
  readonly id = "mid-drift-mint";
  readonly dataMode: DataMode = "tick";

  // ── Config ──
  private signalDelaySec = 60;
  private driftThreshold = 0.10;
  private betDollars: number;
  private sellTimeoutSec = 120;   // cancel unfilled sell after N seconds
  private mintPairs = 10;         // pairs to mint at open (oversize to cover varying effective cost)

  constructor() {
    const config = loadConfig();
    this.betDollars = config.execution.betDollars;
  }

  private static readonly MARKET_DURATION_MS = 300_000;

  // ── Per-market state ──
  private marketOpenTime = 0;
  private skippingMarket = false;
  private splitSent = false;
  private signalFired = false;
  private sellPlaced = false;
  private sellPlacedTime = 0;
  private cancelSent = false;
  private targetLoserAssetId: string | null = null;
  private targetDirection: "Up" | "Down" | null = null;
  private sellSize = 0;

  async onMarketOpen(state: MarketState): Promise<void> {
    this.marketOpenTime = state.market.endTime.getTime() - MidDriftMintStrategy.MARKET_DURATION_MS;
    this.skippingMarket = false;
    this.splitSent = false;
    this.signalFired = false;
    this.sellPlaced = false;
    this.sellPlacedTime = 0;
    this.cancelSent = false;
    this.targetLoserAssetId = null;
    this.targetDirection = null;
    this.sellSize = 0;

    const elapsed = (Date.now() - this.marketOpenTime) / 1000;
    if (elapsed > this.signalDelaySec) {
      console.log(`[DRIFT-MINT] Joined ${elapsed.toFixed(0)}s into market — skipping ${state.market.slug}`);
      this.skippingMarket = true;
      return;
    }

    console.log(`[DRIFT-MINT] Market open — ${state.market.slug} (${elapsed.toFixed(0)}s elapsed)`);
  }

  async evaluate(state: MarketState): Promise<TradeAction[]> {
    if (this.skippingMarket) return [{ type: "NOOP" }];

    const elapsed = (Date.now() - this.marketOpenTime) / 1000;

    // ── Phase 1: Split at market open (first evaluate call) ──
    if (!this.splitSent) {
      this.splitSent = true;

      const splitAmount = BigInt(this.mintPairs) * 1_000_000n;
      console.log(
        `[DRIFT-MINT] Splitting ${this.mintPairs} USDC into YES+NO pairs (pre-signal mint)`
      );

      return [{
        type: "SPLIT",
        conditionId: state.market.conditionId,
        amount: splitAmount,
      }];
    }

    // ── Phase 2: Wait for signal time, then sell loser ──
    if (!this.signalFired) {
      if (elapsed < this.signalDelaySec) return [{ type: "NOOP" }];

      this.signalFired = true;

      const upBook = state.books[state.market.upTokenId];
      const downBook = state.books[state.market.downTokenId];

      if (!upBook || !downBook) {
        console.log("[DRIFT-MINT] Missing book data — skipping (holding hedged)");
        return [{ type: "NOOP" }];
      }

      const upMid = getMid(upBook);
      const downMid = getMid(downBook);

      if (upMid === null || downMid === null) {
        console.log("[DRIFT-MINT] Empty book (no bid/ask) — skipping (holding hedged)");
        return [{ type: "NOOP" }];
      }

      const upDrift = upMid - 0.50;
      const downDrift = downMid - 0.50;

      console.log(
        `[DRIFT-MINT] +${elapsed.toFixed(0)}s — Up mid=${upMid.toFixed(4)} (drift=${upDrift > 0 ? "+" : ""}${upDrift.toFixed(4)})` +
        ` | Down mid=${downMid.toFixed(4)} (drift=${downDrift > 0 ? "+" : ""}${downDrift.toFixed(4)})`
      );

      // Pick the winner (higher mid) and loser
      let bestDrift: number;
      let winnerDirection: "Up" | "Down";
      let loserAssetId: string;

      if (upDrift >= downDrift) {
        bestDrift = upDrift;
        winnerDirection = "Up";
        loserAssetId = state.market.downTokenId;
      } else {
        bestDrift = downDrift;
        winnerDirection = "Down";
        loserAssetId = state.market.upTokenId;
      }

      if (bestDrift < this.driftThreshold) {
        console.log(
          `[DRIFT-MINT] Max drift ${bestDrift.toFixed(4)} < threshold ${this.driftThreshold} — skipping (holding hedged)`
        );
        return [{ type: "NOOP" }];
      }

      // Check loser book for our sell price
      const loserBook = state.books[loserAssetId]!;
      if (loserBook.asks.length === 0 || loserBook.bids.length === 0) {
        console.log("[DRIFT-MINT] Loser book empty — skipping (holding hedged)");
        return [{ type: "NOOP" }];
      }

      // Sell at the loser's current ask (join the ask, maker)
      const sellPrice = loserBook.asks[0]!.price;
      if (sellPrice <= 0 || sellPrice >= 1) {
        console.log("[DRIFT-MINT] Invalid loser ask price — skipping (holding hedged)");
        return [{ type: "NOOP" }];
      }

      // Effective cost per winner token = 1.00 - sellPrice
      // Sell as many loser tokens as betDollars allows, capped by minted supply
      const effectiveCost = 1.00 - sellPrice;
      const numTokens = Math.min(
        Math.floor(this.betDollars / effectiveCost),
        this.mintPairs,
      );
      if (numTokens < 5) {
        console.log(`[DRIFT-MINT] Size ${numTokens} too small — skipping (holding hedged)`);
        return [{ type: "NOOP" }];
      }

      this.targetDirection = winnerDirection;
      this.targetLoserAssetId = loserAssetId;
      this.sellPlaced = true;
      this.sellPlacedTime = Date.now();
      this.sellSize = numTokens;

      const loserDir = winnerDirection === "Up" ? "Down" : "Up";
      console.log(
        `[DRIFT-MINT] Signal: ${winnerDirection} (drift=${bestDrift.toFixed(4)}) | ` +
        `SELL ${numTokens} ${loserDir} @ ${sellPrice.toFixed(3)} (maker) | ` +
        `Effective cost: ${effectiveCost.toFixed(3)} | Hedged remainder: ${this.mintPairs - numTokens} pairs`
      );

      return [{
        type: "PLACE_ORDER",
        assetId: loserAssetId,
        side: "SELL",
        price: sellPrice,
        size: numTokens,
        orderType: "GTC",
      }];
    }

    // ── Phase 3: Sell placed, wait for fill or timeout ──
    if (this.sellPlaced && !this.cancelSent && state.openOrders.length > 0) {
      const sellElapsed = (Date.now() - this.sellPlacedTime) / 1000;
      if (sellElapsed > this.sellTimeoutSec) {
        console.log(`[DRIFT-MINT] Sell unfilled after ${this.sellTimeoutSec}s — cancelling`);
        this.cancelSent = true;
        return [{ type: "CANCEL_ALL" }];
      }
    }

    return [{ type: "NOOP" }];
  }

  async onMarketClose(state: MarketState): Promise<TradeAction[]> {
    const posStr = this.targetDirection ?? "hedged (no signal)";
    console.log(`[DRIFT-MINT] Market close — position: ${posStr}`);

    if (state.openOrders.length > 0) {
      return [{ type: "CANCEL_ALL" }];
    }
    return [];
  }
}

function getMid(book: { bids: { price: number }[]; asks: { price: number }[] }): number | null {
  if (book.bids.length === 0 || book.asks.length === 0) return null;
  return (book.bids[0]!.price + book.asks[0]!.price) / 2;
}

registerStrategy("mid-drift-mint", () => new MidDriftMintStrategy());
