import type { MarketState, TradeAction, DataMode } from "../types.ts";
import type { Strategy } from "./strategy.ts";
import { registerStrategy } from "./strategy.ts";
import { loadConfig } from "../config.ts";

/**
 * Mid-price drift strategy — buy the token whose mid-price drifts above 0.50.
 *
 * Signal: At signalDelaySec after market open, check both tokens' mid-prices.
 * If one token's mid has drifted > driftThreshold above 0.50, buy it.
 * The Polymarket orderbook's own price discovery is the best predictor —
 * no external feed needed.
 *
 * Backtest results (4,283 markets):
 *   t=60s, d=0.10: 72.8% win rate, 2,460 trades, +4,495 (taker) / +9,750 (maker)
 */
class MidDriftStrategy implements Strategy {
  readonly id = "mid-drift";
  readonly dataMode: DataMode = "tick";

  // ── Config ──
  private signalDelaySec = 60;        // seconds after open to read signal
  private driftThreshold = 0.10;      // minimum |mid - 0.50| to act
  private betDollars: number;
  private cancelAfterSec = 30;        // cancel unfilled limit after N seconds

  constructor() {
    const config = loadConfig();
    this.betDollars = config.execution.betDollars;
  }

  private static readonly MARKET_DURATION_MS = 300_000; // 5 minutes

  // ── Per-market state ──
  private marketOpenTime = 0;
  private skippingMarket = false;
  private signalFired = false;
  private orderPlaced = false;
  private orderPlacedTime = 0;
  private cancelSent = false;
  private targetAssetId: string | null = null;
  private targetDirection: "Up" | "Down" | null = null;

  async onMarketOpen(state: MarketState): Promise<void> {
    this.marketOpenTime = state.market.endTime.getTime() - MidDriftStrategy.MARKET_DURATION_MS;
    this.skippingMarket = false;
    this.signalFired = false;
    this.orderPlaced = false;
    this.orderPlacedTime = 0;
    this.cancelSent = false;
    this.targetAssetId = null;
    this.targetDirection = null;

    const elapsed = (Date.now() - this.marketOpenTime) / 1000;
    if (elapsed > this.signalDelaySec) {
      console.log(`[DRIFT] Joined ${elapsed.toFixed(0)}s into market — skipping ${state.market.slug}`);
      this.skippingMarket = true;
      return;
    }

    console.log(`[DRIFT] Market open — ${state.market.slug} (${elapsed.toFixed(0)}s elapsed)`);
  }

  async evaluate(state: MarketState): Promise<TradeAction[]> {
    if (this.skippingMarket) return [{ type: "NOOP" }];

    const elapsed = (Date.now() - this.marketOpenTime) / 1000;

    // ── Phase 1: Wait for signal time ──
    if (!this.signalFired) {
      if (elapsed < this.signalDelaySec) return [{ type: "NOOP" }];

      this.signalFired = true;

      const upBook = state.books[state.market.upTokenId];
      const downBook = state.books[state.market.downTokenId];

      if (!upBook || !downBook) {
        console.log("[DRIFT] Missing book data — skipping");
        return [{ type: "NOOP" }];
      }

      const upMid = getMid(upBook);
      const downMid = getMid(downBook);

      if (upMid === null || downMid === null) {
        console.log("[DRIFT] Empty book (no bid/ask) — skipping");
        return [{ type: "NOOP" }];
      }

      const upDrift = upMid - 0.50;
      const downDrift = downMid - 0.50;

      console.log(
        `[DRIFT] +${elapsed.toFixed(0)}s — Up mid=${upMid.toFixed(4)} (drift=${upDrift > 0 ? "+" : ""}${upDrift.toFixed(4)})` +
        ` | Down mid=${downMid.toFixed(4)} (drift=${downDrift > 0 ? "+" : ""}${downDrift.toFixed(4)})`
      );

      // Pick the token with the larger positive drift
      let bestDrift: number;
      let bestDirection: "Up" | "Down";
      let bestAssetId: string;

      if (upDrift >= downDrift) {
        bestDrift = upDrift;
        bestDirection = "Up";
        bestAssetId = state.market.upTokenId;
      } else {
        bestDrift = downDrift;
        bestDirection = "Down";
        bestAssetId = state.market.downTokenId;
      }

      if (bestDrift < this.driftThreshold) {
        console.log(
          `[DRIFT] Max drift ${bestDrift.toFixed(4)} < threshold ${this.driftThreshold} — skipping`
        );
        return [{ type: "NOOP" }];
      }

      // Place a buy at the best ask (taker entry)
      const book = state.books[bestAssetId]!;
      if (book.asks.length === 0) {
        console.log("[DRIFT] No asks available — skipping");
        return [{ type: "NOOP" }];
      }

      const askPrice = book.asks[0]!.price;
      const size = Math.floor(this.betDollars / askPrice);
      if (size < 5) {
        console.log(`[DRIFT] Size ${size} too small at ask ${askPrice.toFixed(3)} — skipping`);
        return [{ type: "NOOP" }];
      }

      this.targetDirection = bestDirection;
      this.targetAssetId = bestAssetId;
      this.orderPlaced = true;
      this.orderPlacedTime = Date.now();

      console.log(
        `[DRIFT] BUY ${bestDirection} ${size}@${askPrice.toFixed(3)} ` +
        `(drift=${bestDrift.toFixed(4)}) at +${elapsed.toFixed(0)}s`
      );

      return [{
        type: "PLACE_ORDER",
        assetId: bestAssetId,
        side: "BUY",
        price: askPrice,
        size,
        orderType: "GTC",
      }];
    }

    // ── Phase 2: Order placed, check fill status ──
    if (this.orderPlaced) {
      // Check if we have a position (order was filled)
      const hasPosition = this.targetAssetId
        ? state.positions.some(p => p.assetId === this.targetAssetId && p.size > 0)
        : false;

      if (hasPosition) {
        // Filled — hold to expiry, nothing to do
        return [{ type: "NOOP" }];
      }

      // Not filled yet — cancel after timeout (once)
      if (!this.cancelSent && state.openOrders.length > 0) {
        const orderElapsed = (Date.now() - this.orderPlacedTime) / 1000;
        if (orderElapsed > this.cancelAfterSec) {
          console.log(`[DRIFT] Order unfilled after ${this.cancelAfterSec}s — cancelling`);
          this.cancelSent = true;
          return [{ type: "CANCEL_ALL" }];
        }
      }
    }

    return [{ type: "NOOP" }];
  }

  async onMarketClose(state: MarketState): Promise<TradeAction[]> {
    const posStr = this.targetDirection ?? "none";
    console.log(`[DRIFT] Market close — position: ${posStr}`);

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

registerStrategy("mid-drift", () => new MidDriftStrategy());
