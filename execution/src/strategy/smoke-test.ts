import type { MarketState, TradeAction, Side, DataMode } from "../types.ts";
import type { Strategy } from "./strategy.ts";
import { registerStrategy } from "./strategy.ts";

/**
 * Smoke test strategy for initial live validation.
 *
 * Waits for the mid price to be in [0.4, 0.6] (symmetric market), then places
 * limit buys for both Up and Down tokens at the current best bid. After 3 seconds,
 * cancels any unfilled orders.
 *
 * If both sides fill: holds to expiry (captures spread).
 * If one side fills: sells if mid deviates > 0.1 from cost basis in either direction.
 *
 * Trades only once per market to minimize exposure.
 */

type Phase =
  | "WAITING"    // watching for entry conditions
  | "POSTED"     // orders placed, waiting for fills
  | "BOTH_HELD"  // both sides filled, holding to expiry
  | "ONE_HELD"   // one side filled, monitoring for exit
  | "EXITING"    // sell order placed, waiting for fill
  | "DONE";      // trade complete for this market

interface HeldPosition {
  assetId: string;
  costBasis: number;
  size: number;
}

export class SmokeTestStrategy implements Strategy {
  readonly id = "smoke-test";
  readonly dataMode: DataMode = "tick";

  private phase: Phase = "WAITING";
  private postedAt = 0;
  private upOrderId: string | null = null;
  private downOrderId: string | null = null;
  private upFilled = false;
  private downFilled = false;
  private held: HeldPosition | null = null;
  private exitPostedAt = 0;

  private tokenLabel(assetId: string, market: MarketState["market"]): string {
    if (assetId === market.upTokenId) return "Up";
    if (assetId === market.downTokenId) return "Down";
    return assetId.slice(0, 8) + "…";
  }

  private ts(): string {
    return new Date().toISOString().slice(11, 23); // HH:MM:SS.mmm
  }

  async onMarketOpen(_state: MarketState): Promise<void> {
    this.phase = "WAITING";
    this.postedAt = 0;
    this.upOrderId = null;
    this.downOrderId = null;
    this.upFilled = false;
    this.downFilled = false;
    this.held = null;
    this.exitPostedAt = 0;
  }

  async evaluate(state: MarketState): Promise<TradeAction[]> {
    const { market, books, openOrders, positions } = state;

    switch (this.phase) {
      case "EXITING":
        return this.evalExiting(state);

      case "DONE":
        return [{ type: "NOOP" }];

      case "WAITING":
        return this.evalWaiting(state);

      case "POSTED":
        return this.evalPosted(state);

      case "BOTH_HELD":
        // Hold to expiry, nothing to do
        return [{ type: "NOOP" }];

      case "ONE_HELD":
        return this.evalOneHeld(state);
    }
  }

  async onMarketClose(_state: MarketState): Promise<TradeAction[]> {
    this.phase = "DONE";
    return [{ type: "CANCEL_ALL" }];
  }

  private evalWaiting(state: MarketState): TradeAction[] {
    const { market, books } = state;
    const upBook = books[market.upTokenId];
    if (!upBook || upBook.bids.length === 0 || upBook.asks.length === 0) {
      return [{ type: "NOOP" }];
    }

    const bestBid = upBook.bids[0]!.price;
    const bestAsk = upBook.asks[0]!.price;
    const mid = (bestBid + bestAsk) / 2;

    if (mid < 0.4 || mid > 0.6) {
      return [{ type: "NOOP" }];
    }

    // Get down token best bid for the down-side order
    const downBook = books[market.downTokenId];
    if (!downBook || downBook.bids.length === 0) {
      return [{ type: "NOOP" }];
    }

    const downBestBid = downBook.bids[0]!.price;

    this.phase = "POSTED";
    this.postedAt = Date.now();

    console.log(
      `[SmokeTest ${this.ts()}] Placing orders — Up bid: ${bestBid}, Down bid: ${downBestBid}, mid: ${mid.toFixed(3)}`
    );

    return [
      {
        type: "PLACE_ORDER",
        assetId: market.upTokenId,
        side: "BUY" as Side,
        price: bestBid,
        size: 5,
        orderType: "GTC",
      },
      {
        type: "PLACE_ORDER",
        assetId: market.downTokenId,
        side: "BUY" as Side,
        price: downBestBid,
        size: 5,
        orderType: "GTC",
      },
    ];
  }

  private evalPosted(state: MarketState): TradeAction[] {
    const { positions, openOrders } = state;

    // Check which sides have filled by looking at positions
    const upPos = positions.find((p) => p.assetId === state.market.upTokenId && p.size > 0);
    const downPos = positions.find((p) => p.assetId === state.market.downTokenId && p.size > 0);

    if (upPos && downPos) {
      // Both filled
      this.phase = "BOTH_HELD";
      console.log(
        `[SmokeTest ${this.ts()}] Both sides filled — Up: ${upPos.avgEntryPrice.toFixed(2)}, Down: ${downPos.avgEntryPrice.toFixed(2)}`
      );
      // Cancel any remaining orders just in case
      if (openOrders.length > 0) {
        return [{ type: "CANCEL_ALL" }];
      }
      return [{ type: "NOOP" }];
    }

    // Check 3-second timeout
    const elapsed = Date.now() - this.postedAt;
    if (elapsed >= 3000) {
      if (upPos || downPos) {
        // One side filled
        const filled = upPos || downPos!;
        this.held = {
          assetId: filled.assetId,
          costBasis: filled.avgEntryPrice,
          size: filled.size,
        };
        this.phase = "ONE_HELD";
        const label = this.tokenLabel(filled.assetId, state.market);
        console.log(
          `[SmokeTest ${this.ts()}] One side filled (${label}) after ${(elapsed / 1000).toFixed(1)}s, cancelling unfilled. Cost basis: ${filled.avgEntryPrice.toFixed(2)}`
        );
        return [{ type: "CANCEL_ALL" }];
      } else {
        // Neither filled after 3s, cancel and we're done
        this.phase = "DONE";
        console.log(`[SmokeTest ${this.ts()}] No fills after ${(elapsed / 1000).toFixed(1)}s, cancelling all`);
        if (openOrders.length > 0) {
          return [{ type: "CANCEL_ALL" }];
        }
        return [{ type: "NOOP" }];
      }
    }

    return [{ type: "NOOP" }];
  }

  private evalOneHeld(state: MarketState): TradeAction[] {
    if (!this.held) {
      this.phase = "DONE";
      return [{ type: "NOOP" }];
    }

    // Wait for on-chain settlement before trying to sell
    const pos = state.positions.find((p) => p.assetId === this.held!.assetId);
    if (pos && !pos.settled) {
      return [{ type: "NOOP" }];
    }

    const book = state.books[this.held.assetId];
    if (!book || book.bids.length === 0 || book.asks.length === 0) {
      return [{ type: "NOOP" }];
    }

    const bestBid = book.bids[0]!.price;
    const bestAsk = book.asks[0]!.price;
    const mid = (bestBid + bestAsk) / 2;
    const deviation = mid - this.held.costBasis;

    if (Math.abs(deviation) > 0.1) {
      this.phase = "EXITING";
      this.exitPostedAt = Date.now();
      const label = this.tokenLabel(this.held.assetId, state.market);
      const reason = deviation > 0 ? "take profit" : "stop loss";
      console.log(
        `[SmokeTest ${this.ts()}] Exiting ${label} position (${reason}) — mid: ${mid.toFixed(3)}, cost: ${this.held.costBasis.toFixed(2)}, deviation: ${deviation.toFixed(3)}`
      );
      return [
        {
          type: "PLACE_ORDER",
          assetId: this.held.assetId,
          side: "SELL" as Side,
          price: bestBid,
          size: this.held.size,
          orderType: "GTC",
        },
      ];
    }

    return [{ type: "NOOP" }];
  }

  private evalExiting(state: MarketState): TradeAction[] {
    if (!this.held) {
      this.phase = "DONE";
      return [{ type: "NOOP" }];
    }

    // Check if position has been sold (size dropped to 0)
    const pos = state.positions.find((p) => p.assetId === this.held!.assetId);
    if (!pos || pos.size <= 0) {
      const label = this.tokenLabel(this.held.assetId, state.market);
      console.log(`[SmokeTest ${this.ts()}] ${label} sell order filled — position closed`);
      this.phase = "DONE";
      this.held = null;
      return [{ type: "NOOP" }];
    }

    return [{ type: "NOOP" }];
  }
}

registerStrategy("smoke-test", () => new SmokeTestStrategy());
