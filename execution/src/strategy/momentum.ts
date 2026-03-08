import type { MarketState, TradeAction, DataMode } from "../types.ts";
import type { Strategy } from "./strategy.ts";
import { registerStrategy } from "./strategy.ts";

/**
 * Momentum strategy (Config B) — directional bet with value-aware entry.
 *
 * 1. At signalDelaySec after market open, read BTC return from reference.
 * 2. If |return| >= minSignalMagnitude, estimate win probability and compute
 *    target price = p_win - desiredEdge.
 * 3. Place a limit order at target price for the predicted direction.
 * 4. If unfilled after limitWindowSec, cancel. One shot per market.
 * 5. Hold to expiry.
 */
class MomentumStrategy implements Strategy {
  readonly id = "momentum";
  readonly dataMode: DataMode = "tick";

  // ── Config ──
  private signalDelaySec = 3;          // seconds after market open to read signal
  private minSignalMagnitude = 0.0001; // minimum |BTC return| to act (0.01%)
  private desiredEdge = 0.10;          // buy at p_win - edge
  private betDollars = 100;            // dollars to risk per market
  private limitWindowSec = 60;         // seconds to wait for limit fill

  // ── Per-market state ──
  private btcRefPrice: number | null = null;
  private marketOpenTime = 0;
  private signalFired = false;
  private orderPlaced = false;
  private orderPlacedTime = 0;
  private targetDirection: "Up" | "Down" | null = null;
  private targetAssetId: string | null = null;
  private targetPrice = 0;
  private estimatedProb = 0;

  async onMarketOpen(state: MarketState): Promise<void> {
    this.btcRefPrice = null;
    this.marketOpenTime = Date.now();
    this.signalFired = false;
    this.orderPlaced = false;
    this.orderPlacedTime = 0;
    this.targetDirection = null;
    this.targetAssetId = null;
    this.targetPrice = 0;
    this.estimatedProb = 0;

    const btc = state.externalData?.["binance-btcusdt"];
    if (btc) {
      this.btcRefPrice = btc.price;
      console.log(`[MOM] Market open — BTC ref: ${btc.price.toFixed(2)}`);
    } else {
      console.log("[MOM] Market open — waiting for BTC price...");
    }
  }

  async evaluate(state: MarketState): Promise<TradeAction[]> {
    const btc = state.externalData?.["binance-btcusdt"];
    if (!btc) return [{ type: "NOOP" }];

    // Capture BTC ref on first tick if we missed it at open
    if (this.btcRefPrice === null) {
      this.btcRefPrice = btc.price;
      console.log(`[MOM] BTC ref captured: ${btc.price.toFixed(2)}`);
    }

    const elapsed = (Date.now() - this.marketOpenTime) / 1000;

    // ── Phase 1: Wait for signal ──
    if (!this.signalFired) {
      if (elapsed < this.signalDelaySec) return [{ type: "NOOP" }];

      this.signalFired = true;
      const btcReturn = (btc.price - this.btcRefPrice) / this.btcRefPrice;

      if (Math.abs(btcReturn) < this.minSignalMagnitude) {
        console.log(
          `[MOM] BTC move ${(Math.abs(btcReturn) * 100).toFixed(4)}% < ` +
          `${(this.minSignalMagnitude * 100).toFixed(2)}% threshold at +${elapsed.toFixed(0)}s — skipping`
        );
        return [{ type: "NOOP" }];
      }

      // Estimate win probability and compute target price
      this.estimatedProb = estimateWinProb(Math.abs(btcReturn));
      this.targetPrice = this.estimatedProb - this.desiredEdge;
      this.targetDirection = btcReturn > 0 ? "Up" : "Down";
      this.targetAssetId = this.targetDirection === "Up"
        ? state.market.upTokenId
        : state.market.downTokenId;

      console.log(
        `[MOM] SIGNAL: BTC ${btcReturn > 0 ? "+" : ""}${(btcReturn * 100).toFixed(4)}% at +${elapsed.toFixed(0)}s` +
        ` → ${this.targetDirection} p_win=${this.estimatedProb.toFixed(2)} target=${this.targetPrice.toFixed(3)}`
      );

      if (this.targetPrice <= 0.01) {
        console.log(`[MOM] Target price ${this.targetPrice.toFixed(3)} too low — skipping`);
        return [{ type: "NOOP" }];
      }

      // Try immediate fill if ask is already at/below target
      return this.tryPlaceOrder(state, elapsed);
    }

    // ── Phase 2: Waiting for limit fill ──
    if (this.orderPlaced) {
      // Check if limit window expired
      const orderElapsed = (Date.now() - this.orderPlacedTime) / 1000;
      if (orderElapsed > this.limitWindowSec) {
        // Check if we actually got filled (have position)
        const hasPosition = state.positions.some(
          p => p.assetId === this.targetAssetId && p.size > 0
        );
        if (!hasPosition && state.openOrders.length > 0) {
          console.log(`[MOM] Limit window expired (${this.limitWindowSec}s) — cancelling unfilled order`);
          return [{ type: "CANCEL_ALL" }];
        }
      }
      return [{ type: "NOOP" }];
    }

    // ── Phase 3: Signal fired but order not yet placed (ask was too high) ──
    if (this.targetAssetId && !this.orderPlaced) {
      // Check if we're still within the limit window from signal time
      const signalTime = this.marketOpenTime + this.signalDelaySec * 1000;
      const sinceSignal = (Date.now() - signalTime) / 1000;
      if (sinceSignal > this.limitWindowSec) {
        console.log(`[MOM] Limit window expired without entry — skipping`);
        this.targetAssetId = null; // prevent further attempts
        return [{ type: "NOOP" }];
      }

      return this.tryPlaceOrder(state, elapsed);
    }

    return [{ type: "NOOP" }];
  }

  private tryPlaceOrder(state: MarketState, elapsed: number): TradeAction[] {
    if (!this.targetAssetId || !this.targetDirection) return [{ type: "NOOP" }];

    const book = state.books[this.targetAssetId];
    if (!book || book.asks.length === 0) return [{ type: "NOOP" }];

    const askPrice = book.asks[0]!.price;

    // Only buy if ask is at or below our target
    if (askPrice > this.targetPrice) return [{ type: "NOOP" }];

    // Compute size in tokens from dollar budget
    const size = Math.floor(this.betDollars / askPrice);
    if (size < 5) {
      console.log(`[MOM] Size ${size} too small at ask ${askPrice.toFixed(3)} — skipping`);
      return [{ type: "NOOP" }];
    }

    console.log(
      `[MOM] BUY ${this.targetDirection} ${size}@${askPrice.toFixed(3)} ` +
      `(target=${this.targetPrice.toFixed(3)}, p=${this.estimatedProb.toFixed(2)}) at +${elapsed.toFixed(0)}s`
    );

    this.orderPlaced = true;
    this.orderPlacedTime = Date.now();

    return [{
      type: "PLACE_ORDER",
      assetId: this.targetAssetId,
      side: "BUY",
      price: askPrice,
      size,
      orderType: "GTC",
    }];
  }

  async onMarketClose(state: MarketState): Promise<TradeAction[]> {
    const btc = state.externalData?.["binance-btcusdt"];
    const btcNow = btc?.price ?? 0;
    const btcReturn = this.btcRefPrice
      ? ((btcNow - this.btcRefPrice) / this.btcRefPrice * 100).toFixed(4)
      : "?";

    const posStr = this.orderPlaced
      ? `${this.targetDirection} @ target=${this.targetPrice.toFixed(3)}`
      : "none";
    console.log(`[MOM] Market close — BTC return: ${btcReturn}% | position: ${posStr}`);

    // Cancel any unfilled orders
    if (state.openOrders.length > 0) {
      return [{ type: "CANCEL_ALL" }];
    }
    return [];
  }
}

/**
 * Piecewise linear estimate of P(signal correct) given |BTC return|.
 * Calibrated from backtest over ~60 markets.
 */
function estimateWinProb(btcReturnAbs: number): number {
  const anchors: [number, number][] = [
    [0.00000, 0.50],
    [0.00005, 0.55],
    [0.00010, 0.60],
    [0.00015, 0.67],
    [0.00020, 0.75],
    [0.00030, 0.83],
    [0.00050, 0.88],
  ];

  if (btcReturnAbs <= anchors[0]![0]) return anchors[0]![1];
  if (btcReturnAbs >= anchors[anchors.length - 1]![0]) {
    return Math.min(anchors[anchors.length - 1]![1], 0.90);
  }

  for (let i = 0; i < anchors.length - 1; i++) {
    const [x0, y0] = anchors[i]!;
    const [x1, y1] = anchors[i + 1]!;
    if (btcReturnAbs >= x0 && btcReturnAbs <= x1) {
      const t = (btcReturnAbs - x0) / (x1 - x0);
      return y0 + t * (y1 - y0);
    }
  }

  return 0.50;
}

registerStrategy("momentum", () => new MomentumStrategy());
