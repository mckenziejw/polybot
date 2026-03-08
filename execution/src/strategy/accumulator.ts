import type { MarketState, TradeAction, DataMode } from "../types.ts";
import type { Strategy } from "./strategy.ts";
import { registerStrategy } from "./strategy.ts";

// ── Normal CDF (Abramowitz & Stegun rational approximation) ──

function normalCdf(x: number): number {
  if (x < -8) return 0;
  if (x > 8) return 1;
  const a1 = 0.31938153, a2 = -0.356563782, a3 = 1.781477937;
  const a4 = -1.821255978, a5 = 1.330274429, p = 0.2316419;
  const sign = x < 0 ? -1 : 1;
  const absX = Math.abs(x);
  const t = 1 / (1 + p * absX);
  const pdf = Math.exp(-0.5 * absX * absX) / Math.sqrt(2 * Math.PI);
  const cdf = 1 - pdf * t * (a1 + t * (a2 + t * (a3 + t * (a4 + t * a5))));
  return sign < 0 ? 1 - cdf : cdf;
}

// ── Rolling volatility tracker ──

class VolTracker {
  private ticks: { t: number; p: number }[] = [];
  private windowMs: number;

  constructor(windowMs: number = 30_000) {
    this.windowMs = windowMs;
  }

  reset(): void { this.ticks = []; }

  push(timestamp: number, price: number): void {
    this.ticks.push({ t: timestamp, p: price });
    const cutoff = timestamp - this.windowMs;
    while (this.ticks.length > 0 && this.ticks[0]!.t < cutoff) {
      this.ticks.shift();
    }
    if (this.ticks.length > 6000) {
      this.ticks = this.ticks.slice(-6000);
    }
  }

  get tickCount(): number { return this.ticks.length; }

  realizedVol(scaleDurationMs: number): number {
    if (this.ticks.length < 10) return 0;
    let sumSqReturns = 0;
    for (let i = 1; i < this.ticks.length; i++) {
      const r = Math.log(this.ticks[i]!.p / this.ticks[i - 1]!.p);
      sumSqReturns += r * r;
    }
    const windowDurationMs = this.ticks[this.ticks.length - 1]!.t - this.ticks[0]!.t;
    if (windowDurationMs <= 0) return 0;
    return Math.sqrt((sumSqReturns / windowDurationMs) * scaleDurationMs);
  }
}

/**
 * Accumulator Strategy — Delta-neutral binary option arbitrage.
 *
 * Accumulates matched pairs of Up + Down tokens at combined cost < $1.00.
 * Since one side always resolves to $1.00 and the other to $0.00,
 * holding N pairs guarantees $N payout regardless of outcome.
 * Profit = N - total_cost, which is positive when avg cost per pair < $1.00.
 *
 * Uses BTC-derived fair value to identify mispriced tokens:
 * only buys when market ask < fair value (i.e., the token is underpriced
 * relative to what BTC is doing).
 *
 * Key constraint: total_cost must always remain < min(n_up, n_down).
 * This guarantees profitability even in worst-case resolution.
 */
export class Accumulator implements Strategy {
  readonly id = "accumulator";
  readonly dataMode: DataMode = "tick";

  // ── Parameters ──
  private orderSize = 5;
  private maxPerSide = 50;
  private minRemainingS = 15;          // stop trading near end
  private maxFirstLegPrice = 0.48;     // don't buy first leg above this
  private minEdge = 0.02;              // only buy when ask < fairValue - minEdge
  private volWindowMs = 30_000;
  private minVolTicks = 50;
  private decidedThreshold = 0.80;     // don't buy tokens our model says are worth < 0.20

  // ── Per-market state ──
  private btcPriceAtOpen: number | null = null;
  private marketDurationMs = 300_000;
  private volTracker: VolTracker;
  private lastBtcTimestamp = 0;
  private lastLogTime = 0;
  private totalCost = 0;               // total $ spent buying tokens this market

  constructor() {
    this.volTracker = new VolTracker(this.volWindowMs);
  }

  async onMarketOpen(state: MarketState): Promise<void> {
    const btc = state.externalData?.["binance-btcusdt"];
    this.btcPriceAtOpen = btc?.price ?? null;
    this.marketDurationMs = state.timeRemainingMs;
    this.volTracker = new VolTracker(this.volWindowMs);
    this.lastBtcTimestamp = 0;
    this.lastLogTime = 0;
    this.totalCost = 0;

    if (this.btcPriceAtOpen) {
      console.log(
        `[ACC] Market open — BTC ref: ${this.btcPriceAtOpen.toFixed(2)}, ` +
        `duration: ${(this.marketDurationMs / 1000).toFixed(0)}s`
      );
    } else {
      console.log("[ACC] Market open — waiting for BTC price...");
    }
  }

  async evaluate(state: MarketState): Promise<TradeAction[]> {
    const { market, books, openOrders, positions, timeRemainingMs } = state;
    const btc = state.externalData?.["binance-btcusdt"];

    // Feed BTC ticks to vol tracker
    if (btc && btc.timestamp > this.lastBtcTimestamp) {
      this.volTracker.push(btc.timestamp, btc.price);
      this.lastBtcTimestamp = btc.timestamp;
    }

    // Capture reference price
    if (!this.btcPriceAtOpen) {
      this.btcPriceAtOpen = btc?.price ?? null;
      if (this.btcPriceAtOpen) {
        console.log(`[ACC] BTC ref captured: ${this.btcPriceAtOpen.toFixed(2)}`);
      }
      return [{ type: "NOOP" }];
    }

    const timeRemainingS = timeRemainingMs / 1000;

    // Last N seconds: cancel everything
    if (timeRemainingS < this.minRemainingS) {
      if (openOrders.length > 0) {
        return [{ type: "CANCEL_ALL" }];
      }
      return [{ type: "NOOP" }];
    }

    // Compute fair value
    const btcNow = btc?.price ?? this.btcPriceAtOpen;
    const btcReturn = (btcNow - this.btcPriceAtOpen) / this.btcPriceAtOpen;
    const timeElapsedMs = Math.max(1, this.marketDurationMs - timeRemainingMs);
    const timeElapsedFrac = Math.max(0.001, timeElapsedMs / this.marketDurationMs);
    const vol = this.volTracker.realizedVol(this.marketDurationMs);

    // Need vol estimate before trading
    if (vol <= 0 || this.volTracker.tickCount < this.minVolTicks) {
      return [{ type: "NOOP" }];
    }

    const z = (btcReturn / vol) * Math.sqrt(timeElapsedFrac);
    const fairUp = normalCdf(z);
    const fairDown = 1 - fairUp;

    // Inventory
    const upPos = positions.find((p) => p.assetId === market.upTokenId);
    const downPos = positions.find((p) => p.assetId === market.downTokenId);
    const upSize = upPos?.size ?? 0;
    const downSize = downPos?.size ?? 0;

    // Derive totalCost from positions every tick (robust to fills we didn't track)
    this.totalCost = (upSize * (upPos?.avgEntryPrice ?? 0)) +
                     (downSize * (downPos?.avgEntryPrice ?? 0));

    const tickSize = market.tickSize || 0.01;
    const clamp = (p: number) =>
      Math.max(tickSize, Math.min(1 - tickSize, Math.round(p / tickSize) * tickSize));

    // Periodic logging
    const now = Date.now();
    if (now - this.lastLogTime > 10_000) {
      this.lastLogTime = now;
      const pairs = Math.min(upSize, downSize);
      const costPerPair = pairs > 0 ? this.totalCost / pairs : 0;
      console.log(
        `[ACC] fair=(${fairUp.toFixed(3)}/${fairDown.toFixed(3)}) ` +
        `z=${z.toFixed(3)} vol=${(vol * 100).toFixed(4)}% ` +
        `up=${upSize} down=${downSize} pairs=${pairs} ` +
        `cost=$${this.totalCost.toFixed(2)} ` +
        (pairs > 0 ? `cost/pair=$${costPerPair.toFixed(3)} ` : "") +
        `${timeRemainingS.toFixed(0)}s left`
      );
    }

    // Determine which side we need
    const needUp = upSize <= downSize;
    const needDown = downSize <= upSize;

    const actions: TradeAction[] = [];

    // Try to buy the side we're short on
    if (needUp && upSize < this.maxPerSide) {
      const action = this.tryBuy(
        market.upTokenId, fairUp, books[market.upTokenId],
        upSize, downSize, openOrders, clamp, tickSize
      );
      if (action) actions.push(action);
    }

    if (needDown && downSize < this.maxPerSide) {
      const action = this.tryBuy(
        market.downTokenId, fairDown, books[market.downTokenId],
        downSize, upSize, openOrders, clamp, tickSize
      );
      if (action) actions.push(action);
    }

    return actions.length > 0 ? actions : [{ type: "NOOP" }];
  }

  /**
   * Try to place a buy order for a token if it's underpriced relative to fair value
   * and the purchase maintains the profitability constraint.
   */
  private tryBuy(
    assetId: string,
    fairValue: number,
    book: import("../types.ts").OrderBookState | undefined,
    thisSize: number,
    otherSize: number,
    openOrders: import("../types.ts").OpenOrder[],
    clamp: (p: number) => number,
    tickSize: number,
  ): TradeAction | null {
    if (!book || book.asks.length === 0) return null;

    const bestAsk = book.asks[0]!.price;

    // Don't buy tokens that are near-certainly worthless
    if (fairValue < (1 - this.decidedThreshold)) return null;

    // Only buy when market ask is below our fair value minus edge
    const maxBid = fairValue - this.minEdge;
    if (bestAsk > maxBid) return null;

    // First leg check: if this would be the ONLY side we hold,
    // apply stricter price limit
    const isFirstLeg = otherSize === 0;
    if (isFirstLeg && bestAsk > this.maxFirstLegPrice) return null;

    // Profitability constraint: total_cost must remain < min(new_this, other)
    const newThisSize = thisSize + this.orderSize;
    const bidPrice = clamp(Math.min(bestAsk, maxBid));
    const newTotalCost = this.totalCost + bidPrice * this.orderSize;
    const newMinTokens = Math.min(newThisSize, otherSize);

    if (!isFirstLeg && newTotalCost >= newMinTokens) {
      return null; // Would break profitability
    }

    // Don't duplicate existing orders for the same asset
    const existingBuy = openOrders.find(
      (o) => o.assetId === assetId && o.side === "BUY"
    );
    if (existingBuy) {
      // Requote if price drifted significantly
      if (Math.abs(existingBuy.price - bidPrice) < tickSize * 2) {
        return null; // Close enough, keep existing
      }
      // Cancel and replace
      return { type: "CANCEL_ORDER", orderId: existingBuy.orderId };
    }

    return {
      type: "PLACE_ORDER",
      assetId,
      side: "BUY",
      price: bidPrice,
      size: this.orderSize,
      orderType: "GTC",
    };
  }

  async onMarketClose(_state: MarketState): Promise<TradeAction[]> {
    const positions = _state.positions;
    const upSize = positions.find((p) => p.assetId === _state.market.upTokenId)?.size ?? 0;
    const downSize = positions.find((p) => p.assetId === _state.market.downTokenId)?.size ?? 0;
    const pairs = Math.min(upSize, downSize);
    const costPerPair = pairs > 0 ? this.totalCost / pairs : 0;

    console.log(
      `[ACC] Market close — up=${upSize} down=${downSize} pairs=${pairs} ` +
      `cost=$${this.totalCost.toFixed(2)} ` +
      (pairs > 0 ? `cost/pair=$${costPerPair.toFixed(3)} ` : "") +
      `guaranteed_pnl=$${pairs > 0 ? (pairs - this.totalCost).toFixed(2) : "0.00"}`
    );
    return [{ type: "CANCEL_ALL" }];
  }
}

registerStrategy("accumulator", () => new Accumulator());
