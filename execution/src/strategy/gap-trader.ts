import type { MarketState, TradeAction, DataMode } from "../types.ts";
import type { Strategy } from "./strategy.ts";
import { registerStrategy } from "./strategy.ts";

// ── Normal CDF (Abramowitz & Stegun) ──

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

// ── Student's t CDF (regularized incomplete beta function approach) ──
// Uses the relationship: tCdf(x, ν) = 1 - 0.5 * I(ν/(ν+x²), ν/2, 1/2) for x >= 0

function logGamma(x: number): number {
  // Lanczos approximation
  const g = 7;
  const c = [
    0.99999999999980993, 676.5203681218851, -1259.1392167224028,
    771.32342877765313, -176.61502916214059, 12.507343278686905,
    -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7
  ];
  if (x < 0.5) {
    return Math.log(Math.PI / Math.sin(Math.PI * x)) - logGamma(1 - x);
  }
  x -= 1;
  let a = c[0]!;
  const t = x + g + 0.5;
  for (let i = 1; i < g + 2; i++) {
    a += c[i]! / (x + i);
  }
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}

function regularizedBeta(x: number, a: number, b: number): number {
  // Continued fraction approximation (Lentz's method)
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  // Use symmetry when x > (a+1)/(a+b+2) for better convergence
  if (x > (a + 1) / (a + b + 2)) {
    return 1 - regularizedBeta(1 - x, b, a);
  }
  const lnPre = a * Math.log(x) + b * Math.log(1 - x)
    - Math.log(a) - logGamma(a) - logGamma(b) + logGamma(a + b);
  const pre = Math.exp(lnPre);
  // Lentz continued fraction
  let f = 1, c = 1, d = 1 - (a + b) * x / (a + 1);
  if (Math.abs(d) < 1e-30) d = 1e-30;
  d = 1 / d;
  f = d;
  for (let m = 1; m <= 200; m++) {
    // Even step
    let num = m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m));
    d = 1 + num * d; if (Math.abs(d) < 1e-30) d = 1e-30; d = 1 / d;
    c = 1 + num / c; if (Math.abs(c) < 1e-30) c = 1e-30;
    f *= c * d;
    // Odd step
    num = -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1));
    d = 1 + num * d; if (Math.abs(d) < 1e-30) d = 1e-30; d = 1 / d;
    c = 1 + num / c; if (Math.abs(c) < 1e-30) c = 1e-30;
    const delta = c * d;
    f *= delta;
    if (Math.abs(delta - 1) < 1e-10) break;
  }
  return pre * f;
}

function studentTCdf(x: number, nu: number): number {
  if (nu <= 0) return normalCdf(x); // fallback
  const t2 = x * x;
  const p = regularizedBeta(nu / (nu + t2), nu / 2, 0.5);
  return x >= 0 ? 1 - 0.5 * p : 0.5 * p;
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
 * Gap Trader Strategy — Directional trading on fair value gaps.
 *
 * Buys a token when the market price is significantly below our BTC-derived
 * fair value. Exits when the gap closes (take profit) or widens against us
 * (stop loss via trailing stop).
 *
 * Only holds one position at a time. Plays whichever side (Up or Down)
 * shows the larger mispricing.
 */

interface ActiveTrade {
  assetId: string;
  tokenLabel: string;
  entryPrice: number;
  entryFairValue: number;
  size: number;
  highWaterFair: number;  // highest fair value seen since entry (for trailing stop)
  volScale: number;       // vol/baselineVol at entry, clamped to [minStopScale, maxStopScale]
}

export class GapTrader implements Strategy {
  readonly id = "gap-trader";
  readonly dataMode: DataMode = "tick";

  // ── Entry parameters ──
  private orderSize = 5;
  private minEntryGap = 0.03;          // only enter when fair - ask >= this
  private maxPositionSize = 25;        // max tokens in a single trade
  private minRemainingS = 30;          // don't enter new trades near end
  private decidedThreshold = 0.85;     // don't trade near-certain outcomes
  private volWindowMs = 30_000;
  private minVolTicks = 50;

  // ── Exit parameters (base values, scaled by vol) ──
  private baseTakeProfitGap = 0.005;   // exit when bid >= fair - this (gap nearly closed)
  private baseTrailingStop = 0.03;     // exit if fair drops this much from high water
  private baseHardStop = 0.05;         // exit if price drops this much below entry
  private exitMinRemainingS = 15;      // hard exit before market close

  // ── Vol-scaling parameters ──
  private baselineVol = 0.0005;        // 0.05% — median observed vol (scaled to fraction)
  private minStopScale = 0.5;          // stops can't shrink below 50% of base
  private maxStopScale = 3.0;          // stops can't grow beyond 300% of base

  // ── Fair value model parameters ──
  private tDof = 4;                    // Student's t degrees of freedom (lower = fatter tails)

  // ── Per-market state ──
  private btcPriceAtOpen: number | null = null;
  private marketDurationMs = 300_000;
  private volTracker: VolTracker;
  private lastBtcTimestamp = 0;
  private lastLogTime = 0;
  private activeTrade: ActiveTrade | null = null;
  private tradesThisMarket = 0;
  private maxTradesPerMarket = Infinity; // unlimited for paper trading
  private lastExitTime = 0;
  private reEntryCooldownMs = 5_000;   // wait 5s after exit before re-entering

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
    this.activeTrade = null;
    this.tradesThisMarket = 0;
    this.lastExitTime = 0;

    if (this.btcPriceAtOpen) {
      console.log(
        `[GAP] Market open — BTC ref: ${this.btcPriceAtOpen.toFixed(2)}, ` +
        `duration: ${(this.marketDurationMs / 1000).toFixed(0)}s`
      );
    } else {
      console.log("[GAP] Market open — waiting for BTC price...");
    }
  }

  /** Compute vol scale factor: vol / baselineVol, clamped. */
  private getVolScale(): number {
    const vol = this.volTracker.realizedVol(this.marketDurationMs);
    if (vol <= 0) return 1;
    const raw = vol / this.baselineVol;
    return Math.max(this.minStopScale, Math.min(this.maxStopScale, raw));
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
        console.log(`[GAP] BTC ref captured: ${this.btcPriceAtOpen.toFixed(2)}`);
      }
      return [{ type: "NOOP" }];
    }

    const timeRemainingS = timeRemainingMs / 1000;

    // Hard exit before market close
    if (timeRemainingS < this.exitMinRemainingS) {
      return this.exitPosition(state, "end-of-window");
    }

    // Compute fair value
    const btcNow = btc?.price ?? this.btcPriceAtOpen;
    const btcReturn = (btcNow - this.btcPriceAtOpen) / this.btcPriceAtOpen;
    const timeElapsedMs = Math.max(1, this.marketDurationMs - timeRemainingMs);
    const timeElapsedFrac = Math.max(0.001, timeElapsedMs / this.marketDurationMs);
    const vol = this.volTracker.realizedVol(this.marketDurationMs);

    if (vol <= 0 || this.volTracker.tickCount < this.minVolTicks) {
      return [{ type: "NOOP" }];
    }

    const z = (btcReturn / vol) * Math.sqrt(timeElapsedFrac);

    // Student's t CDF for fatter tails — less confident about extreme outcomes
    // than normal distribution. ν=4 gives meaningful tail mass even at |z|>2.
    const fairUp = studentTCdf(z, this.tDof);
    const fairDown = 1 - fairUp;

    // Periodic logging
    const now = Date.now();
    if (now - this.lastLogTime > 10_000) {
      this.lastLogTime = now;
      const upBook = books[market.upTokenId];
      const downBook = books[market.downTokenId];
      const upAsk = upBook?.asks[0]?.price ?? 0;
      const downAsk = downBook?.asks[0]?.price ?? 0;
      const upGap = fairUp - upAsk;
      const downGap = fairDown - downAsk;

      let tradeStr = "none";
      if (this.activeTrade) {
        const pos = positions.find(p => p.assetId === this.activeTrade!.assetId);
        const currentSize = pos?.size ?? 0;
        const fair = this.activeTrade.assetId === market.upTokenId ? fairUp : fairDown;
        tradeStr = `${this.activeTrade.tokenLabel} ${currentSize}@${this.activeTrade.entryPrice.toFixed(3)} fair=${fair.toFixed(3)} hw=${this.activeTrade.highWaterFair.toFixed(3)}`;
      }

      const vs = this.getVolScale();
      console.log(
        `[GAP] fair=(${fairUp.toFixed(3)}/${fairDown.toFixed(3)}) ` +
        `gap_up=${upGap > 0 ? "+" : ""}${upGap.toFixed(3)} ` +
        `gap_dn=${downGap > 0 ? "+" : ""}${downGap.toFixed(3)} ` +
        `vol=${(vol * 100).toFixed(4)}% vs=${vs.toFixed(2)}x ` +
        `trade=[${tradeStr}] ` +
        `${timeRemainingS.toFixed(0)}s left`
      );
    }

    // If we have an active trade, check exit conditions
    if (this.activeTrade) {
      return this.evaluateExit(state, fairUp, fairDown);
    }

    // Otherwise, look for entry
    return this.evaluateEntry(state, fairUp, fairDown);
  }

  private evaluateEntry(state: MarketState, fairUp: number, fairDown: number): TradeAction[] {
    const { market, books, openOrders, timeRemainingMs } = state;

    // Don't enter new trades too close to end
    if (timeRemainingMs / 1000 < this.minRemainingS) return [{ type: "NOOP" }];

    // Cooldown after exit to avoid re-entering on same book state
    if (Date.now() - this.lastExitTime < this.reEntryCooldownMs) return [{ type: "NOOP" }];

    // Trade limit per market
    if (this.tradesThisMarket >= this.maxTradesPerMarket) return [{ type: "NOOP" }];

    // Don't trade if we already have pending orders
    if (openOrders.length > 0) return [{ type: "NOOP" }];

    const upBook = books[market.upTokenId];
    const downBook = books[market.downTokenId];

    // Compute gap for each side
    const upAsk = upBook?.asks[0]?.price;
    const downAsk = downBook?.asks[0]?.price;

    const upGap = upAsk != null ? fairUp - upAsk : -Infinity;
    const downGap = downAsk != null ? fairDown - downAsk : -Infinity;

    // Pick the side with the larger gap (if any meets threshold)
    let targetAssetId: string;
    let targetLabel: string;
    let targetFair: number;
    let targetAsk: number;

    if (upGap >= downGap && upGap >= this.minEntryGap) {
      // Don't buy near-certain outcomes (too risky if model is slightly wrong)
      if (fairUp > this.decidedThreshold || fairUp < (1 - this.decidedThreshold)) {
        return [{ type: "NOOP" }];
      }
      targetAssetId = market.upTokenId;
      targetLabel = "Up";
      targetFair = fairUp;
      targetAsk = upAsk!;
    } else if (downGap >= this.minEntryGap) {
      if (fairDown > this.decidedThreshold || fairDown < (1 - this.decidedThreshold)) {
        return [{ type: "NOOP" }];
      }
      targetAssetId = market.downTokenId;
      targetLabel = "Down";
      targetFair = fairDown;
      targetAsk = downAsk!;
    } else {
      return [{ type: "NOOP" }]; // No gap worth trading
    }

    // Check book has reasonable depth
    const book = books[targetAssetId];
    if (!book || book.asks.length === 0 || book.bids.length === 0) return [{ type: "NOOP" }];
    const spread = book.asks[0]!.price - book.bids[0]!.price;
    if (spread > 0.08) return [{ type: "NOOP" }]; // Book too thin

    // Compute vol-scaled hard stop for entry check
    const volScale = this.getVolScale();
    const hardStop = this.baseHardStop * volScale;

    // Don't enter if bid is already at or below our hard stop
    // (spread wider than stop loss = instant stop-out)
    const bestBidEntry = book.bids[0]!.price;
    if (bestBidEntry <= targetAsk - hardStop) return [{ type: "NOOP" }];

    // Place a limit buy at the ask (aggressive — we want the fill)
    const tickSize = state.market.tickSize || 0.01;
    const bidPrice = Math.round(targetAsk / tickSize) * tickSize;

    console.log(
      `[GAP] ENTRY: ${targetLabel} ask=${targetAsk.toFixed(3)} fair=${targetFair.toFixed(3)} ` +
      `gap=${(targetFair - targetAsk).toFixed(3)} vs=${volScale.toFixed(2)}x ` +
      `stops: trail=${(this.baseTrailingStop * volScale).toFixed(3)} hard=${hardStop.toFixed(3)}`
    );

    this.activeTrade = {
      assetId: targetAssetId,
      tokenLabel: targetLabel,
      entryPrice: bidPrice,
      entryFairValue: targetFair,
      size: this.orderSize,
      highWaterFair: targetFair,
      volScale,
    };
    this.tradesThisMarket++;

    return [{
      type: "PLACE_ORDER",
      assetId: targetAssetId,
      side: "BUY",
      price: bidPrice,
      size: this.orderSize,
      orderType: "GTC",
    }];
  }

  private evaluateExit(state: MarketState, fairUp: number, fairDown: number): TradeAction[] {
    const trade = this.activeTrade!;
    const { market, books, positions, openOrders } = state;

    const pos = positions.find(p => p.assetId === trade.assetId);
    const currentSize = pos?.size ?? 0;

    // If we have no position (order didn't fill or was cancelled), reset
    if (currentSize === 0 && openOrders.length === 0) {
      this.activeTrade = null;
      return [{ type: "NOOP" }];
    }

    // If order is still pending (not filled), check if gap closed before fill
    if (currentSize === 0 && openOrders.length > 0) {
      const fair = trade.assetId === market.upTokenId ? fairUp : fairDown;
      const book = books[trade.assetId];
      const bestAsk = book?.asks[0]?.price ?? 1;
      const gap = fair - bestAsk;

      // Cancel if gap closed or reversed
      if (gap < this.minEntryGap * 0.5) {
        console.log(`[GAP] Gap closed before fill, cancelling entry`);
        this.activeTrade = null;
        return [{ type: "CANCEL_ALL" }];
      }
      return [{ type: "NOOP" }];
    }

    // We have a position — check exit conditions
    const fair = trade.assetId === market.upTokenId ? fairUp : fairDown;
    const book = books[trade.assetId];
    const bestBid = book?.bids[0]?.price ?? 0;

    // Update high water mark
    if (fair > trade.highWaterFair) {
      trade.highWaterFair = fair;
    }

    // Vol-scaled stops (locked to entry-time vol scale)
    const vs = trade.volScale;
    const takeProfitGap = this.baseTakeProfitGap * vs;
    const trailingStop = this.baseTrailingStop * vs;
    const hardStop = this.baseHardStop * vs;

    let exitReason: string | null = null;

    // 1. Take profit: gap has closed
    if (bestBid >= fair - takeProfitGap) {
      exitReason = `take-profit (bid=${bestBid.toFixed(3)} >= fair-tp=${(fair - takeProfitGap).toFixed(3)})`;
    }

    // 2. Trailing stop: fair value dropped from high water
    if (!exitReason && trade.highWaterFair - fair >= trailingStop) {
      exitReason = `trailing-stop (fair=${fair.toFixed(3)} dropped ${(trade.highWaterFair - fair).toFixed(3)} from hw=${trade.highWaterFair.toFixed(3)}, trail=${trailingStop.toFixed(3)})`;
    }

    // 3. Hard stop loss: price dropped below entry
    if (!exitReason && bestBid <= trade.entryPrice - hardStop) {
      exitReason = `hard-stop (bid=${bestBid.toFixed(3)} <= entry-stop=${(trade.entryPrice - hardStop).toFixed(3)})`;
    }

    if (exitReason) {
      return this.exitPosition(state, exitReason);
    }

    return [{ type: "NOOP" }];
  }

  private exitPosition(state: MarketState, reason: string): TradeAction[] {
    const trade = this.activeTrade;
    const actions: TradeAction[] = [];

    // Cancel any open orders
    if (state.openOrders.length > 0) {
      actions.push({ type: "CANCEL_ALL" });
    }

    if (!trade) return actions.length > 0 ? actions : [{ type: "NOOP" }];

    const pos = state.positions.find(p => p.assetId === trade.assetId);
    const currentSize = pos?.size ?? 0;

    if (currentSize > 0 && pos?.settled) {
      const book = state.books[trade.assetId];
      const bestBid = book?.bids[0]?.price ?? 0;
      const tickSize = state.market.tickSize || 0.01;

      // Sell at best bid minus small discount to ensure fill
      const sellPrice = Math.max(tickSize, Math.round((bestBid - 0.01) / tickSize) * tickSize);

      const pnl = (sellPrice - trade.entryPrice) * currentSize;
      console.log(
        `[GAP] EXIT (${reason}): ${trade.tokenLabel} ${currentSize}@${sellPrice.toFixed(3)} ` +
        `entry=${trade.entryPrice.toFixed(3)} pnl=${pnl >= 0 ? "+" : ""}${pnl.toFixed(3)}`
      );

      actions.push({
        type: "PLACE_ORDER",
        assetId: trade.assetId,
        side: "SELL",
        price: sellPrice,
        size: currentSize,
        orderType: "GTC",
      });
    }

    this.activeTrade = null;
    this.lastExitTime = Date.now();
    return actions.length > 0 ? actions : [{ type: "NOOP" }];
  }

  async onMarketClose(state: MarketState): Promise<TradeAction[]> {
    if (this.activeTrade) {
      console.log(`[GAP] Market close — force exit`);
    } else {
      console.log(`[GAP] Market close — no position`);
    }
    console.log(`[GAP] Trades this market: ${this.tradesThisMarket}`);
    return this.exitPosition(state, "market-close");
  }
}

registerStrategy("gap-trader", () => new GapTrader());
