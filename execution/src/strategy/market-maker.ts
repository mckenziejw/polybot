import type { MarketState, TradeAction, Side, DataMode } from "../types.ts";
import type { Strategy } from "./strategy.ts";
import { registerStrategy } from "./strategy.ts";

// ── Normal CDF (Abramowitz & Stegun rational approximation, |error| < 7.5e-8) ──

function normalCdf(x: number): number {
  if (x < -8) return 0;
  if (x > 8) return 1;

  const a1 = 0.319381530;
  const a2 = -0.356563782;
  const a3 = 1.781477937;
  const a4 = -1.821255978;
  const a5 = 1.330274429;
  const p = 0.2316419;

  const sign = x < 0 ? -1 : 1;
  const absX = Math.abs(x);
  const t = 1 / (1 + p * absX);
  const pdf = Math.exp(-0.5 * absX * absX) / Math.sqrt(2 * Math.PI);
  const cdf = 1 - pdf * t * (a1 + t * (a2 + t * (a3 + t * (a4 + t * a5))));

  return sign < 0 ? 1 - cdf : cdf;
}

// ── Rolling volatility tracker ──

interface PriceTick {
  t: number; // timestamp ms
  p: number; // price
}

class VolTracker {
  private ticks: PriceTick[] = [];
  private windowMs: number;
  private maxTicks: number;

  constructor(windowMs: number = 30_000, maxTicks: number = 6000) {
    this.windowMs = windowMs;
    this.maxTicks = maxTicks;
  }

  reset(): void {
    this.ticks = [];
  }

  push(timestamp: number, price: number): void {
    this.ticks.push({ t: timestamp, p: price });

    // Evict old ticks
    const cutoff = timestamp - this.windowMs;
    while (this.ticks.length > 0 && this.ticks[0]!.t < cutoff) {
      this.ticks.shift();
    }
    // Hard cap
    if (this.ticks.length > this.maxTicks) {
      this.ticks = this.ticks.slice(-this.maxTicks);
    }
  }

  get tickCount(): number {
    return this.ticks.length;
  }

  /**
   * Compute realized volatility scaled to the given duration.
   * Returns the expected standard deviation of returns over `scaleDurationMs`.
   *
   * Uses sum-of-squared-log-returns approach:
   *   variance_per_ms = sum(r_i^2) / windowDurationMs
   *   vol = sqrt(variance_per_ms * scaleDurationMs)
   */
  realizedVol(scaleDurationMs: number): number {
    if (this.ticks.length < 10) return 0;

    let sumSqReturns = 0;
    for (let i = 1; i < this.ticks.length; i++) {
      const r = Math.log(this.ticks[i]!.p / this.ticks[i - 1]!.p);
      sumSqReturns += r * r;
    }

    const windowDurationMs =
      this.ticks[this.ticks.length - 1]!.t - this.ticks[0]!.t;
    if (windowDurationMs <= 0) return 0;

    const variancePerMs = sumSqReturns / windowDurationMs;
    return Math.sqrt(variancePerMs * scaleDurationMs);
  }
}

// ── Smart Market Maker Strategy ──

interface QuoteTarget {
  bidPrice: number;
  askPrice: number;
  bidSize: number;
  askSize: number;
}

export class SmartMarketMaker implements Strategy {
  readonly id = "market-maker";
  readonly dataMode: DataMode = "tick";

  // ── Parameters ──
  private orderSize = 5;
  private maxPositionPerSide = 50;
  private maxNetExposure = 30;
  private halfSpreadBase = 0.03;
  private volSpreadScale = 0.5;
  private requoteThreshold = 0.01;
  private minRemainingS = 15;
  private flattenStartS = 60;       // start exiting positions with this many seconds left
  private flattenDiscountPct = 0.02; // sell discount below fair value to ensure exit fills
  private skewPerToken = 0.002;
  private volWindowMs = 30_000;
  private minVolTicks = 50;

  // Liquidity guards
  private minBookDepth = 20;       // minimum $ size on each side to consider a book "healthy"
  private maxSpread = 0.08;        // max bid-ask spread — wider means too thin to trade
  private decidedThreshold = 0.85; // fair value beyond this = market is decided, stop quoting

  // ── Per-market state ──
  private btcPriceAtOpen: number | null = null;
  private marketDurationMs = 300_000;
  private volTracker: VolTracker;
  private lastBtcTimestamp = 0;
  private lastLogTime = 0;

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

    if (this.btcPriceAtOpen) {
      console.log(
        `[MM] Market open — BTC ref: ${this.btcPriceAtOpen.toFixed(2)}, duration: ${(this.marketDurationMs / 1000).toFixed(0)}s`
      );
    } else {
      console.log("[MM] Market open — waiting for BTC price...");
    }
  }

  async evaluate(state: MarketState): Promise<TradeAction[]> {
    const { market, books, openOrders, positions, timeRemainingMs } = state;
    const btc = state.externalData?.["binance-btcusdt"];

    // Feed BTC ticks to vol tracker (deduplicate by timestamp)
    if (btc && btc.timestamp > this.lastBtcTimestamp) {
      this.volTracker.push(btc.timestamp, btc.price);
      this.lastBtcTimestamp = btc.timestamp;
    }

    // Capture reference price if we don't have one yet
    if (!this.btcPriceAtOpen) {
      this.btcPriceAtOpen = btc?.price ?? null;
      if (this.btcPriceAtOpen) {
        console.log(`[MM] BTC ref captured: ${this.btcPriceAtOpen.toFixed(2)}`);
      }
      return [{ type: "NOOP" }];
    }

    const timeRemainingS = timeRemainingMs / 1000;

    // ── Last N seconds: cancel everything ──
    if (timeRemainingS < this.minRemainingS) {
      if (openOrders.length > 0) {
        console.log(`[MM] End of window (${timeRemainingS.toFixed(0)}s left) — cancelling all`);
        return [{ type: "CANCEL_ALL" }];
      }
      return [{ type: "NOOP" }];
    }

    // ── Flatten phase: exit positions before expiry ──
    if (timeRemainingS < this.flattenStartS) {
      return this.evaluateFlatten(state);
    }

    // ── Compute fair value ──
    const btcNow = btc?.price ?? this.btcPriceAtOpen;
    const btcReturn = (btcNow - this.btcPriceAtOpen) / this.btcPriceAtOpen;
    const timeElapsedMs = Math.max(1, this.marketDurationMs - timeRemainingMs);
    const timeElapsedFrac = Math.max(0.001, timeElapsedMs / this.marketDurationMs);

    // Realized vol scaled to full market window
    const vol = this.volTracker.realizedVol(this.marketDurationMs);

    // Not enough data for vol estimate
    if (vol <= 0 || this.volTracker.tickCount < this.minVolTicks) {
      return [{ type: "NOOP" }];
    }

    // Binary option fair value: Φ(z) where z = (return / vol) * sqrt(timeElapsed/totalTime)
    const z = (btcReturn / vol) * Math.sqrt(timeElapsedFrac);
    const fairUp = normalCdf(z);
    const fairDown = 1 - fairUp;

    // ── Inventory ──
    const upPos = positions.find((p) => p.assetId === market.upTokenId);
    const downPos = positions.find((p) => p.assetId === market.downTokenId);
    const upSize = upPos?.size ?? 0;
    const downSize = downPos?.size ?? 0;
    const netInventory = upSize - downSize;

    // ── Spread computation ──
    // Time-based spread multiplier
    let spreadMultiplier = 1.0;
    if (timeElapsedFrac < 0.2) {
      // Early window: wide spreads, gathering information
      spreadMultiplier = 2.0;
    } else if (timeElapsedFrac > 0.8) {
      // Late window: tighter spreads
      spreadMultiplier = 0.6;
    }

    const halfSpread = Math.max(
      0.01,
      (this.halfSpreadBase + vol * this.volSpreadScale) * spreadMultiplier
    );
    const inventorySkew = netInventory * this.skewPerToken;

    const tickSize = market.tickSize || 0.01;
    const round = (p: number) =>
      Math.round(p / tickSize) * tickSize;
    const clamp = (p: number) =>
      Math.max(tickSize, Math.min(1 - tickSize, round(p)));

    // ── Periodic logging ──
    const now = Date.now();
    if (now - this.lastLogTime > 10_000) {
      this.lastLogTime = now;
      console.log(
        `[MM] fair=(${fairUp.toFixed(3)}/${fairDown.toFixed(3)}) ` +
        `btc=${btcNow.toFixed(0)} ret=${(btcReturn * 100).toFixed(4)}% ` +
        `vol=${(vol * 100).toFixed(4)}% z=${z.toFixed(3)} ` +
        `spread=${(halfSpread * 2).toFixed(3)} inv=${netInventory.toFixed(0)} ` +
        `elapsed=${(timeElapsedFrac * 100).toFixed(0)}% ` +
        `ticks=${this.volTracker.tickCount}`
      );
    }

    // ── Market decided check: stop quoting when outcome is near-certain ──
    if (fairUp > this.decidedThreshold || fairDown > this.decidedThreshold) {
      // Market is decided — cancel all quotes, don't add new exposure
      if (openOrders.length > 0) {
        if (now - this.lastLogTime > 10_000) {
          this.lastLogTime = now;
          console.log(
            `[MM] Market decided (fair=${fairUp.toFixed(3)}/${fairDown.toFixed(3)}) — pulling quotes`
          );
        }
        return [{ type: "CANCEL_ALL" }];
      }
      return [{ type: "NOOP" }];
    }

    // ── Liquidity checks per token ──
    const upBook = books[market.upTokenId];
    const downBook = books[market.downTokenId];
    const upHealthy = this.isBookHealthy(upBook);
    const downHealthy = this.isBookHealthy(downBook);

    // If neither book is healthy, pull everything
    if (!upHealthy && !downHealthy) {
      if (openOrders.length > 0) {
        return [{ type: "CANCEL_ALL" }];
      }
      return [{ type: "NOOP" }];
    }

    // ── Compute target quotes ──
    // Don't place new buy orders on unhealthy books (sells to exit positions are always ok)
    const upTarget: QuoteTarget = {
      bidPrice: clamp(fairUp - halfSpread - inventorySkew),
      askPrice: clamp(fairUp + halfSpread - inventorySkew),
      bidSize: upHealthy && upSize < this.maxPositionPerSide ? this.orderSize : 0,
      askSize: upSize > 0 && upPos?.settled ? Math.min(this.orderSize, upSize) : 0,
    };

    const downTarget: QuoteTarget = {
      bidPrice: clamp(fairDown - halfSpread + inventorySkew),
      askPrice: clamp(fairDown + halfSpread + inventorySkew),
      bidSize: downHealthy && downSize < this.maxPositionPerSide ? this.orderSize : 0,
      askSize: downSize > 0 && downPos?.settled ? Math.min(this.orderSize, downSize) : 0,
    };

    // Net exposure check
    if (Math.abs(netInventory) >= this.maxNetExposure) {
      // Only allow orders that reduce exposure
      if (netInventory > 0) {
        upTarget.bidSize = 0; // Don't buy more Up
        downTarget.askSize = 0; // Don't sell Down (that would increase net Up)
      } else {
        downTarget.bidSize = 0;
        upTarget.askSize = 0;
      }
    }

    // ── Reconcile with existing orders ──
    return this.reconcileQuotes(state, market.upTokenId, upTarget, market.downTokenId, downTarget);
  }

  /**
   * Flatten phase: actively exit all positions before market close.
   * Cancel all buy orders. Place aggressive sell orders for held tokens.
   */
  private evaluateFlatten(state: MarketState): TradeAction[] {
    const { market, books, openOrders, positions, timeRemainingMs } = state;
    const actions: TradeAction[] = [];

    // Cancel all existing orders first
    if (openOrders.length > 0) {
      actions.push({ type: "CANCEL_ALL" });
    }

    const upPos = positions.find((p) => p.assetId === market.upTokenId);
    const downPos = positions.find((p) => p.assetId === market.downTokenId);
    const upSize = upPos?.size ?? 0;
    const downSize = downPos?.size ?? 0;

    if (upSize === 0 && downSize === 0) {
      const now = Date.now();
      if (now - this.lastLogTime > 10_000) {
        this.lastLogTime = now;
        console.log(`[MM] Flatten phase — no positions to exit (${(timeRemainingMs / 1000).toFixed(0)}s left)`);
      }
      return actions.length > 0 ? actions : [{ type: "NOOP" }];
    }

    const tickSize = market.tickSize || 0.01;
    const clamp = (p: number) =>
      Math.max(tickSize, Math.min(1 - tickSize, Math.round(p / tickSize) * tickSize));

    // Sell Up tokens if we hold any
    if (upSize > 0 && upPos?.settled) {
      const upBook = books[market.upTokenId];
      const bestBid = upBook?.bids[0]?.price ?? 0;
      if (bestBid > 0) {
        // Sell at best bid minus discount to ensure fill
        const exitPrice = clamp(bestBid - this.flattenDiscountPct);
        if (exitPrice > tickSize) {
          actions.push({
            type: "PLACE_ORDER",
            assetId: market.upTokenId,
            side: "SELL",
            price: exitPrice,
            size: Math.min(upSize, this.orderSize),
            orderType: "GTC",
          });
        }
      }
    }

    // Sell Down tokens if we hold any
    if (downSize > 0 && downPos?.settled) {
      const downBook = books[market.downTokenId];
      const bestBid = downBook?.bids[0]?.price ?? 0;
      if (bestBid > 0) {
        const exitPrice = clamp(bestBid - this.flattenDiscountPct);
        if (exitPrice > tickSize) {
          actions.push({
            type: "PLACE_ORDER",
            assetId: market.downTokenId,
            side: "SELL",
            price: exitPrice,
            size: Math.min(downSize, this.orderSize),
            orderType: "GTC",
          });
        }
      }
    }

    const now = Date.now();
    if (now - this.lastLogTime > 5_000) {
      this.lastLogTime = now;
      console.log(
        `[MM] Flatten phase (${(timeRemainingMs / 1000).toFixed(0)}s left) — ` +
        `Up=${upSize.toFixed(0)} Down=${downSize.toFixed(0)}`
      );
    }

    return actions.length > 0 ? actions : [{ type: "NOOP" }];
  }

  async onMarketClose(_state: MarketState): Promise<TradeAction[]> {
    console.log("[MM] Market close — cancelling all orders");
    return [{ type: "CANCEL_ALL" }];
  }

  /**
   * Check if an orderbook has sufficient two-sided liquidity for market making.
   * Returns false if:
   *  - Book is missing or empty on either side
   *  - Spread exceeds maxSpread
   *  - Top-of-book size is below minBookDepth on either side
   */
  private isBookHealthy(book: import("../types.ts").OrderBookState | undefined): boolean {
    if (!book) return false;
    if (book.bids.length === 0 || book.asks.length === 0) return false;

    const spread = book.asks[0]!.price - book.bids[0]!.price;
    if (spread > this.maxSpread) return false;

    // Check depth: sum of top 3 levels on each side
    const bidDepth = book.bids.slice(0, 3).reduce((s, l) => s + l.size * l.price, 0);
    const askDepth = book.asks.slice(0, 3).reduce((s, l) => s + l.size * l.price, 0);
    if (bidDepth < this.minBookDepth || askDepth < this.minBookDepth) return false;

    return true;
  }

  /**
   * Compare target quotes to existing open orders.
   * Cancel orders that have drifted beyond threshold, place missing quotes.
   */
  private reconcileQuotes(
    state: MarketState,
    upTokenId: string,
    upTarget: QuoteTarget,
    downTokenId: string,
    downTarget: QuoteTarget
  ): TradeAction[] {
    const actions: TradeAction[] = [];
    const { openOrders } = state;

    // Find existing orders by asset + side
    const existingUpBid = openOrders.find(
      (o) => o.assetId === upTokenId && o.side === "BUY"
    );
    const existingUpAsk = openOrders.find(
      (o) => o.assetId === upTokenId && o.side === "SELL"
    );
    const existingDownBid = openOrders.find(
      (o) => o.assetId === downTokenId && o.side === "BUY"
    );
    const existingDownAsk = openOrders.find(
      (o) => o.assetId === downTokenId && o.side === "SELL"
    );

    // Helper: check if an order needs requoting
    const needsRequote = (
      existing: typeof existingUpBid,
      targetPrice: number,
      targetSize: number
    ): boolean => {
      if (!existing && targetSize > 0) return true; // Need to place
      if (existing && targetSize <= 0) return true; // Need to cancel
      if (existing && Math.abs(existing.price - targetPrice) > this.requoteThreshold) {
        return true; // Price drifted
      }
      return false;
    };

    // Process each quote leg
    const legs: Array<{
      existing: typeof existingUpBid;
      targetPrice: number;
      targetSize: number;
      assetId: string;
      side: Side;
    }> = [
      { existing: existingUpBid, targetPrice: upTarget.bidPrice, targetSize: upTarget.bidSize, assetId: upTokenId, side: "BUY" },
      { existing: existingUpAsk, targetPrice: upTarget.askPrice, targetSize: upTarget.askSize, assetId: upTokenId, side: "SELL" },
      { existing: existingDownBid, targetPrice: downTarget.bidPrice, targetSize: downTarget.bidSize, assetId: downTokenId, side: "BUY" },
      { existing: existingDownAsk, targetPrice: downTarget.askPrice, targetSize: downTarget.askSize, assetId: downTokenId, side: "SELL" },
    ];

    for (const leg of legs) {
      if (!needsRequote(leg.existing, leg.targetPrice, leg.targetSize)) {
        continue;
      }

      // Cancel existing if present
      if (leg.existing) {
        actions.push({ type: "CANCEL_ORDER", orderId: leg.existing.orderId });
      }

      // Place new if target size > 0
      if (leg.targetSize > 0) {
        actions.push({
          type: "PLACE_ORDER",
          assetId: leg.assetId,
          side: leg.side,
          price: leg.targetPrice,
          size: leg.targetSize,
          orderType: "GTC",
        });
      }
    }

    return actions.length > 0 ? actions : [{ type: "NOOP" }];
  }
}

// Register with the strategy registry
registerStrategy("market-maker", () => new SmartMarketMaker());
