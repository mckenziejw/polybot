import type {
  MarketState,
  TradeAction,
  DataMode,
  Side,
  ExternalDataPoint,
  OrderBookState,
} from "../types.ts";
import type { Strategy } from "./strategy.ts";
import { registerStrategy } from "./strategy.ts";
import { loadConfig } from "../config.ts";

/**
 * Conditional Entropy + Binance CEX Price Signal Strategy
 *
 * Architecture (two independent sub-problems):
 *   1. WHEN to trade: conditional entropy regime detection — when H(alt|others)
 *      drops below threshold, cross-asset coupling is tight and alt outcomes
 *      are predictable from BTC direction.
 *   2. WHICH direction: Binance BTC spot price return over first N seconds
 *      of the market window. BTC price movement on CEX predicts BTC binary
 *      outcome, which propagates to alts via cross-asset coupling.
 *
 * Lifecycle per market:
 *   - onMarketOpen: reset per-market state, record BTC open price
 *   - evaluate (WAITING phase): accumulate BTC price ticks, wait for signal delay
 *   - evaluate (SIGNAL phase): once delay elapsed, compute BTC return.
 *     If |return| >= threshold and entropy regime is active, place order on
 *     the target alt token at current ask.
 *   - Hold to binary settlement.
 *
 * The entropy regime is maintained as a rolling window across markets — it
 * tracks the joint outcomes of recent BTC/ETH/SOL/XRP markets and fires
 * when conditional entropy drops below threshold.
 *
 * Config:
 *   execution.strategy_id: "entropy-binance"
 *   execution.external_feeds: ["binance-btcusdt"]
 *   execution.bet_dollars: dollar amount per trade (default 10)
 */

// ── Configuration constants ──

/** Seconds after market open to wait before reading BTC return */
const SIGNAL_DELAY_S = 15;

/** Minimum |BTC return| to generate a signal (filters noise) */
const MIN_BTC_RETURN = 0.0003;

/** Entry price cap — don't buy tokens priced above this */
const ENTRY_PRICE_CAP = 0.85;

/** Entry price floor — don't buy tokens priced below this (no edge at ~0.50) */
const ENTRY_PRICE_FLOOR = 0.10;

/** Minimum time remaining in market to place an order (ms) */
const MIN_REMAINING_MS = 180_000; // 3 minutes

/** Polymarket minimum order size */
const MIN_TOKENS = 5;

/** Maximum spread to enter (reject illiquid books) */
const MAX_SPREAD = 0.08;

// ── Entropy regime parameters ──

/** Rolling window size for entropy computation */
const ENTROPY_WINDOW = 50;

/** Conditional entropy threshold — trade when H < this */
const ENTROPY_THRESHOLD = 0.5;

/** Number of assets tracked for joint outcomes */
const ASSETS = ["btc", "eth", "sol", "xrp"] as const;
const ALT_ASSETS = ["eth", "sol", "xrp"] as const;
type Asset = (typeof ASSETS)[number];

// ── Types ──

interface MarketOutcome {
  epoch: number;
  btc: 0 | 1;  // 0 = Down, 1 = Up
  eth: 0 | 1;
  sol: 0 | 1;
  xrp: 0 | 1;
}

type Phase =
  | "WAITING"    // waiting for signal delay
  | "DECIDING"   // signal delay reached, evaluating
  | "TRADED"     // order placed, holding to settlement
  | "SKIP";      // not trading this market

export class EntropyBinanceStrategy implements Strategy {
  readonly id = "entropy-binance";
  readonly dataMode: DataMode = "tick";

  // ── Per-market state (reset each open) ──
  private phase: Phase = "SKIP";
  private marketOpenMs = 0;
  private btcOpenPrice = 0;       // BTC mid at market open
  private btcSignalPrice = 0;     // BTC mid at signal time
  private decisionMade = false;
  private currentAsset: Asset = "btc"; // which asset this market is for

  // ── Config ──
  private betDollars: number;
  private spotFeedName: string;

  // ── Persistent across markets: outcome history for entropy ──
  private outcomeHistory: MarketOutcome[] = [];

  /**
   * Tracks the last BTC binary outcome we observed, keyed by epoch.
   * Populated from market resolutions we observe.
   * For alts, we track separately.
   */
  private btcLastOutcome: 0 | 1 | null = null;
  private pendingOutcomes: Map<number, Partial<Record<Asset, 0 | 1>>> = new Map();

  constructor(betDollars = 10, spotFeedName = "binance-btcusdt") {
    this.betDollars = betDollars;
    this.spotFeedName = spotFeedName;
  }

  // ── Lifecycle ──

  async onMarketOpen(state: MarketState): Promise<void> {
    this.phase = "WAITING";
    this.marketOpenMs = Date.now();
    this.btcOpenPrice = 0;
    this.btcSignalPrice = 0;
    this.decisionMade = false;

    // Determine which asset this market is for from the slug
    const slug = state.market.slug.toLowerCase();
    if (slug.includes("eth")) this.currentAsset = "eth";
    else if (slug.includes("sol")) this.currentAsset = "sol";
    else if (slug.includes("xrp")) this.currentAsset = "xrp";
    else if (slug.includes("btc")) this.currentAsset = "btc";
    else this.currentAsset = "btc";

    // Record BTC open price from external feed
    const spot = state.externalData?.[this.spotFeedName];
    if (spot) {
      this.btcOpenPrice = spot.price;
    }

    const entropyInfo = this.getEntropyInfo();
    console.log(
      `[Entropy] Market open: ${state.market.slug} (asset=${this.currentAsset}), ` +
      `BTC open=$${this.btcOpenPrice.toFixed(2)}, ` +
      `outcome history=${this.outcomeHistory.length}, ` +
      `${entropyInfo}`
    );
  }

  async evaluate(state: MarketState): Promise<TradeAction[]> {
    const { market, books, externalData, timeRemainingMs } = state;
    const now = Date.now();
    const elapsedMs = now - this.marketOpenMs;
    const elapsedS = elapsedMs / 1000;

    // Always try to capture BTC open price if we missed it
    if (this.btcOpenPrice === 0) {
      const spot = externalData?.[this.spotFeedName];
      if (spot) {
        this.btcOpenPrice = spot.price;
      }
    }

    if (this.phase === "SKIP" || this.phase === "TRADED") {
      return [{ type: "NOOP" }];
    }

    // ── WAITING: accumulate until signal delay ──
    if (this.phase === "WAITING" && elapsedS >= SIGNAL_DELAY_S && !this.decisionMade) {
      this.decisionMade = true;
      this.phase = "DECIDING";

      // Get BTC price at signal time
      const spot = externalData?.[this.spotFeedName];
      if (!spot || this.btcOpenPrice === 0) {
        console.log(`[Entropy] No BTC price data — skipping`);
        this.phase = "SKIP";
        return [{ type: "NOOP" }];
      }
      this.btcSignalPrice = spot.price;

      // Compute BTC return
      const btcReturn = (this.btcSignalPrice - this.btcOpenPrice) / this.btcOpenPrice;

      // Apply magnitude filter
      if (Math.abs(btcReturn) < MIN_BTC_RETURN) {
        console.log(
          `[Entropy] BTC return too small: ${(btcReturn * 100).toFixed(4)}% ` +
          `(threshold: ${(MIN_BTC_RETURN * 100).toFixed(4)}%) — skipping`
        );
        this.phase = "SKIP";
        return [{ type: "NOOP" }];
      }

      // Check entropy regime
      if (!this.isLowEntropyRegime()) {
        console.log(`[Entropy] Not in low-entropy regime — skipping`);
        this.phase = "SKIP";
        return [{ type: "NOOP" }];
      }

      // Determine direction: BTC price up → buy Up token, down → buy Down token
      const direction: "Up" | "Down" = btcReturn > 0 ? "Up" : "Down";

      // For BTC markets, we trade BTC directly.
      // For alt markets (when entropy says alt is tightly coupled), trade the alt.
      const buyUpToken = direction === "Up";
      const buyAssetId = buyUpToken ? market.upTokenId : market.downTokenId;

      // Check time remaining
      if (timeRemainingMs < MIN_REMAINING_MS) {
        console.log(
          `[Entropy] Insufficient time remaining ` +
          `(${(timeRemainingMs / 1000).toFixed(0)}s) — skipping`
        );
        this.phase = "SKIP";
        return [{ type: "NOOP" }];
      }

      // Get current book for the target token
      const book = books[buyAssetId];
      if (!book || book.asks.length === 0 || book.bids.length === 0) {
        console.log(`[Entropy] No book for ${direction} token — skipping`);
        this.phase = "SKIP";
        return [{ type: "NOOP" }];
      }

      const bestAsk = book.asks[0]!.price;
      const bestBid = book.bids[0]!.price;
      const spread = bestAsk - bestBid;

      if (spread > MAX_SPREAD) {
        console.log(
          `[Entropy] Spread too wide (${spread.toFixed(3)}) — skipping`
        );
        this.phase = "SKIP";
        return [{ type: "NOOP" }];
      }

      if (bestAsk < ENTRY_PRICE_FLOOR || bestAsk > ENTRY_PRICE_CAP) {
        console.log(
          `[Entropy] Ask ${bestAsk.toFixed(3)} outside range ` +
          `[${ENTRY_PRICE_FLOOR}, ${ENTRY_PRICE_CAP}] — skipping`
        );
        this.phase = "SKIP";
        return [{ type: "NOOP" }];
      }

      // Size the order
      const rawSize = Math.floor(this.betDollars / bestAsk);
      const size = Math.max(rawSize, MIN_TOKENS);

      this.phase = "TRADED";
      console.log(
        `[Entropy] BUY ${size} ${direction} @ ${bestAsk.toFixed(3)} ` +
        `(BTC ret=${(btcReturn * 100).toFixed(4)}%, ` +
        `open=$${this.btcOpenPrice.toFixed(2)}, ` +
        `signal=$${this.btcSignalPrice.toFixed(2)}, ` +
        `delay=${elapsedS.toFixed(1)}s, ` +
        `cost=$${(size * bestAsk).toFixed(2)})`
      );

      return [
        {
          type: "PLACE_ORDER",
          assetId: buyAssetId,
          side: "BUY" as Side,
          price: bestAsk,
          size,
          orderType: "GTC",
        },
      ];
    }

    return [{ type: "NOOP" }];
  }

  async onMarketClose(state: MarketState): Promise<TradeAction[]> {
    // Record the outcome for entropy tracking
    this.recordMarketOutcome(state);

    console.log(
      `[Entropy] Market close (phase=${this.phase}, ` +
      `asset=${this.currentAsset}, ` +
      `outcomes=${this.outcomeHistory.length})`
    );

    this.phase = "SKIP";
    return [{ type: "CANCEL_ALL" }];
  }

  onFill(assetId: string, side: Side, size: number, price: number): void {
    console.log(
      `[Entropy] Fill: ${side} ${size} @ ${price.toFixed(3)} (asset=${assetId})`
    );
  }

  getState(): Record<string, unknown> {
    const targetAlt = this.findLowestEntropyAlt();
    return {
      phase: this.phase,
      currentAsset: this.currentAsset,
      btcOpenPrice: this.btcOpenPrice,
      btcSignalPrice: this.btcSignalPrice,
      outcomeHistoryLength: this.outcomeHistory.length,
      entropyRegimeActive: this.isLowEntropyRegime(),
      lowestEntropyAlt: targetAlt?.asset ?? null,
      lowestEntropy: targetAlt?.entropy ?? null,
      signalDelayS: SIGNAL_DELAY_S,
      minBtcReturn: MIN_BTC_RETURN,
    };
  }

  // ── Entropy regime detection ──

  /**
   * Record the outcome of the market that just closed.
   * We infer the winner from the final book state: if the Up token bid > 0.80,
   * the market resolved Up (and vice versa).
   */
  private recordMarketOutcome(state: MarketState): void {
    const { market, books } = state;
    const slug = state.market.slug.toLowerCase();

    // Extract epoch from slug (last segment)
    const parts = slug.split("-");
    const epochStr = parts[parts.length - 1];
    if (!epochStr) return;
    const epoch = parseInt(epochStr, 10);
    if (isNaN(epoch)) return;

    // Determine winner from book state
    const upBook = books[market.upTokenId];
    const downBook = books[market.downTokenId];

    let winner: 0 | 1 | null = null;
    if (upBook && upBook.bids.length > 0) {
      const upBid = upBook.bids[0]!.price;
      if (upBid > 0.80) winner = 1; // Up won
      else if (upBid < 0.20) winner = 0; // Down won
    }
    if (winner === null && downBook && downBook.bids.length > 0) {
      const downBid = downBook.bids[0]!.price;
      if (downBid > 0.80) winner = 0; // Down won (= Up lost)
      else if (downBid < 0.20) winner = 1; // Up won
    }

    if (winner === null) return;

    // Store in pending outcomes map
    if (!this.pendingOutcomes.has(epoch)) {
      this.pendingOutcomes.set(epoch, {});
    }
    const pending = this.pendingOutcomes.get(epoch)!;
    pending[this.currentAsset] = winner;

    // Check if we have all 4 assets for this epoch
    if (
      pending.btc !== undefined &&
      pending.eth !== undefined &&
      pending.sol !== undefined &&
      pending.xrp !== undefined
    ) {
      this.outcomeHistory.push({
        epoch,
        btc: pending.btc,
        eth: pending.eth,
        sol: pending.sol,
        xrp: pending.xrp,
      });

      this.pendingOutcomes.delete(epoch);

      // Keep only the last 200 outcomes (we only need ENTROPY_WINDOW)
      if (this.outcomeHistory.length > 200) {
        this.outcomeHistory = this.outcomeHistory.slice(-200);
      }
    }

    // Clean up stale pending outcomes (older than 100 epochs)
    if (this.pendingOutcomes.size > 100) {
      const cutoff = epoch - 100;
      for (const [e] of this.pendingOutcomes) {
        if (e < cutoff) this.pendingOutcomes.delete(e);
      }
    }
  }

  /**
   * Check if we're in a low-entropy regime for the current alt asset.
   * If we don't have enough outcome history yet, default to trading
   * (optimistic — we'll build up the history over time).
   */
  private isLowEntropyRegime(): boolean {
    // If this IS a BTC market, we don't need entropy — BTC is the signal source
    if (this.currentAsset === "btc") return true;

    // Need enough history for entropy computation
    if (this.outcomeHistory.length < ENTROPY_WINDOW) {
      console.log(
        `[Entropy] Insufficient outcome history ` +
        `(${this.outcomeHistory.length}/${ENTROPY_WINDOW}) — ` +
        `trading optimistically`
      );
      return true;
    }

    const target = this.findLowestEntropyAlt();
    if (!target) return false;

    // Only trade if the lowest-entropy alt matches our current market
    if (target.asset !== this.currentAsset) {
      console.log(
        `[Entropy] Lowest entropy alt is ${target.asset} ` +
        `(H=${target.entropy.toFixed(4)}) but we're on ${this.currentAsset} — skipping`
      );
      return false;
    }

    return target.entropy < ENTROPY_THRESHOLD;
  }

  /**
   * Find the alt asset with the lowest conditional entropy over the
   * trailing window. Returns null if entropy can't be computed.
   */
  private findLowestEntropyAlt(): { asset: Asset; entropy: number } | null {
    if (this.outcomeHistory.length < ENTROPY_WINDOW) return null;

    const window = this.outcomeHistory.slice(-ENTROPY_WINDOW);
    let bestAlt: Asset | null = null;
    let bestH = Infinity;

    for (const alt of ALT_ASSETS) {
      const conditioningAssets = ASSETS.filter(a => a !== alt);
      const h = this.computeConditionalEntropy(window, alt, conditioningAssets);
      if (h < bestH) {
        bestH = h;
        bestAlt = alt;
      }
    }

    return bestAlt ? { asset: bestAlt, entropy: bestH } : null;
  }

  /**
   * Compute H(target | conditioning_assets) from the outcome window.
   * H(X|Y) = H(X,Y) - H(Y)
   */
  private computeConditionalEntropy(
    window: MarketOutcome[],
    target: Asset,
    conditioning: readonly Asset[],
  ): number {
    const n = window.length;

    // Encode conditioning state as single int (up to 3 bits for 3 assets)
    const condStates: number[] = [];
    const targetVals: number[] = [];

    for (const outcome of window) {
      targetVals.push(outcome[target]);
      let condState = 0;
      for (let i = 0; i < conditioning.length; i++) {
        condState += outcome[conditioning[i]!] * (1 << i);
      }
      condStates.push(condState);
    }

    // Joint distribution: target * 8 + condState
    const jointCounts = new Map<number, number>();
    const condCounts = new Map<number, number>();

    for (let i = 0; i < n; i++) {
      const jointKey = targetVals[i]! * 8 + condStates[i]!;
      jointCounts.set(jointKey, (jointCounts.get(jointKey) ?? 0) + 1);
      condCounts.set(condStates[i]!, (condCounts.get(condStates[i]!) ?? 0) + 1);
    }

    // H(joint)
    let hJoint = 0;
    for (const count of jointCounts.values()) {
      const p = count / n;
      if (p > 0) hJoint -= p * Math.log2(p);
    }

    // H(cond)
    let hCond = 0;
    for (const count of condCounts.values()) {
      const p = count / n;
      if (p > 0) hCond -= p * Math.log2(p);
    }

    return hJoint - hCond;
  }

  private getEntropyInfo(): string {
    if (this.outcomeHistory.length < ENTROPY_WINDOW) {
      return `entropy: warming up (${this.outcomeHistory.length}/${ENTROPY_WINDOW})`;
    }
    const target = this.findLowestEntropyAlt();
    if (!target) return "entropy: no valid alt";
    return `lowest H: ${target.asset}=${target.entropy.toFixed(4)}`;
  }
}

registerStrategy("entropy-binance", () => {
  const config = loadConfig();
  const spotFeed =
    config.execution.externalFeeds.find((f) => f.startsWith("binance-")) ??
    "binance-btcusdt";
  return new EntropyBinanceStrategy(config.execution.betDollars, spotFeed);
});
