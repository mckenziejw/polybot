import type {
  MarketState,
  TradeAction,
  DataMode,
  Side,
  OpenOrder,
  OrderBookState,
  MarketMakerMode,
  PricingMode,
  PriceLevel,
} from "../types.ts";
import type { Strategy } from "./strategy.ts";
import { registerStrategy } from "./strategy.ts";
import { loadConfig } from "../config.ts";

/**
 * Dual-mode market maker for BTC Up/Down binary markets.
 *
 * Two operating modes:
 *
 * **mint-and-sell**: Structural edge, no directional risk.
 *   1. Mint token pairs via CTF splitPosition ($1.00 USDC -> 1 Up + 1 Down)
 *   2. Post ask orders on BOTH tokens simultaneously
 *   3. Capture combined spread: ask_up + ask_down > $1.00 (mean $1.067)
 *   4. Redeem unsold pairs at market end for $1.00 (breakeven minus gas)
 *
 * **traditional**: Classic bid/ask market making, higher fill rate.
 *   1. Post bids AND asks on both tokens
 *   2. Buy tokens from sellers (bid fills), sell to buyers (ask fills)
 *   3. Capture bid-ask spread on each token independently
 *   4. Higher fill rate (2x sides), but excess tokens settle at binary outcome
 *
 * Both modes use Avellaneda-Stoikov inventory skewing to manage imbalance.
 */

const MIN_ORDER_SIZE = 5; // Polymarket CLOB minimum
const USDC_DECIMALS = 1_000_000n; // 6 decimals

// Debounce: don't re-quote more often than this
const MIN_REQUOTE_INTERVAL_MS = 500;

// Don't post new orders within this many seconds of market end
const STOP_QUOTING_BEFORE_CLOSE_S = 60;

// Minimum ask price floor — never sell tokens for less than this
const MIN_ASK_PRICE = 0.10;

// Minimum bid price floor — never buy tokens for less than this
const MIN_BID_PRICE = 0.01;

// How long to wait for a mint tx before allowing another (ms)
// Polygon on-chain txs can take 60-120s during congestion
const MINT_COOLDOWN_MS = 120_000;

export class MintAndSellMaker implements Strategy {
  readonly id = "market-maker";
  readonly dataMode: DataMode = "tick";

  // ── Config ──
  private readonly mode: MarketMakerMode;
  private readonly pricing: PricingMode;
  private readonly k: number;                     // A-S skew parameter (cents per unit of imbalance)
  private readonly initialPairs: number;           // pairs to mint at market open (mint-and-sell only)
  private readonly maxTotalInventory: number;      // max tokens per side (USDC capital at risk)
  private readonly maxUnhedgedExposure: number;    // max |invUp - invDown| before pausing lagging side
  private readonly replenishThreshold: number;     // mint more when min(invUp,invDown) drops below this
  private readonly requoteThreshold: number;       // price change needed to trigger requote (cents)
  private readonly whaleThreshold: number;         // min size to identify whale resting orders
  private readonly whalePullCooldownMs: number;    // cooldown duration after whale signal
  private readonly whaleMoveTicks: number;         // bid/ask move >= this triggers cooldown

  // ── Inventory tracking ──
  // In mint-and-sell: tokens we hold from minting, decreased by sell fills
  // In traditional: tokens we hold from buy fills, decreased by sell fills
  private inventoryUp = 0;
  private inventoryDown = 0;
  private pairsMinted = 0;

  // ── PnL tracking ──
  private sellFillsUp = 0;
  private sellFillsDown = 0;
  private sellRevenueUp = 0;
  private sellRevenueDown = 0;
  private buyFillsUp = 0;       // traditional mode: tokens acquired via bids
  private buyFillsDown = 0;
  private buyCostUp = 0;        // traditional mode: total cost of bid fills
  private buyCostDown = 0;

  // ── Per-market state ──
  private upTokenId = "";
  private downTokenId = "";
  private conditionId = "";
  private lastRequoteMs = 0;
  private lastLogMs = 0;
  private initialMintSent = false;
  private pendingMint = false;
  private lastMintRequestMs = 0;  // when the last mint was requested (for cooldown)

  // ── Whale tracking (whale-front mode) ──
  private lastWhaleAskUp: number | null = null;   // whale ask level last tick
  private lastWhaleAskDown: number | null = null;
  private lastWhaleBidUp: number | null = null;    // whale bid level last tick
  private lastWhaleBidDown: number | null = null;
  private whalePullUntilMs = 0;  // cancel our orders until this timestamp (cooldown after whale signal)

  constructor(
    mode: MarketMakerMode = "mint-and-sell",
    k = 0.005,
    initialPairs = 100,
    maxTotalInventory = 200,
    maxUnhedgedExposure = 50,
    replenishThreshold = 20,
    requoteThreshold = 0.005,
    pricing: PricingMode = "inventory-skew",
    whaleThreshold = 1000,
    whalePullCooldownMs = 3000,
    whaleMoveTicks = 3,
  ) {
    this.mode = mode;
    this.pricing = pricing;
    this.k = k;
    this.initialPairs = initialPairs;
    this.maxTotalInventory = maxTotalInventory;
    this.maxUnhedgedExposure = maxUnhedgedExposure;
    this.replenishThreshold = replenishThreshold;
    this.requoteThreshold = requoteThreshold;
    this.whaleThreshold = whaleThreshold;
    this.whalePullCooldownMs = whalePullCooldownMs;
    this.whaleMoveTicks = whaleMoveTicks;
  }

  // ── Strategy interface ──

  async onMarketOpen(state: MarketState): Promise<void> {
    const { market } = state;

    this.upTokenId = market.upTokenId;
    this.downTokenId = market.downTokenId;
    this.conditionId = market.conditionId;

    // Reset per-market state
    this.inventoryUp = 0;
    this.inventoryDown = 0;
    this.pairsMinted = 0;
    this.sellFillsUp = 0;
    this.sellFillsDown = 0;
    this.sellRevenueUp = 0;
    this.sellRevenueDown = 0;
    this.buyFillsUp = 0;
    this.buyFillsDown = 0;
    this.buyCostUp = 0;
    this.buyCostDown = 0;
    this.lastRequoteMs = 0;
    this.lastLogMs = 0;
    this.initialMintSent = false;
    this.pendingMint = false;
    this.lastMintRequestMs = 0;
    this.lastWhaleAskUp = null;
    this.lastWhaleAskDown = null;
    this.lastWhaleBidUp = null;
    this.lastWhaleBidDown = null;
    this.whalePullUntilMs = 0;

    console.log(
      `[MM] Market open: ${market.slug} | mode=${this.mode} pricing=${this.pricing} | ` +
      `k=${this.k} maxTotal=${this.maxTotalInventory} maxUnhedged=${this.maxUnhedgedExposure}` +
      (this.pricing === "whale-front" ? ` whaleThreshold=${this.whaleThreshold}` : "") +
      (this.mode === "mint-and-sell"
        ? ` | initial mint: ${this.initialPairs} pairs ($${this.initialPairs})`
        : ` | starting empty, building inventory from bid fills`)
    );
  }

  async evaluate(state: MarketState): Promise<TradeAction[]> {
    const { market, books, openOrders, timeRemainingMs } = state;
    const now = Date.now();
    const actions: TradeAction[] = [];

    // ── Phase 0: Initial mint (mint-and-sell only) ──
    if (this.mode === "mint-and-sell" && !this.initialMintSent) {
      this.initialMintSent = true;
      this.pendingMint = true;
      this.lastMintRequestMs = now;
      const amount = BigInt(this.initialPairs) * USDC_DECIMALS;
      console.log(`[MM] Minting initial ${this.initialPairs} pairs`);
      // Inventory is credited in onMintComplete() after on-chain confirmation
      return [{
        type: "SPLIT",
        conditionId: this.conditionId,
        amount,
      }];
    }

    // In traditional mode, mark init as done immediately
    if (!this.initialMintSent) {
      this.initialMintSent = true;
    }

    // ── Safety fallback: clear pending mint after cooldown if onMintComplete was never called ──
    if (this.pendingMint && now - this.lastMintRequestMs > MINT_COOLDOWN_MS) {
      console.log(`[MM] Mint cooldown expired without onMintComplete — clearing pendingMint`);
      this.pendingMint = false;
    }

    // ── Stop quoting near market close ──
    const timeRemainingS = timeRemainingMs / 1000;
    if (timeRemainingS < STOP_QUOTING_BEFORE_CLOSE_S) {
      if (openOrders.length > 0) {
        console.log(`[MM] ${timeRemainingS.toFixed(0)}s left — cancelling all orders`);
        return [{ type: "CANCEL_ALL" }];
      }
      return [{ type: "NOOP" }];
    }

    // ── Debounce requotes ──
    if (now - this.lastRequoteMs < MIN_REQUOTE_INTERVAL_MS) {
      return [{ type: "NOOP" }];
    }

    // ── Get books ──
    const upBook = books[this.upTokenId];
    const downBook = books[this.downTokenId];

    if (!upBook || !downBook) {
      return [{ type: "NOOP" }];
    }

    if (upBook.asks.length === 0 || downBook.asks.length === 0) {
      return [{ type: "NOOP" }];
    }

    // ── Compute prices ──
    const imbalance = this.inventoryUp - this.inventoryDown;
    // Positive imbalance = hold more Up -> lower Up ask/bid (more eager to sell Up, less eager to buy)

    const bestAskUp = upBook.asks[0]!.price;
    const bestBidUp = upBook.bids.length > 0 ? upBook.bids[0]!.price : 0;
    const bestAskDown = downBook.asks[0]!.price;
    const bestBidDown = downBook.bids.length > 0 ? downBook.bids[0]!.price : 0;

    const tick = market.tickSize || 0.01;

    let ourAskUp: number;
    let ourAskDown: number;

    if (this.pricing === "whale-front") {
      // ── Whale front-run pricing ──
      // Find whale resting orders (asks and bids) and place one tick inside their asks.
      // Falls back to best ask if no whale level found.
      const whaleAskUp = this.findWhaleLevel(upBook.asks, this.whaleThreshold);
      const whaleAskDown = this.findWhaleLevel(downBook.asks, this.whaleThreshold);
      const whaleBidUp = this.findWhaleLevel(upBook.bids, this.whaleThreshold);
      const whaleBidDown = this.findWhaleLevel(downBook.bids, this.whaleThreshold);

      // ── Whale signal detection ──
      // Signals that whales have advance info and we should pull our orders:
      //   1. Whale ask disappears (pulled entirely)
      //   2. Whale bid drops N+ ticks (retreating from a side = expects it to lose)
      //   3. Whale ask drops N+ ticks toward mid (rushing to sell = expects it to be worthless)
      const moveThreshold = this.whaleMoveTicks * tick;
      const signals: string[] = [];

      // Ask pulls
      if (this.lastWhaleAskUp !== null && whaleAskUp === null) signals.push("Up ask GONE");
      if (this.lastWhaleAskDown !== null && whaleAskDown === null) signals.push("Down ask GONE");

      // Bid retreats (price drops = pulling back)
      if (this.lastWhaleBidUp !== null && whaleBidUp !== null &&
          this.lastWhaleBidUp - whaleBidUp >= moveThreshold) {
        signals.push(`Up bid dropped ${((this.lastWhaleBidUp - whaleBidUp) / tick).toFixed(0)} ticks`);
      }
      if (this.lastWhaleBidDown !== null && whaleBidDown !== null &&
          this.lastWhaleBidDown - whaleBidDown >= moveThreshold) {
        signals.push(`Down bid dropped ${((this.lastWhaleBidDown - whaleBidDown) / tick).toFixed(0)} ticks`);
      }
      // Bid pulls
      if (this.lastWhaleBidUp !== null && whaleBidUp === null) signals.push("Up bid GONE");
      if (this.lastWhaleBidDown !== null && whaleBidDown === null) signals.push("Down bid GONE");

      // Ask tightening toward mid (rushing to dump = expects token to lose value)
      if (this.lastWhaleAskUp !== null && whaleAskUp !== null &&
          this.lastWhaleAskUp - whaleAskUp >= moveThreshold) {
        signals.push(`Up ask tightened ${((this.lastWhaleAskUp - whaleAskUp) / tick).toFixed(0)} ticks`);
      }
      if (this.lastWhaleAskDown !== null && whaleAskDown !== null &&
          this.lastWhaleAskDown - whaleAskDown >= moveThreshold) {
        signals.push(`Down ask tightened ${((this.lastWhaleAskDown - whaleAskDown) / tick).toFixed(0)} ticks`);
      }

      if (signals.length > 0) {
        this.whalePullUntilMs = now + this.whalePullCooldownMs;
        console.log(`[MM] Whale signal: [${signals.join(", ")}] — cooldown ${this.whalePullCooldownMs}ms`);
      }

      // Update tracking for next tick
      this.lastWhaleAskUp = whaleAskUp;
      this.lastWhaleAskDown = whaleAskDown;
      this.lastWhaleBidUp = whaleBidUp;
      this.lastWhaleBidDown = whaleBidDown;

      // During cooldown, cancel all our orders and skip quoting
      if (now < this.whalePullUntilMs) {
        if (openOrders.length > 0) {
          return [{ type: "CANCEL_ALL" }];
        }
        return [{ type: "NOOP" }];
      }

      ourAskUp = (whaleAskUp ?? bestAskUp) - tick;
      ourAskDown = (whaleAskDown ?? bestAskDown) - tick;

      // Still apply inventory skew on top — nudge the side we're long
      ourAskUp -= this.k * imbalance;
      ourAskDown += this.k * imbalance;
    } else {
      // ── Inventory-skew pricing (Avellaneda-Stoikov) ──
      ourAskUp = bestAskUp - this.k * imbalance;
      ourAskDown = bestAskDown + this.k * imbalance;
    }

    // Profitability floor (mint-and-sell): combined asks must exceed $1.00 + margin.
    // Use best asks as reference for the other side's likely fill price.
    if (this.mode === "mint-and-sell") {
      const minCombined = 1.02; // $0.02 margin per pair to cover gas + fees
      ourAskUp = Math.max(ourAskUp, minCombined - bestAskDown);
      ourAskDown = Math.max(ourAskDown, minCombined - bestAskUp);
    }

    // Floor at MIN_ASK_PRICE to prevent selling tokens for near-zero when bids are empty
    ourAskUp = this.clampPrice(ourAskUp, Math.max(MIN_ASK_PRICE, bestBidUp + tick), 0.99, tick);
    ourAskDown = this.clampPrice(ourAskDown, Math.max(MIN_ASK_PRICE, bestBidDown + tick), 0.99, tick);

    // Skewed bids (traditional mode): higher when we hold less on that side
    let ourBidUp = 0;
    let ourBidDown = 0;
    if (this.mode === "traditional" && bestBidUp > 0 && bestBidDown > 0) {
      ourBidUp = bestBidUp - this.k * imbalance;
      ourBidDown = bestBidDown + this.k * imbalance;
      ourBidUp = this.clampPrice(ourBidUp, MIN_BID_PRICE, bestAskUp - tick, tick);
      ourBidDown = this.clampPrice(ourBidDown, MIN_BID_PRICE, bestAskDown - tick, tick);
    }

    // ── Reconcile asks (both modes) ──
    // In mint-and-sell, cap sell size so imbalance doesn't exceed maxUnhedgedExposure.
    // Selling n tokens from one side increases imbalance by n (when that side is
    // already shorter or equal). Cap: n <= maxUnhedgedExposure - currentImbalance.
    const unhedged = Math.abs(imbalance);
    let sellableUp = this.inventoryUp;
    let sellableDown = this.inventoryDown;
    if (this.mode === "mint-and-sell") {
      // Cap sell sizes so that if ONE side fills completely while the other gets
      // ZERO fills, the resulting imbalance stays within maxUnhedgedExposure.
      //
      // Selling the longer side first reduces imbalance toward zero, then if we
      // sell past the balance point it creates imbalance in the other direction.
      // So the max we can sell of the longer side = imbalance + maxUnhedgedExposure.
      //
      // Selling the shorter side always increases imbalance, so cap = headroom.
      if (unhedged < MIN_ORDER_SIZE) {
        // Effectively balanced (dust imbalance from CLOB rounding).
        // Treat as balanced so both sides can post at least MIN_ORDER_SIZE.
        sellableUp = Math.min(this.inventoryUp, this.maxUnhedgedExposure);
        sellableDown = Math.min(this.inventoryDown, this.maxUnhedgedExposure);
      } else if (this.inventoryUp > this.inventoryDown) {
        // Long Up — can sell up to (imbalance + maxUnhedged) before flipping too far
        sellableUp = Math.min(this.inventoryUp, unhedged + this.maxUnhedgedExposure);
        // Short Down — selling increases imbalance
        const headroom = Math.max(0, this.maxUnhedgedExposure - unhedged);
        sellableDown = Math.min(this.inventoryDown, headroom);
      } else if (this.inventoryDown > this.inventoryUp) {
        // Long Down — can sell up to (imbalance + maxUnhedged) before flipping too far
        sellableDown = Math.min(this.inventoryDown, unhedged + this.maxUnhedgedExposure);
        // Short Up — selling increases imbalance
        const headroom = Math.max(0, this.maxUnhedgedExposure - unhedged);
        sellableUp = Math.min(this.inventoryUp, headroom);
      } else {
        // Perfectly balanced — selling either side creates imbalance from zero
        sellableUp = Math.min(this.inventoryUp, this.maxUnhedgedExposure);
        sellableDown = Math.min(this.inventoryDown, this.maxUnhedgedExposure);
      }
    }
    this.reconcileAsk(actions, openOrders, this.upTokenId, ourAskUp, sellableUp);
    this.reconcileAsk(actions, openOrders, this.downTokenId, ourAskDown, sellableDown);

    // ── Reconcile bids (traditional mode only) ──
    if (this.mode === "traditional" && ourBidUp > 0 && ourBidDown > 0) {
      const bidSizeUp = Math.max(0, this.maxTotalInventory - this.inventoryUp);
      const bidSizeDown = Math.max(0, this.maxTotalInventory - this.inventoryDown);
      this.reconcileBid(actions, openOrders, this.upTokenId, ourBidUp, bidSizeUp);
      this.reconcileBid(actions, openOrders, this.downTokenId, ourBidDown, bidSizeDown);
    }

    // ── Replenishment minting (mint-and-sell only) ──
    // Mint adds equal tokens to BOTH sides, so it never changes imbalance.
    // Trigger: min(invUp, invDown) < replenishThreshold (running low on paired inventory).
    // Target: bring min side back up to initialPairs.
    // Cap: neither side exceeds maxTotalInventory.
    if (this.mode === "mint-and-sell") {
      const minInventory = Math.min(this.inventoryUp, this.inventoryDown);
      if (minInventory < this.replenishThreshold && !this.pendingMint) {
        const deficit = this.initialPairs - minInventory;
        // Cap so neither side exceeds maxTotalInventory after minting
        const maxMintable = this.maxTotalInventory - Math.max(this.inventoryUp, this.inventoryDown);
        const mintAmount = Math.floor(Math.min(deficit, maxMintable));
        if (mintAmount >= MIN_ORDER_SIZE) {
          this.pendingMint = true;
          this.lastMintRequestMs = now;
          const amount = BigInt(mintAmount) * USDC_DECIMALS;
          console.log(
            `[MM] Replenishing: mint ${mintAmount} pairs ` +
            `(inv Up=${this.inventoryUp} Down=${this.inventoryDown})`
          );
          actions.push({
            type: "SPLIT",
            conditionId: this.conditionId,
            amount,
          });
        } else if (maxMintable < MIN_ORDER_SIZE && minInventory < MIN_ORDER_SIZE) {
          // Can't mint because one side is near maxTotalInventory.
          // Need to sell that side to free up room for minting.
          console.warn(
            `[MM] Replenish blocked: need ${deficit} pairs but maxMintable=${maxMintable} ` +
            `(inv Up=${this.inventoryUp} Down=${this.inventoryDown}, cap=${this.maxTotalInventory})`
          );
        }
      }
    }

    if (actions.length > 0) {
      this.lastRequoteMs = now;
    }

    // ── Periodic logging ──
    if (now - this.lastLogMs > 10_000) {
      this.lastLogMs = now;
      this.logStatus(ourAskUp, ourAskDown, ourBidUp, ourBidDown, imbalance, timeRemainingS);
    }

    return actions.length > 0 ? actions : [{ type: "NOOP" }];
  }

  async onMarketClose(state: MarketState): Promise<TradeAction[]> {
    const actions: TradeAction[] = [];

    if (state.openOrders.length > 0) {
      actions.push({ type: "CANCEL_ALL" });
    }

    // Mint-and-sell: merge remaining pairs for $1.00 each
    if (this.mode === "mint-and-sell") {
      const redeemablePairs = Math.floor(Math.min(this.inventoryUp, this.inventoryDown));
      if (redeemablePairs > 0) {
        const amount = BigInt(redeemablePairs) * USDC_DECIMALS;
        console.log(`[MM] Market close — merging ${redeemablePairs} remaining pairs`);
        actions.push({
          type: "MERGE_PAIRS",
          conditionId: this.conditionId,
          amount,
        });
      }
    }

    this.logSessionSummary();

    return actions.length > 0 ? actions : [{ type: "NOOP" }];
  }

  onFill(assetId: string, side: Side, size: number, price: number): void {
    if (side === "SELL") {
      if (assetId === this.upTokenId) {
        this.inventoryUp = Math.max(0, this.inventoryUp - size);
        this.sellFillsUp += size;
        this.sellRevenueUp += price * size;
        console.log(
          `[MM] Fill: SELL ${size} Up @ ${price.toFixed(3)} | ` +
          `inv=(Up=${this.inventoryUp} Down=${this.inventoryDown})`
        );
      } else if (assetId === this.downTokenId) {
        this.inventoryDown = Math.max(0, this.inventoryDown - size);
        this.sellFillsDown += size;
        this.sellRevenueDown += price * size;
        console.log(
          `[MM] Fill: SELL ${size} Down @ ${price.toFixed(3)} | ` +
          `inv=(Up=${this.inventoryUp} Down=${this.inventoryDown})`
        );
      }
    } else if (side === "BUY") {
      if (assetId === this.upTokenId) {
        const newInv = this.inventoryUp + size;
        if (newInv > this.maxTotalInventory) {
          console.warn(
            `[MM] BUY fill would exceed max inventory: ${newInv} > ${this.maxTotalInventory}, capping`
          );
        }
        this.inventoryUp = Math.min(this.maxTotalInventory, newInv);
        this.buyFillsUp += size;
        this.buyCostUp += price * size;
        console.log(
          `[MM] Fill: BUY ${size} Up @ ${price.toFixed(3)} | ` +
          `inv=(Up=${this.inventoryUp} Down=${this.inventoryDown})`
        );
      } else if (assetId === this.downTokenId) {
        const newInv = this.inventoryDown + size;
        if (newInv > this.maxTotalInventory) {
          console.warn(
            `[MM] BUY fill would exceed max inventory: ${newInv} > ${this.maxTotalInventory}, capping`
          );
        }
        this.inventoryDown = Math.min(this.maxTotalInventory, newInv);
        this.buyFillsDown += size;
        this.buyCostDown += price * size;
        console.log(
          `[MM] Fill: BUY ${size} Down @ ${price.toFixed(3)} | ` +
          `inv=(Up=${this.inventoryUp} Down=${this.inventoryDown})`
        );
      }
    }
  }

  // ── Private helpers ──

  private reconcileAsk(
    actions: TradeAction[],
    openOrders: OpenOrder[],
    assetId: string,
    targetPrice: number,
    sellableSize: number,
  ): void {
    const existing = openOrders.filter(o => o.assetId === assetId && o.side === "SELL");

    if (sellableSize >= MIN_ORDER_SIZE) {
      if (existing.length === 0) {
        actions.push({
          type: "PLACE_ORDER",
          assetId,
          side: "SELL" as Side,
          price: targetPrice,
          size: sellableSize,
          orderType: "GTC",
        });
      } else if (this.shouldRequote(existing[0], targetPrice, sellableSize)) {
        for (const order of existing) {
          actions.push({ type: "CANCEL_ORDER", orderId: order.orderId });
        }
      }
    } else {
      for (const order of existing) {
        actions.push({ type: "CANCEL_ORDER", orderId: order.orderId });
      }
    }
  }

  private reconcileBid(
    actions: TradeAction[],
    openOrders: OpenOrder[],
    assetId: string,
    targetPrice: number,
    capacity: number,
  ): void {
    const existing = openOrders.filter(o => o.assetId === assetId && o.side === "BUY");

    if (capacity >= MIN_ORDER_SIZE && targetPrice > 0) {
      if (existing.length === 0) {
        actions.push({
          type: "PLACE_ORDER",
          assetId,
          side: "BUY" as Side,
          price: targetPrice,
          size: capacity,
          orderType: "GTC",
        });
      } else if (this.shouldRequote(existing[0], targetPrice, capacity)) {
        for (const order of existing) {
          actions.push({ type: "CANCEL_ORDER", orderId: order.orderId });
        }
      }
    } else {
      for (const order of existing) {
        actions.push({ type: "CANCEL_ORDER", orderId: order.orderId });
      }
    }
  }

  private shouldRequote(
    existing: OpenOrder | undefined,
    targetPrice: number,
    _targetSize: number,
  ): boolean {
    if (!existing) return true;
    // Only requote when price has actually moved beyond threshold.
    // Size changes at the same price are not worth losing queue priority.
    if (Math.abs(existing.price - targetPrice) > this.requoteThreshold) return true;
    return false;
  }

  private clampPrice(price: number, minPrice: number, maxPrice: number, tick: number): number {
    const clamped = Math.max(minPrice, Math.min(maxPrice, price));
    return Math.round(clamped / tick) * tick;
  }

  /**
   * Find the first ask level with size >= threshold (a whale resting order).
   * Returns the price, or null if no level meets the threshold.
   * Asks are sorted ascending by price, so the first match is the tightest whale level.
   */
  private findWhaleLevel(asks: PriceLevel[], threshold: number): number | null {
    for (const level of asks) {
      if (level.size >= threshold) return level.price;
    }
    return null;
  }

  private logStatus(
    askUp: number, askDown: number,
    bidUp: number, bidDown: number,
    imbalance: number, timeRemainingS: number,
  ): void {
    const pnl = this.computeRealizedPnl();
    const base = `[MM] inv=(Up=${this.inventoryUp} Down=${this.inventoryDown}) ` +
      `sells=(Up=${this.sellFillsUp} Down=${this.sellFillsDown}) ` +
      `pnl=$${pnl.toFixed(2)} ` +
      `asks=(${askUp.toFixed(3)}/${askDown.toFixed(3)}) ` +
      `imbalance=${imbalance > 0 ? "+" : ""}${imbalance} ` +
      `remaining=${timeRemainingS.toFixed(0)}s`;

    if (this.mode === "mint-and-sell") {
      const completedPairs = Math.min(this.sellFillsUp, this.sellFillsDown);
      console.log(base + ` | minted=${this.pairsMinted} pairs_completed=${completedPairs}`);
    } else {
      console.log(
        base +
        ` | buys=(Up=${this.buyFillsUp} Down=${this.buyFillsDown}) ` +
        `bids=(${bidUp.toFixed(3)}/${bidDown.toFixed(3)})`
      );
    }
  }

  private computeRealizedPnl(): number {
    if (this.mode === "mint-and-sell") {
      // Realized PnL = revenue from completed pairs - cost of those pairs.
      // A "completed pair" requires one sell on each side.
      // Cost per pair = $1.00 (the USDC.e spent to mint).
      // Revenue = sell price UP + sell price DOWN for each completed pair.
      //
      // We approximate by using min(sellFills) as completed pairs and
      // charging $1.00 per completed pair against total sell revenue.
      const completedPairs = Math.min(this.sellFillsUp, this.sellFillsDown);
      const totalSellRevenue = this.sellRevenueUp + this.sellRevenueDown;
      return totalSellRevenue - completedPairs;
    } else {
      // Traditional: realized PnL = sell revenue - buy cost
      // (remaining inventory has unrealized value that settles at 0 or 1)
      const pnlUp = this.sellRevenueUp - this.buyCostUp;
      const pnlDown = this.sellRevenueDown - this.buyCostDown;
      return pnlUp + pnlDown;
    }
  }

  getState(): Record<string, unknown> {
    return {
      mode: this.mode,
      pricing: this.pricing,
      inventoryUp: this.inventoryUp,
      inventoryDown: this.inventoryDown,
      pairsMinted: this.pairsMinted,
      sellFillsUp: this.sellFillsUp,
      sellFillsDown: this.sellFillsDown,
      sellRevenueUp: this.sellRevenueUp,
      sellRevenueDown: this.sellRevenueDown,
      buyFillsUp: this.buyFillsUp,
      buyFillsDown: this.buyFillsDown,
      buyCostUp: this.buyCostUp,
      buyCostDown: this.buyCostDown,
      realizedPnl: this.computeRealizedPnl(),
      pendingMint: this.pendingMint,
      redeemablePairs: Math.min(this.inventoryUp, this.inventoryDown),
    };
  }

  onInventorySync(
    upBalance: number,
    downBalance: number,
    openOrders: { assetId: string; side: string; price: number; size: number }[],
  ): void {
    this.inventoryUp = upBalance;
    this.inventoryDown = downBalance;

    // Count existing inventory as already minted (so PnL doesn't charge for it again)
    // Only in mint-and-sell mode — these tokens came from previous mints
    if (this.mode === "mint-and-sell") {
      const pairsHeld = Math.floor(Math.min(upBalance, downBalance));
      this.pairsMinted = pairsHeld;
    }

    // If we already have inventory, skip the initial mint
    if (upBalance > 0 || downBalance > 0) {
      this.initialMintSent = true;
    }

    console.log(
      `[MM] Inventory synced from on-chain: Up=${upBalance} Down=${downBalance} ` +
      `openOrders=${openOrders.length}`
    );
  }

  onMintComplete(conditionId: string, pairsMinted: number, success: boolean): void {
    this.pendingMint = false;
    if (success) {
      this.inventoryUp += pairsMinted;
      this.inventoryDown += pairsMinted;
      this.pairsMinted += pairsMinted;
      console.log(
        `[MM] Mint confirmed: +${pairsMinted} pairs → ` +
        `inv=(Up=${this.inventoryUp} Down=${this.inventoryDown})`
      );
    } else {
      console.error(`[MM] Mint FAILED for ${pairsMinted} pairs — inventory NOT credited`);
    }
  }

  private logSessionSummary(): void {
    const pnl = this.computeRealizedPnl();

    if (this.mode === "mint-and-sell") {
      const redeemablePairs = Math.min(this.inventoryUp, this.inventoryDown);
      const excessUp = this.inventoryUp - redeemablePairs;
      const excessDown = this.inventoryDown - redeemablePairs;
      const completedPairs = Math.min(this.sellFillsUp, this.sellFillsDown);

      console.log(
        `[MM] Session summary (mint-and-sell): ` +
        `minted=${this.pairsMinted} sells=(Up=${this.sellFillsUp} Down=${this.sellFillsDown}) ` +
        `completed_pairs=${completedPairs} merged=${redeemablePairs} ` +
        `excess=(Up=${excessUp} Down=${excessDown}) ` +
        `pnl=$${pnl.toFixed(2)}`
      );

      if (excessUp > 0 || excessDown > 0) {
        console.warn(
          `[MM] WARNING: ${excessUp + excessDown} orphaned tokens will settle at binary outcome`
        );
      }
    } else {
      console.log(
        `[MM] Session summary (traditional): ` +
        `buys=(Up=${this.buyFillsUp} Down=${this.buyFillsDown}) ` +
        `sells=(Up=${this.sellFillsUp} Down=${this.sellFillsDown}) ` +
        `remaining_inv=(Up=${this.inventoryUp} Down=${this.inventoryDown}) ` +
        `cash_pnl=$${pnl.toFixed(2)}`
      );
    }
  }
}

registerStrategy("market-maker", () => {
  const config = loadConfig();
  const mm = config.marketMaker;
  return new MintAndSellMaker(
    mm.mode ?? "mint-and-sell",
    mm.skewK ?? 0.005,
    mm.initialPairs ?? 100,
    mm.maxTotalInventory ?? 200,
    mm.maxUnhedgedExposure ?? 50,
    mm.replenishThreshold ?? 20,
    mm.requoteThreshold ?? 0.005,
    mm.pricing ?? "inventory-skew",
    mm.whaleThreshold ?? 1000,
    mm.whalePullCooldownMs ?? 3000,
    mm.whaleMoveTicks ?? 3,
  );
});
