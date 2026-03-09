import type {
  MarketState,
  TradeAction,
  DataMode,
  Side,
  OpenOrder,
  OrderBookState,
  MarketMakerMode,
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
const MINT_COOLDOWN_MS = 30_000;

export class MintAndSellMaker implements Strategy {
  readonly id = "market-maker";
  readonly dataMode: DataMode = "tick";

  // ── Config ──
  private readonly mode: MarketMakerMode;
  private readonly k: number;                     // A-S skew parameter (cents per unit of imbalance)
  private readonly initialPairs: number;           // pairs to mint at market open (mint-and-sell only)
  private readonly maxInventoryPerSide: number;    // safety cap
  private readonly replenishThreshold: number;     // mint more when inventory drops below this
  private readonly requoteThreshold: number;       // price change needed to trigger requote (cents)

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

  constructor(
    mode: MarketMakerMode = "mint-and-sell",
    k = 0.005,
    initialPairs = 50,
    maxInventoryPerSide = 200,
    replenishThreshold = 10,
    requoteThreshold = 0.005,
  ) {
    this.mode = mode;
    this.k = k;
    this.initialPairs = initialPairs;
    this.maxInventoryPerSide = maxInventoryPerSide;
    this.replenishThreshold = replenishThreshold;
    this.requoteThreshold = requoteThreshold;
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

    console.log(
      `[MM] Market open: ${market.slug} | mode=${this.mode} | ` +
      `k=${this.k} maxInv=${this.maxInventoryPerSide}` +
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
      // Optimistically credit inventory — if mint fails, we'll have no tokens
      // to sell and order placements will fail gracefully at the CLOB level.
      this.inventoryUp = this.initialPairs;
      this.inventoryDown = this.initialPairs;
      this.pairsMinted = this.initialPairs;
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

    // ── Clear pending mint after cooldown ──
    if (this.pendingMint && now - this.lastMintRequestMs > MINT_COOLDOWN_MS) {
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

    // ── Compute A-S skewed prices ──
    const imbalance = this.inventoryUp - this.inventoryDown;
    // Positive imbalance = hold more Up -> lower Up ask/bid (more eager to sell Up, less eager to buy)

    const bestAskUp = upBook.asks[0]!.price;
    const bestBidUp = upBook.bids.length > 0 ? upBook.bids[0]!.price : 0;
    const bestAskDown = downBook.asks[0]!.price;
    const bestBidDown = downBook.bids.length > 0 ? downBook.bids[0]!.price : 0;

    const tick = market.tickSize || 0.01;

    // Skewed asks: lower when we hold excess on that side
    let ourAskUp = bestAskUp - this.k * imbalance;
    let ourAskDown = bestAskDown + this.k * imbalance;
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
    this.reconcileAsk(actions, openOrders, this.upTokenId, ourAskUp, this.inventoryUp);
    this.reconcileAsk(actions, openOrders, this.downTokenId, ourAskDown, this.inventoryDown);

    // ── Reconcile bids (traditional mode only) ──
    if (this.mode === "traditional" && ourBidUp > 0 && ourBidDown > 0) {
      const bidSizeUp = Math.max(0, this.maxInventoryPerSide - this.inventoryUp);
      const bidSizeDown = Math.max(0, this.maxInventoryPerSide - this.inventoryDown);
      this.reconcileBid(actions, openOrders, this.upTokenId, ourBidUp, bidSizeUp);
      this.reconcileBid(actions, openOrders, this.downTokenId, ourBidDown, bidSizeDown);
    }

    // ── Replenishment minting (mint-and-sell only) ──
    if (this.mode === "mint-and-sell") {
      const minInventory = Math.min(this.inventoryUp, this.inventoryDown);
      if (minInventory < this.replenishThreshold && !this.pendingMint) {
        const deficit = this.initialPairs - minInventory;
        // Cap replenishment so neither side exceeds maxInventoryPerSide
        const maxMintable = this.maxInventoryPerSide - Math.max(this.inventoryUp, this.inventoryDown);
        const mintAmount = Math.min(deficit, maxMintable);
        if (mintAmount >= MIN_ORDER_SIZE) {
          this.pendingMint = true;
          this.lastMintRequestMs = now;
          const amount = BigInt(mintAmount) * USDC_DECIMALS;
          console.log(
            `[MM] Replenishing: mint ${mintAmount} pairs ` +
            `(inv Up=${this.inventoryUp} Down=${this.inventoryDown})`
          );
          this.inventoryUp += mintAmount;
          this.inventoryDown += mintAmount;
          this.pairsMinted += mintAmount;
          actions.push({
            type: "SPLIT",
            conditionId: this.conditionId,
            amount,
          });
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
      const redeemablePairs = Math.min(this.inventoryUp, this.inventoryDown);
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
        if (newInv > this.maxInventoryPerSide) {
          console.warn(
            `[MM] BUY fill would exceed max inventory: ${newInv} > ${this.maxInventoryPerSide}, capping`
          );
        }
        this.inventoryUp = Math.min(this.maxInventoryPerSide, newInv);
        this.buyFillsUp += size;
        this.buyCostUp += price * size;
        console.log(
          `[MM] Fill: BUY ${size} Up @ ${price.toFixed(3)} | ` +
          `inv=(Up=${this.inventoryUp} Down=${this.inventoryDown})`
        );
      } else if (assetId === this.downTokenId) {
        const newInv = this.inventoryDown + size;
        if (newInv > this.maxInventoryPerSide) {
          console.warn(
            `[MM] BUY fill would exceed max inventory: ${newInv} > ${this.maxInventoryPerSide}, capping`
          );
        }
        this.inventoryDown = Math.min(this.maxInventoryPerSide, newInv);
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
    inventory: number,
  ): void {
    const existing = openOrders.filter(o => o.assetId === assetId && o.side === "SELL");

    if (inventory >= MIN_ORDER_SIZE) {
      // Use first order for requote comparison; cancel all if requoting
      if (this.shouldRequote(existing[0], targetPrice, inventory)) {
        for (const order of existing) {
          actions.push({ type: "CANCEL_ORDER", orderId: order.orderId });
        }
        const size = Math.min(inventory, this.maxInventoryPerSide);
        actions.push({
          type: "PLACE_ORDER",
          assetId,
          side: "SELL" as Side,
          price: targetPrice,
          size,
          orderType: "GTC",
        });
      }
    } else {
      // No inventory — cancel all stale asks
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
      if (this.shouldRequote(existing[0], targetPrice, capacity)) {
        for (const order of existing) {
          actions.push({ type: "CANCEL_ORDER", orderId: order.orderId });
        }
        const size = Math.min(capacity, this.maxInventoryPerSide);
        actions.push({
          type: "PLACE_ORDER",
          assetId,
          side: "BUY" as Side,
          price: targetPrice,
          size,
          orderType: "GTC",
        });
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
    targetSize: number,
  ): boolean {
    if (!existing && targetSize >= MIN_ORDER_SIZE) return true;
    if (existing && Math.abs(existing.price - targetPrice) > this.requoteThreshold) return true;
    if (existing && Math.abs(existing.remainingSize - targetSize) >= MIN_ORDER_SIZE) return true;
    return false;
  }

  private clampPrice(price: number, minPrice: number, maxPrice: number, tick: number): number {
    const clamped = Math.max(minPrice, Math.min(maxPrice, price));
    return Math.round(clamped / tick) * tick;
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
      // Total cost = $1.00 per pair minted
      // Total recovered = sell revenue + value of mergeable pairs ($1.00 each)
      const redeemablePairs = Math.min(this.inventoryUp, this.inventoryDown);
      const totalRecovered = this.sellRevenueUp + this.sellRevenueDown + redeemablePairs;
      return totalRecovered - this.pairsMinted;
    } else {
      // Traditional: realized PnL = sell revenue - buy cost
      // (remaining inventory has unrealized value that settles at 0 or 1)
      const pnlUp = this.sellRevenueUp - this.buyCostUp;
      const pnlDown = this.sellRevenueDown - this.buyCostDown;
      return pnlUp + pnlDown;
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
    mm.initialPairs ?? 50,
    mm.maxInventoryPerSide ?? 200,
    mm.replenishThreshold ?? 10,
    mm.requoteThreshold ?? 0.005,
  );
});
