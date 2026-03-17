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

  // ── Black-Scholes pricing state ──
  private readonly bsVolWindowS: number;
  private readonly bsSpreadMultiplier: number;
  private readonly bsMinSpread: number;
  private readonly bsExternalFeed: string;
  private readonly bsTDof: number;                 // Student's t degrees of freedom (0 = normal)
  private bsStrike = 0;                           // BTC price at market open (the binary option strike)
  private bsMarketDurationMs = 0;                  // total market duration for annualization
  private bsPriceHistory: { ts: number; price: number }[] = [];  // ring buffer for vol estimation

  constructor(
    mode: MarketMakerMode = "mint-and-sell",
    k = 0.005,
    initialPairs = 100,
    maxTotalInventory = 200,
    maxUnhedgedExposure = 20,
    replenishThreshold = 20,
    requoteThreshold = 0.005,
    pricing: PricingMode = "whale-front",
    whaleThreshold = 2000,
    whalePullCooldownMs = 3000,
    whaleMoveTicks = 3,
    bsVolWindowS = 300,
    bsSpreadMultiplier = 1.0,
    bsMinSpread = 0.02,
    bsExternalFeed = "binance-btcusdt",
    bsTDof = 4,
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
    this.bsVolWindowS = bsVolWindowS;
    this.bsSpreadMultiplier = bsSpreadMultiplier;
    this.bsMinSpread = bsMinSpread;
    this.bsExternalFeed = bsExternalFeed;
    this.bsTDof = bsTDof;
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
    this.bsPriceHistory = [];
    this.bsStrike = 0;
    this.bsMarketDurationMs = state.timeRemainingMs;

    // Capture strike price from external feed at market open
    if (this.pricing === "black-scholes" && state.externalData) {
      const feed = state.externalData[this.bsExternalFeed];
      if (feed) {
        this.bsStrike = feed.price;
        console.log(`[MM/BS] Strike set from ${this.bsExternalFeed}: $${this.bsStrike.toFixed(2)}`);
      }
    }

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

      // ── Whale movement logging (no cooldown — just reposition alongside them) ──
      const moveThreshold = this.whaleMoveTicks * tick;
      const signals: string[] = [];

      if (this.lastWhaleAskUp !== null && whaleAskUp === null) signals.push("Up ask GONE");
      if (this.lastWhaleAskDown !== null && whaleAskDown === null) signals.push("Down ask GONE");
      if (this.lastWhaleBidUp !== null && whaleBidUp !== null &&
          this.lastWhaleBidUp - whaleBidUp >= moveThreshold) {
        signals.push(`Up bid dropped ${((this.lastWhaleBidUp - whaleBidUp) / tick).toFixed(0)} ticks`);
      }
      if (this.lastWhaleBidDown !== null && whaleBidDown !== null &&
          this.lastWhaleBidDown - whaleBidDown >= moveThreshold) {
        signals.push(`Down bid dropped ${((this.lastWhaleBidDown - whaleBidDown) / tick).toFixed(0)} ticks`);
      }
      if (this.lastWhaleBidUp !== null && whaleBidUp === null) signals.push("Up bid GONE");
      if (this.lastWhaleBidDown !== null && whaleBidDown === null) signals.push("Down bid GONE");
      if (this.lastWhaleAskUp !== null && whaleAskUp !== null &&
          this.lastWhaleAskUp - whaleAskUp >= moveThreshold) {
        signals.push(`Up ask tightened ${((this.lastWhaleAskUp - whaleAskUp) / tick).toFixed(0)} ticks`);
      }
      if (this.lastWhaleAskDown !== null && whaleAskDown !== null &&
          this.lastWhaleAskDown - whaleAskDown >= moveThreshold) {
        signals.push(`Down ask tightened ${((this.lastWhaleAskDown - whaleAskDown) / tick).toFixed(0)} ticks`);
      }

      if (signals.length > 0) {
        console.log(`[MM] Whale move: [${signals.join(", ")}] — repositioning`);
      }

      // Update tracking for next tick
      this.lastWhaleAskUp = whaleAskUp;
      this.lastWhaleAskDown = whaleAskDown;
      this.lastWhaleBidUp = whaleBidUp;
      this.lastWhaleBidDown = whaleBidDown;

      // Only use whale levels that are competitive — within maxWhaleSpread of
      // the best bid. Whale asks deep in the book (e.g. 0.98 when mid is 0.64)
      // are not meaningful levels to front-run.
      const maxWhaleSpread = 0.15; // ignore whale asks more than 15c above best bid
      const defaultSpread = 0.04;  // fallback: 4c above best bid when no usable whale

      const usableWhaleUp = whaleAskUp !== null && bestBidUp > 0 &&
        (whaleAskUp - bestBidUp) <= maxWhaleSpread ? whaleAskUp : null;
      const usableWhaleDown = whaleAskDown !== null && bestBidDown > 0 &&
        (whaleAskDown - bestBidDown) <= maxWhaleSpread ? whaleAskDown : null;

      if (usableWhaleUp !== null) {
        ourAskUp = usableWhaleUp - tick;
      } else {
        ourAskUp = bestBidUp > 0 ? bestBidUp + defaultSpread : bestAskUp;
      }
      if (usableWhaleDown !== null) {
        ourAskDown = usableWhaleDown - tick;
      } else {
        ourAskDown = bestBidDown > 0 ? bestBidDown + defaultSpread : bestAskDown;
      }

      // Apply inventory skew on top — nudge the side we're long
      ourAskUp -= this.k * imbalance;
      ourAskDown += this.k * imbalance;
    } else if (this.pricing === "black-scholes") {
      // ── Black-Scholes binary option pricing ──
      // Fair value = N(d2) for a cash-or-nothing binary call.
      // Ask = fair + halfSpread (we're selling), with inventory skew on top.
      const bs = this.bsUpdate(state.externalData, timeRemainingMs);

      if (bs === null) {
        // Not enough data yet — fall back to inventory-skew
        ourAskUp = bestAskUp - this.k * imbalance;
        ourAskDown = bestAskDown + this.k * imbalance;
      } else {
        ourAskUp = bs.fairUp + bs.halfSpread;
        ourAskDown = bs.fairDown + bs.halfSpread;

        // Inventory skew on top
        ourAskUp -= this.k * imbalance;
        ourAskDown += this.k * imbalance;

        // Never ask below fair value — we'd be giving away edge
        ourAskUp = Math.max(ourAskUp, bs.fairUp + tick);
        ourAskDown = Math.max(ourAskDown, bs.fairDown + tick);

        // Periodic BS-specific logging
        if (now - this.lastLogMs > 10_000) {
          const moneyness = ((bs.spot - this.bsStrike) / this.bsStrike * 100).toFixed(3);
          console.log(
            `[MM/BS] spot=$${bs.spot.toFixed(1)} strike=$${this.bsStrike.toFixed(1)} ` +
            `moneyness=${moneyness}% σ=${(bs.sigma * 100).toFixed(1)}% ` +
            `fair=(${bs.fairUp.toFixed(3)}/${bs.fairDown.toFixed(3)}) ` +
            `spread=${(bs.halfSpread * 2).toFixed(3)} ` +
            `ask=(${ourAskUp.toFixed(3)}/${ourAskDown.toFixed(3)})`
          );
        }
      }
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
    // Must account for outstanding sell orders that haven't filled yet — if they
    // fill while our new order is also live, the combined exposure could blow
    // past the cap.
    const unhedged = Math.abs(imbalance);
    let sellableUp = this.inventoryUp;
    let sellableDown = this.inventoryDown;

    // Subtract outstanding sell order sizes — these tokens are "spoken for"
    const outstandingSellUp = openOrders
      .filter(o => o.assetId === this.upTokenId && o.side === "SELL")
      .reduce((sum, o) => sum + o.remainingSize, 0);
    const outstandingSellDown = openOrders
      .filter(o => o.assetId === this.downTokenId && o.side === "SELL")
      .reduce((sum, o) => sum + o.remainingSize, 0);

    if (this.mode === "mint-and-sell") {
      // Cap sell sizes so that if ONE side fills completely while the other gets
      // ZERO fills, the resulting imbalance stays within maxUnhedgedExposure.
      //
      // Treat outstanding orders as if already sold for imbalance purposes:
      // effective imbalance after all outstanding orders fill =
      //   (inventoryUp - outstandingSellUp) - (inventoryDown - outstandingSellDown)
      const effectiveImbalance = (this.inventoryUp - outstandingSellUp) -
                                  (this.inventoryDown - outstandingSellDown);
      const effectiveUnhedged = Math.abs(effectiveImbalance);

      if (effectiveUnhedged < MIN_ORDER_SIZE) {
        sellableUp = Math.min(this.inventoryUp, this.maxUnhedgedExposure);
        sellableDown = Math.min(this.inventoryDown, this.maxUnhedgedExposure);
      } else if (effectiveImbalance > 0) {
        // Effectively long Up after outstanding fills
        sellableUp = Math.min(this.inventoryUp, effectiveUnhedged + this.maxUnhedgedExposure);
        const headroom = Math.max(0, this.maxUnhedgedExposure - effectiveUnhedged);
        sellableDown = Math.min(this.inventoryDown, headroom);
      } else {
        // Effectively long Down after outstanding fills
        sellableDown = Math.min(this.inventoryDown, effectiveUnhedged + this.maxUnhedgedExposure);
        const headroom = Math.max(0, this.maxUnhedgedExposure - effectiveUnhedged);
        sellableUp = Math.min(this.inventoryUp, headroom);
      }

      // Subtract what's already posted — sellable is the NEW amount we can add
      sellableUp = Math.max(0, sellableUp - outstandingSellUp);
      sellableDown = Math.max(0, sellableDown - outstandingSellDown);
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
      bsStrike: this.bsStrike,
      bsVolSamples: this.bsPriceHistory.length,
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

  // ── Black-Scholes helpers ──

  /**
   * Update the price history buffer and compute realized vol.
   * Returns { spot, sigma, fairUp, fairDown, halfSpread } or null if insufficient data.
   */
  private bsUpdate(
    externalData: Record<string, import("../types.ts").ExternalDataPoint> | undefined,
    timeRemainingMs: number,
  ): {
    spot: number;
    sigma: number;
    fairUp: number;
    fairDown: number;
    halfSpread: number;
  } | null {
    if (!externalData) return null;
    const feed = externalData[this.bsExternalFeed];
    if (!feed) return null;

    const spot = feed.price;
    const now = feed.timestamp;

    // Capture strike on first price if not set in onMarketOpen
    if (this.bsStrike === 0) {
      this.bsStrike = spot;
      console.log(`[MM/BS] Strike set (deferred): $${this.bsStrike.toFixed(2)}`);
    }

    // Add to price history (keep only bsVolWindowS worth)
    this.bsPriceHistory.push({ ts: now, price: spot });
    const cutoff = now - this.bsVolWindowS * 1000;
    while (this.bsPriceHistory.length > 0 && this.bsPriceHistory[0]!.ts < cutoff) {
      this.bsPriceHistory.shift();
    }

    // Need at least 10 samples for a vol estimate
    if (this.bsPriceHistory.length < 10) return null;

    // Compute realized vol from log returns, sampled at 1-second intervals
    const sigma = this.computeRealizedVol();
    if (sigma <= 0) return null;

    // Time to expiry as fraction of a year
    const tau = Math.max(timeRemainingMs / (365.25 * 24 * 3600 * 1000), 1e-10);

    // Binary call fair value: N(d2) where d2 = (ln(S/K) + (r - σ²/2)τ) / (σ√τ)
    // r = 0 for short-duration contracts
    const sqrtTau = Math.sqrt(tau);
    const d2 = (Math.log(spot / this.bsStrike) - (sigma * sigma / 2) * tau) / (sigma * sqrtTau);
    const fairUp = this.bsTDof > 0 ? this.studentTCdf(d2, this.bsTDof) : this.normalCdf(d2);
    const fairDown = 1 - fairUp;

    // Half-spread proportional to uncertainty: σ√τ scaled to option-price space
    // The vega of a binary option ≈ n(d2)/(σ√τ), so spread ~ multiplier * n(d2)
    // Simpler: spread = multiplier * sigma * sqrt(tau) * scaling factor
    // Scale factor converts annualized vol to option-price movement
    const halfSpread = Math.max(
      this.bsMinSpread,
      this.bsSpreadMultiplier * sigma * sqrtTau,
    );

    return { spot, sigma, fairUp, fairDown, halfSpread };
  }

  /**
   * Compute annualized realized volatility from the price history buffer.
   * Uses 1-second sampled log returns.
   */
  private computeRealizedVol(): number {
    const h = this.bsPriceHistory;
    if (h.length < 10) return 0;

    // Sample at ~1 second intervals to avoid microstructure noise
    const sampleIntervalMs = 1000;
    const sampledPrices: number[] = [h[0]!.price];
    let lastSampleTs = h[0]!.ts;

    for (let i = 1; i < h.length; i++) {
      if (h[i]!.ts - lastSampleTs >= sampleIntervalMs) {
        sampledPrices.push(h[i]!.price);
        lastSampleTs = h[i]!.ts;
      }
    }

    if (sampledPrices.length < 5) return 0;

    // Log returns
    const returns: number[] = [];
    for (let i = 1; i < sampledPrices.length; i++) {
      returns.push(Math.log(sampledPrices[i]! / sampledPrices[i - 1]!));
    }

    // Variance of returns
    const mean = returns.reduce((s, r) => s + r, 0) / returns.length;
    const variance = returns.reduce((s, r) => s + (r - mean) ** 2, 0) / (returns.length - 1);

    // Annualize: each return spans ~1 second, so multiply by seconds-per-year
    const secondsPerYear = 365.25 * 24 * 3600;
    return Math.sqrt(variance * secondsPerYear);
  }

  /**
   * Approximation of the standard normal CDF.
   * Abramowitz & Stegun formula 26.2.17, accurate to ~1.5e-7.
   */
  private normalCdf(x: number): number {
    if (x > 6) return 1;
    if (x < -6) return 0;

    const a1 = 0.254829592;
    const a2 = -0.284496736;
    const a3 = 1.421413741;
    const a4 = -1.453152027;
    const a5 = 1.061405429;
    const p = 0.3275911;

    const sign = x < 0 ? -1 : 1;
    // A&S formula 7.1.26 — input must be |x|/√2
    const z = Math.abs(x) / Math.SQRT2;
    const t = 1.0 / (1.0 + p * z);
    const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-z * z);

    return 0.5 * (1.0 + sign * y);
  }

  /**
   * Student's t distribution CDF.
   * Uses the regularized incomplete beta function for accuracy.
   * Fat tails (low ν) assign more probability to extreme moves,
   * which widens fair-value uncertainty and thus our spread.
   *
   * @param x - the quantile (d2 in our BS model)
   * @param nu - degrees of freedom (lower = fatter tails; 3-5 typical for crypto)
   */
  private studentTCdf(x: number, nu: number): number {
    if (x === 0) return 0.5;

    const t2 = x * x;
    const betaVal = this.regIncBeta(nu / (nu + t2), nu / 2, 0.5);

    if (x > 0) {
      return 1 - 0.5 * betaVal;
    } else {
      return 0.5 * betaVal;
    }
  }

  /**
   * Regularized incomplete beta function I_x(a, b).
   * Continued fraction evaluation based on Numerical Recipes (betacf).
   */
  private regIncBeta(x: number, a: number, b: number): number {
    if (x <= 0) return 0;
    if (x >= 1) return 1;

    // Use symmetry relation when x > (a+1)/(a+b+2) for better convergence
    if (x > (a + 1) / (a + b + 2)) {
      return 1 - this.regIncBeta(1 - x, b, a);
    }

    const lnBeta = this.logBeta(a, b);
    const front = Math.exp(Math.log(x) * a + Math.log(1 - x) * b - lnBeta) / a;

    return front * this.betacf(a, b, x);
  }

  /** Continued fraction for incomplete beta (Numerical Recipes). */
  private betacf(a: number, b: number, x: number): number {
    const TINY = 1e-30;
    const EPS = 1e-14;
    const qab = a + b;
    const qap = a + 1;
    const qam = a - 1;

    let c = 1;
    let d = 1 - qab * x / qap;
    if (Math.abs(d) < TINY) d = TINY;
    d = 1 / d;
    let h = d;

    for (let m = 1; m <= 300; m++) {
      const m2 = 2 * m;

      // Even step
      let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
      d = 1 + aa * d;
      if (Math.abs(d) < TINY) d = TINY;
      c = 1 + aa / c;
      if (Math.abs(c) < TINY) c = TINY;
      d = 1 / d;
      h *= d * c;

      // Odd step
      aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
      d = 1 + aa * d;
      if (Math.abs(d) < TINY) d = TINY;
      c = 1 + aa / c;
      if (Math.abs(c) < TINY) c = TINY;
      d = 1 / d;
      const del = d * c;
      h *= del;

      if (Math.abs(del - 1) < EPS) break;
    }

    return h;
  }

  /**
   * Log of the beta function: ln(B(a,b)) = lnΓ(a) + lnΓ(b) - lnΓ(a+b)
   */
  private logBeta(a: number, b: number): number {
    return this.logGamma(a) + this.logGamma(b) - this.logGamma(a + b);
  }

  /**
   * Lanczos approximation to ln(Γ(x)), accurate to ~15 digits.
   */
  private logGamma(x: number): number {
    const g = 7;
    const coefs = [
      0.99999999999980993,
      676.5203681218851,
      -1259.1392167224028,
      771.32342877765313,
      -176.61502916214059,
      12.507343278686905,
      -0.13857109526572012,
      9.9843695780195716e-6,
      1.5056327351493116e-7,
    ];

    if (x < 0.5) {
      // Reflection formula: Γ(x)Γ(1-x) = π / sin(πx)
      return Math.log(Math.PI / Math.sin(Math.PI * x)) - this.logGamma(1 - x);
    }

    x -= 1;
    let sum = coefs[0]!;
    for (let i = 1; i < g + 2; i++) {
      sum += coefs[i]! / (x + i);
    }
    const t = x + g + 0.5;
    return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(sum);
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
    mm.bsVolWindowS ?? 300,
    mm.bsSpreadMultiplier ?? 1.0,
    mm.bsMinSpread ?? 0.02,
    mm.bsExternalFeed ?? "binance-btcusdt",
    mm.bsTDof ?? 4,
  );
});
