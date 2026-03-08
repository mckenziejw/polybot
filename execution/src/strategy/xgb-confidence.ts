import type {
  MarketState,
  TradeAction,
  TradeMetadata,
  DataMode,
  Side,
  OrderBookState,
  ExternalDataPoint,
} from "../types.ts";
import type { Strategy } from "./strategy.ts";
import { registerStrategy } from "./strategy.ts";
import { loadConfig } from "../config.ts";

/**
 * XGBoost v3 confidence strategy for live trading.
 *
 * Accumulates orderbook snapshots (1/second) and BTC prices over the first
 * 120 seconds of each 5-minute market. At t=120s, sends all accumulated data
 * to the Python model server for feature extraction + XGBoost inference.
 *
 * If the model returns prediction >= 0.70 and the BTC volatility filter passes,
 * places a single BUY order for the leader token (whichever token has mid >= 0.50).
 * Holds to binary settlement ($1 or $0).
 *
 * The model server (model_server.py) uses the exact same feature extraction code
 * as training, eliminating train/serve skew.
 *
 * Config:
 *   execution.betDollars: dollar amount per trade (default 5)
 *   execution.strategyId: "xgb-confidence"
 *   execution.externalFeeds: ["binance-btcusdt"]
 */

// --- Types for model server communication ---

interface BookSnapshot {
  timestamp_ms: number;
  asset_id: string;
  mid_price: number;
  spread: number;
  book_imbalance: number;
  bid_prices: number[];
  bid_sizes: number[];
  ask_prices: number[];
  ask_sizes: number[];
}

interface BtcPricePoint {
  timestamp_ms: number;
  mid_price: number;
  spread: number;
}

interface PredictRequest {
  market_open_ms: number;
  observe_s: number;
  up_snapshots: BookSnapshot[];
  down_snapshots: BookSnapshot[];
  btc_prices: BtcPricePoint[];
}

interface PredictResponse {
  prediction: number;
  should_trade: boolean;
  buy_token: "up" | "down";
  leader_label: string;
  leader_asset_id: string;
  entry_price: number;
  btc_vol_5m: number;
  vol_threshold: number | null;
  vol_pass: boolean;
  confidence_threshold: number;
  error?: string;
  reason?: string;
}

// --- Constants ---

const MODEL_SERVER_URL = "http://127.0.0.1:8000";
const MODEL_TIMEOUT_MS = 5000; // 5s — model inference is not latency-critical
const OBSERVATION_TIME_S = 120;
const SNAPSHOT_INTERVAL_MS = 1000; // record one snapshot per second
const MIN_TOKENS = 5; // Polymarket minimum order size
const BTC_FEED_NAME = "binance-btcusdt";

// How long before market end we need to place the order (buffer for fills)
const MIN_REMAINING_MS = 150_000; // 2.5 minutes — need time for settlement

// BTC lookback buffer: keep 1 hour of BTC prices across markets so the model
// has data for btc_ret_3600s, btc_ret_900s, btc_vol_15m, etc.
const BTC_BUFFER_MAX_AGE_MS = 3_700_000; // ~61 minutes (slight buffer)

type Phase =
  | "ACCUMULATING" // collecting snapshots during observation window
  | "DECIDING"     // called model server, waiting for response
  | "TRADED"       // order placed, holding to settlement
  | "SKIP";        // decided not to trade this market

export class XgbConfidenceStrategy implements Strategy {
  readonly id = "xgb-confidence";
  readonly dataMode: DataMode = "tick";

  // Per-market state (reset on each market open)
  private phase: Phase = "ACCUMULATING";
  private marketOpenMs = 0;
  private lastSnapshotMs = 0;
  private upSnapshots: BookSnapshot[] = [];
  private downSnapshots: BookSnapshot[] = [];
  private decisionMade = false;
  private betDollars: number;

  // Persistent across markets: BTC price ring buffer (1 hour rolling)
  // NOT reset on market open — the model needs pre-market BTC history
  private btcBuffer: BtcPricePoint[] = [];
  private lastBtcTimestamp = 0;

  constructor(betDollars = 20) {
    this.betDollars = betDollars;
  }

  async onMarketOpen(state: MarketState): Promise<void> {
    this.phase = "ACCUMULATING";
    // Use actual wall-clock time as market open, not endTime - 300s,
    // to avoid issues if markets open late or durations change
    this.marketOpenMs = Date.now();
    this.lastSnapshotMs = 0;
    this.upSnapshots = [];
    this.downSnapshots = [];
    this.decisionMade = false;

    // Prune BTC buffer: keep only last ~1 hour
    this.pruneBtcBuffer();

    console.log(
      `[XgbConf] Market open: ${state.market.slug}, ` +
      `collecting snapshots for ${OBSERVATION_TIME_S}s ` +
      `(btc buffer: ${this.btcBuffer.length} points)`
    );
  }

  async evaluate(state: MarketState): Promise<TradeAction[]> {
    const { market, books, externalData, timeRemainingMs } = state;
    const now = Date.now();
    const elapsedMs = now - this.marketOpenMs;
    const elapsedS = elapsedMs / 1000;

    // --- Phase: SKIP or TRADED — do nothing ---
    if (this.phase === "SKIP" || this.phase === "TRADED") {
      // Still accumulate BTC data even when not trading, to maintain the buffer
      this.recordBtcPrice(externalData);
      return [{ type: "NOOP" }];
    }

    // --- Phase: DECIDING — already sent request, waiting ---
    if (this.phase === "DECIDING") {
      return [{ type: "NOOP" }];
    }

    // --- Phase: ACCUMULATING — collect snapshots ---

    // Always record BTC prices (persistent buffer across markets)
    this.recordBtcPrice(externalData);

    // Record orderbook snapshots once per second
    if (now - this.lastSnapshotMs >= SNAPSHOT_INTERVAL_MS) {
      this.lastSnapshotMs = now;

      const upBook = books[market.upTokenId];
      const downBook = books[market.downTokenId];

      if (upBook && upBook.bids.length > 0 && upBook.asks.length > 0) {
        this.upSnapshots.push(bookToSnapshot(upBook, now, market.upTokenId));
      }
      if (downBook && downBook.bids.length > 0 && downBook.asks.length > 0) {
        this.downSnapshots.push(bookToSnapshot(downBook, now, market.downTokenId));
      }
    }

    // --- Decision time: t >= 120s ---
    if (elapsedS >= OBSERVATION_TIME_S && !this.decisionMade) {
      this.decisionMade = true;
      this.phase = "DECIDING";

      // Guard: need minimum data
      if (this.upSnapshots.length < 10 || this.downSnapshots.length < 10) {
        console.log(
          `[XgbConf] Insufficient data: ${this.upSnapshots.length} up, ` +
          `${this.downSnapshots.length} down snapshots — skipping`
        );
        this.phase = "SKIP";
        return [{ type: "NOOP" }];
      }

      // Guard: enough time remaining
      if (timeRemainingMs < MIN_REMAINING_MS) {
        console.log(`[XgbConf] Not enough time remaining (${(timeRemainingMs / 1000).toFixed(0)}s) — skipping`);
        this.phase = "SKIP";
        return [{ type: "NOOP" }];
      }

      // Call model server
      const actions = await this.callModelServer(state);
      return actions;
    }

    return [{ type: "NOOP" }];
  }

  async onMarketClose(state: MarketState): Promise<TradeAction[]> {
    const snap = `up=${this.upSnapshots.length} down=${this.downSnapshots.length} btc=${this.btcBuffer.length}`;
    console.log(`[XgbConf] Market close (phase=${this.phase}, ${snap})`);
    this.phase = "SKIP";
    // Cancel any open orders (safety net — shouldn't have any if holding to settlement)
    return [{ type: "CANCEL_ALL" }];
  }

  // --- BTC price buffer management ---

  private recordBtcPrice(externalData?: Record<string, ExternalDataPoint>): void {
    if (!externalData) return;
    const btc = externalData[BTC_FEED_NAME];
    if (!btc || btc.timestamp === this.lastBtcTimestamp) return;

    this.lastBtcTimestamp = btc.timestamp;
    this.btcBuffer.push({
      timestamp_ms: btc.timestamp,
      mid_price: btc.price,
      spread: (btc.ask ?? btc.price) - (btc.bid ?? btc.price),
    });
  }

  private pruneBtcBuffer(): void {
    const cutoff = Date.now() - BTC_BUFFER_MAX_AGE_MS;
    // Find first index that's >= cutoff
    let pruneIdx = 0;
    while (pruneIdx < this.btcBuffer.length && this.btcBuffer[pruneIdx]!.timestamp_ms < cutoff) {
      pruneIdx++;
    }
    if (pruneIdx > 0) {
      this.btcBuffer = this.btcBuffer.slice(pruneIdx);
    }
  }

  // --- Model server communication ---

  private async callModelServer(state: MarketState): Promise<TradeAction[]> {
    const request: PredictRequest = {
      market_open_ms: this.marketOpenMs,
      observe_s: OBSERVATION_TIME_S,
      up_snapshots: this.upSnapshots,
      down_snapshots: this.downSnapshots,
      // Send the full BTC buffer (includes pre-market history)
      btc_prices: this.btcBuffer,
    };

    try {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), MODEL_TIMEOUT_MS);

      const response = await fetch(`${MODEL_SERVER_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(request),
        signal: controller.signal,
      });

      clearTimeout(timer);

      if (!response.ok) {
        const text = await response.text().catch(() => "");
        console.warn(`[XgbConf] Model server error ${response.status}: ${text}`);
        this.phase = "SKIP";
        return [{ type: "NOOP" }];
      }

      const result = (await response.json()) as PredictResponse;

      console.log(
        `[XgbConf] Prediction: ${result.prediction.toFixed(3)}, ` +
        `leader=${result.leader_label}, entry=${result.entry_price.toFixed(3)}, ` +
        `should_trade=${result.should_trade}`
      );

      if (result.error) {
        console.warn(`[XgbConf] Model error: ${result.reason}`);
        this.phase = "SKIP";
        return [{ type: "NOOP" }];
      }

      if (!result.should_trade) {
        console.log(`[XgbConf] Skipping: low confidence (${result.prediction.toFixed(3)} < ${result.confidence_threshold})`);
        this.phase = "SKIP";
        return [{ type: "NOOP" }];
      }

      // --- Place trade ---
      return this.placeTrade(state, result);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      console.warn(`[XgbConf] Model request failed: ${msg}`);
      this.phase = "SKIP";
      return [{ type: "NOOP" }];
    }
  }

  private placeTrade(state: MarketState, result: PredictResponse): TradeAction[] {
    const { market, books } = state;

    // Determine which token to buy
    const buyAssetId = result.buy_token === "up"
      ? market.upTokenId
      : market.downTokenId;

    const book = books[buyAssetId];
    if (!book || book.asks.length === 0 || book.bids.length === 0) {
      console.warn(`[XgbConf] No book for ${result.buy_token} token — skipping`);
      this.phase = "SKIP";
      return [{ type: "NOOP" }];
    }

    const bestBid = book.bids[0]!.price;
    const bestAsk = book.asks[0]!.price;
    const spread = bestAsk - bestBid;

    // Sanity checks
    if (spread > 0.08) {
      console.warn(`[XgbConf] Spread too wide (${spread.toFixed(3)}) — skipping`);
      this.phase = "SKIP";
      return [{ type: "NOOP" }];
    }

    // Leader token should have mid >= 0.50, so ask should be in a reasonable range.
    // Reject extreme prices (very cheap = low prob, very expensive = no upside).
    // Cap at 0.90: walk-forward shows 0.90+ entries barely profitable ($0.07/trade),
    // and capping improves ROI from 14.0% to 15.3% with minimal PnL loss.
    if (bestAsk > 0.90 || bestAsk < 0.45) {
      console.warn(`[XgbConf] Ask price ${bestAsk} out of range [0.45, 0.90] — skipping`);
      this.phase = "SKIP";
      return [{ type: "NOOP" }];
    }

    // Place a limit buy at best ask (taker-style for guaranteed fill).
    // Using FOK would be ideal but Polymarket's CLOB converts FOK to GTC for
    // limit orders. GTC at best ask will fill immediately if liquidity exists;
    // if the ask is pulled, it becomes a resting bid which onMarketClose cancels.
    const entryPrice = bestAsk;

    // Size: betDollars worth of tokens, respecting minimum
    const rawSize = Math.floor(this.betDollars / entryPrice);
    const size = Math.max(rawSize, MIN_TOKENS);

    this.phase = "TRADED";
    const label = result.buy_token === "up" ? "Up" : "Down";
    console.log(
      `[XgbConf] BUY ${size} ${label} @ ${entryPrice.toFixed(2)} ` +
      `(pred=${result.prediction.toFixed(3)}, spread=${spread.toFixed(3)}, ` +
      `cost=$${(size * entryPrice).toFixed(2)})`
    );

    const metadata: TradeMetadata = {
      prediction: result.prediction,
      confidenceThreshold: result.confidence_threshold,
      btcVol5m: result.btc_vol_5m,
      volThreshold: result.vol_threshold,
      volPass: result.vol_pass,
      shouldTrade: result.should_trade,
      leaderLabel: result.leader_label,
      buyToken: result.buy_token,
      strategyId: this.id,
    };

    return [
      {
        type: "PLACE_ORDER",
        assetId: buyAssetId,
        side: "BUY" as Side,
        price: entryPrice,
        size,
        orderType: "GTC",
        metadata,
      },
    ];
  }
}

// --- Helpers ---

function bookToSnapshot(
  book: OrderBookState,
  timestampMs: number,
  assetId: string,
): BookSnapshot {
  const bestBid = book.bids[0]?.price ?? 0;
  const bestAsk = book.asks[0]?.price ?? 0;
  const mid = bestBid > 0 && bestAsk > 0 ? (bestBid + bestAsk) / 2 : 0;
  const spread = bestAsk > 0 && bestBid > 0 ? bestAsk - bestBid : 0;

  // Book imbalance: (total bid size - total ask size) / total
  let bidTotal = 0;
  let askTotal = 0;
  for (const level of book.bids.slice(0, 5)) bidTotal += level.size;
  for (const level of book.asks.slice(0, 5)) askTotal += level.size;
  const total = bidTotal + askTotal;
  const imbalance = total > 0 ? (bidTotal - askTotal) / total : 0;

  // Extract top-10 levels (model expects up to 10)
  const bidPrices: number[] = [];
  const bidSizes: number[] = [];
  const askPrices: number[] = [];
  const askSizes: number[] = [];

  for (let i = 0; i < 10; i++) {
    bidPrices.push(book.bids[i]?.price ?? 0);
    bidSizes.push(book.bids[i]?.size ?? 0);
    askPrices.push(book.asks[i]?.price ?? 0);
    askSizes.push(book.asks[i]?.size ?? 0);
  }

  return {
    timestamp_ms: timestampMs,
    asset_id: assetId,
    mid_price: mid,
    spread,
    book_imbalance: imbalance,
    bid_prices: bidPrices,
    bid_sizes: bidSizes,
    ask_prices: askPrices,
    ask_sizes: askSizes,
  };
}

registerStrategy("xgb-confidence", () => {
  const config = loadConfig();
  return new XgbConfidenceStrategy(config.execution.betDollars);
});
