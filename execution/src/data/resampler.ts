import type { ResampledBar } from "../types.ts";
import { OrderBook } from "./orderbook.ts";

const MAX_STALENESS_MS = 5_000;
const TOP_LEVELS = 5;

interface AssetState {
  /** The latest OrderBook snapshot received for this asset. */
  book: OrderBook;
  /** Whether a new tick arrived since the last emit(). */
  hasNewTick: boolean;
  /** Cumulative ms since the last real tick (forward-fill runs). */
  stalenessMs: number;
}

/**
 * Resamples raw orderbook ticks into fixed-interval bars.
 *
 * Replicates the algorithm from preprocess_markets.py:
 *   - Each asset is assigned to a bar when a tick arrives (forward-fill on emit
 *     if no new tick came in).
 *   - staleness_ms accumulates across consecutive no-tick intervals and is
 *     capped at MAX_STALENESS_MS (5 000 ms).
 *   - Top-5 bid/ask levels are extracted per bar, padded with zeros when
 *     fewer levels exist in the book.
 */
export class Resampler {
  private readonly intervalMs: number;
  private assets: Map<string, AssetState> = new Map();

  constructor(intervalMs: number = 100) {
    this.intervalMs = intervalMs;
  }

  /**
   * Feed a new orderbook tick. Called on every book snapshot.
   * The caller retains ownership of `book`; we only read from it here
   * and record a reference until the next emit().
   */
  tick(assetId: string, book: OrderBook): void {
    const existing = this.assets.get(assetId);
    if (existing) {
      existing.book = book;
      existing.hasNewTick = true;
    } else {
      this.assets.set(assetId, {
        book,
        hasNewTick: true,
        stalenessMs: 0,
      });
    }
  }

  /**
   * Called by the orchestrator on a timer every intervalMs.
   *
   * For each tracked asset:
   *   - If a new tick arrived since the last emit: use it, reset staleness to 0.
   *   - Otherwise: forward-fill from last known state, increment staleness by
   *     intervalMs (capped at MAX_STALENESS_MS).
   *
   * Returns a ResampledBar for every asset that has at least one tick.
   * Assets with no data yet are silently omitted.
   */
  emit(): ResampledBar[] {
    const now = Date.now();
    const bars: ResampledBar[] = [];

    for (const [assetId, state] of this.assets) {
      if (state.hasNewTick) {
        // Real tick arrived: reset staleness.
        state.stalenessMs = 0;
        state.hasNewTick = false;
      } else {
        // No new tick: forward-fill, accumulate staleness.
        state.stalenessMs = Math.min(
          state.stalenessMs + this.intervalMs,
          MAX_STALENESS_MS,
        );
      }

      const bar = this._buildBar(assetId, state, now);
      bars.push(bar);
    }

    return bars;
  }

  /** Reset all stored state (call on market rotation). */
  reset(): void {
    this.assets.clear();
  }

  // ---------------------------------------------------------------------------
  // Private helpers
  // ---------------------------------------------------------------------------

  private _buildBar(
    assetId: string,
    state: AssetState,
    timestamp: number,
  ): ResampledBar {
    const book = state.book;
    const { bidPrice, bidSize, askPrice, askSize } = book.topLevels(TOP_LEVELS);

    return {
      timestamp,
      assetId,
      midPrice: book.mid,
      spread: book.spread,
      bookImbalance: book.imbalance,
      bidPrice,
      bidSize,
      askPrice,
      askSize,
      stalenessMs: state.stalenessMs,
    };
  }
}
