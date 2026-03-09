import type { ResampledBar, IndicatorSnapshot } from "../types.ts";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const SMA_PERIOD = 20;
const EMA_PERIOD = 20;
const EMA_ALPHA = 2 / (EMA_PERIOD + 1);
const VOL_PERIOD = 50;  // lookback for realized volatility (returns)
const RSI_PERIOD = 14;
const SPREAD_SMA_PERIOD = 10;

/** Circular buffer capacity — must be >= max lookback needed across all indicators. */
const BUFFER_CAPACITY = VOL_PERIOD;

// ---------------------------------------------------------------------------
// Circular buffer
// ---------------------------------------------------------------------------

/**
 * Fixed-capacity circular buffer.  Oldest element is overwritten when full.
 * Provides O(1) push and O(n) slice for the last n entries.
 */
class CircularBuffer {
  private readonly buf: ResampledBar[];
  private head = 0;   // next write position
  private _count = 0; // number of valid entries (saturates at capacity)

  constructor(private readonly capacity: number) {
    this.buf = new Array(capacity);
  }

  push(bar: ResampledBar): void {
    this.buf[this.head] = bar;
    this.head = (this.head + 1) % this.capacity;
    if (this._count < this.capacity) this._count++;
  }

  get count(): number {
    return this._count;
  }

  /**
   * Return the last `n` bars in chronological order (oldest first).
   * Clamps n to the number of available entries.
   */
  last(n: number): ResampledBar[] {
    const take = Math.min(n, this._count);
    const result: ResampledBar[] = new Array(take);
    // Most-recent entry is at (head - 1 + capacity) % capacity.
    // Entry i steps back is at (head - 1 - i + capacity * k) % capacity.
    for (let i = 0; i < take; i++) {
      const idx = (this.head - take + i + this.capacity * 2) % this.capacity;
      result[i] = this.buf[idx]!;
    }
    return result;
  }

  clear(): void {
    this.head = 0;
    this._count = 0;
  }
}

// ---------------------------------------------------------------------------
// IndicatorEngine
// ---------------------------------------------------------------------------

/**
 * Rolling indicators computed on resampled bars.
 *
 * All indicators operate on midPrice unless noted.  Returns 0 for any
 * indicator that does not have enough data yet.
 *
 * Indicators:
 *   - SMA(20)            — simple moving average of last 20 midPrices
 *   - EMA(20)            — exponential moving average, alpha = 2/(20+1)
 *   - Realized Vol       — std dev of log returns over last 50 bars
 *   - RSI(14)            — Wilder's smoothed RSI over 14 bars
 *   - Spread SMA(10)     — simple moving average of last 10 spreads
 */
export class IndicatorEngine {
  private buffer = new CircularBuffer(BUFFER_CAPACITY);

  // EMA state
  private ema: number | null = null;

  // RSI state (Wilder's smoothing)
  private avgGain: number | null = null;
  private avgLoss: number | null = null;
  private prevMid: number | null = null;
  private rsiBarCount = 0; // bars fed into RSI so far

  constructor() {}

  /**
   * Feed a new resampled bar.  Updates all rolling indicators.
   */
  update(bar: ResampledBar): void {
    // --- EMA update (before pushing so we can read prevMid first) ---
    if (this.ema === null) {
      this.ema = bar.midPrice;
    } else {
      this.ema = EMA_ALPHA * bar.midPrice + (1 - EMA_ALPHA) * this.ema;
    }

    // --- RSI update ---
    this._updateRsi(bar.midPrice);

    // Push bar into circular buffer (used for SMA, vol, spread SMA)
    this.buffer.push(bar);
  }

  /**
   * Return current indicator values.  Any indicator without enough data
   * returns 0.
   */
  snapshot(): IndicatorSnapshot {
    return {
      midPriceSma20: this._sma(SMA_PERIOD),
      midPriceEma20: this.ema ?? 0,
      realizedVol: this._realizedVol(),
      rsi14: this._rsi(),
      spreadSma10: this._spreadSma(SPREAD_SMA_PERIOD),
    };
  }

  /** Reset all state (call on market rotation). */
  reset(): void {
    this.buffer.clear();
    this.ema = null;
    this.avgGain = null;
    this.avgLoss = null;
    this.prevMid = null;
    this.rsiBarCount = 0;
  }

  // ---------------------------------------------------------------------------
  // Private indicator helpers
  // ---------------------------------------------------------------------------

  /** Simple moving average of midPrice over the last n bars. */
  private _sma(n: number): number {
    const bars = this.buffer.last(n);
    if (bars.length < n) return 0;
    let sum = 0;
    for (const b of bars) sum += b.midPrice;
    return sum / n;
  }

  /** Simple moving average of spread over the last n bars. */
  private _spreadSma(n: number): number {
    const bars = this.buffer.last(n);
    if (bars.length < n) return 0;
    let sum = 0;
    for (const b of bars) sum += b.spread;
    return sum / n;
  }

  /**
   * Realized volatility: population std dev of simple returns
   * (mid[i] / mid[i-1] - 1) over the last VOL_PERIOD bars.
   *
   * We need VOL_PERIOD bars to compute VOL_PERIOD-1 returns, but we
   * require at least VOL_PERIOD bars in the buffer before reporting.
   */
  private _realizedVol(): number {
    const bars = this.buffer.last(VOL_PERIOD);
    if (bars.length < VOL_PERIOD) return 0;

    // Build return series (length = VOL_PERIOD - 1)
    const returns: number[] = [];
    for (let i = 1; i < bars.length; i++) {
      const prev = bars[i - 1]!.midPrice;
      if (prev === 0) continue;
      returns.push(bars[i]!.midPrice / prev - 1);
    }
    if (returns.length === 0) return 0;

    const n = returns.length;
    const mean = returns.reduce((a, b) => a + b, 0) / n;
    const variance = returns.reduce((acc, r) => acc + (r - mean) ** 2, 0) / n;
    return Math.sqrt(variance);
  }

  /**
   * Wilder's RSI(14).
   *
   * Phase 1: first RSI_PERIOD+1 prices → compute initial avg gain / avg loss
   *          as simple averages over the first RSI_PERIOD changes.
   * Phase 2: subsequent bars → Wilder's smoothing:
   *          avgGain = (prevAvgGain * (RSI_PERIOD - 1) + gain) / RSI_PERIOD
   *
   * The RSI state is maintained incrementally here; the circular buffer is
   * not used for RSI so that we avoid iterating over old bars on every bar.
   */
  private _updateRsi(mid: number): void {
    if (this.prevMid === null) {
      this.prevMid = mid;
      this.rsiBarCount = 1;
      return;
    }

    const change = mid - this.prevMid;
    const gain = change > 0 ? change : 0;
    const loss = change < 0 ? -change : 0;
    this.prevMid = mid;
    this.rsiBarCount++;

    if (this.avgGain === null) {
      // Still in seed accumulation phase.
      // We need RSI_PERIOD + 1 prices (RSI_PERIOD changes) to seed.
      // Use the circular buffer to collect the first RSI_PERIOD changes.
      // We accumulate into a temporary array stored in the buffer itself —
      // but since CircularBuffer holds bars, we track manually via a seed array.
      this._seedGains.push(gain);
      this._seedLosses.push(loss);

      if (this._seedGains.length === RSI_PERIOD) {
        // Compute initial averages (simple mean over first RSI_PERIOD periods)
        const sumGain = this._seedGains.reduce((a, b) => a + b, 0);
        const sumLoss = this._seedLosses.reduce((a, b) => a + b, 0);
        this.avgGain = sumGain / RSI_PERIOD;
        this.avgLoss = sumLoss / RSI_PERIOD;
        // Free the seed arrays
        this._seedGains = [];
        this._seedLosses = [];
      }
    } else {
      // Wilder's smoothing
      this.avgGain = (this.avgGain * (RSI_PERIOD - 1) + gain) / RSI_PERIOD;
      this.avgLoss = (this.avgLoss! * (RSI_PERIOD - 1) + loss) / RSI_PERIOD;
    }
  }

  // Seed arrays for the initial RSI calculation (discarded after seeding)
  private _seedGains: number[] = [];
  private _seedLosses: number[] = [];

  private _rsi(): number {
    if (this.avgGain === null || this.avgLoss === null) return 0;
    if (this.avgLoss === 0) return 100;
    const rs = this.avgGain / this.avgLoss;
    return 100 - 100 / (1 + rs);
  }
}
