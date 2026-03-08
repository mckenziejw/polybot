import { describe, test, expect, beforeEach } from "bun:test";
import { IndicatorEngine } from "../src/data/indicators.ts";
import { OrderBook } from "../src/data/orderbook.ts";
import type { ResampledBar, OrderBookState } from "../src/types.ts";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Build a minimal ResampledBar with the given midPrice and spread.
 * All array fields are zero-padded to length 5.
 */
function makeBar(midPrice: number, spread = 0.01, assetId = "asset-1"): ResampledBar {
  const zeros = [0, 0, 0, 0, 0];
  return {
    timestamp: Date.now(),
    assetId,
    midPrice,
    spread,
    bookImbalance: 0,
    bidPrice: zeros,
    bidSize: zeros,
    askPrice: zeros,
    askSize: zeros,
    stalenessMs: 0,
  };
}

/**
 * Feed `n` bars of the same midPrice to an engine.
 */
function feedN(engine: IndicatorEngine, n: number, midPrice: number, spread = 0.01): void {
  for (let i = 0; i < n; i++) {
    engine.update(makeBar(midPrice, spread));
  }
}

// ---------------------------------------------------------------------------
// IndicatorEngine
// ---------------------------------------------------------------------------

describe("IndicatorEngine", () => {
  let engine: IndicatorEngine;

  beforeEach(() => {
    engine = new IndicatorEngine();
  });

  // ── SMA(20) ───────────────────────────────────────────────────────────────

  describe("SMA(20)", () => {
    test("returns 0 when fewer than 20 bars have been fed", () => {
      feedN(engine, 19, 0.5);
      expect(engine.snapshot().midPriceSma20).toBe(0);
    });

    test("returns 0 after exactly 19 bars", () => {
      feedN(engine, 19, 0.5);
      expect(engine.snapshot().midPriceSma20).toBe(0);
    });

    test("returns the correct average at exactly 20 bars (all same price)", () => {
      feedN(engine, 20, 0.5);
      expect(engine.snapshot().midPriceSma20).toBeCloseTo(0.5, 10);
    });

    test("returns the correct average at exactly 20 bars (varying prices)", () => {
      // Feed bars 1..20; average = (1+2+...+20)/20 = 210/20 = 10.5
      for (let i = 1; i <= 20; i++) {
        engine.update(makeBar(i));
      }
      expect(engine.snapshot().midPriceSma20).toBeCloseTo(10.5, 10);
    });

    test("slides correctly: after 21 bars the oldest price is dropped", () => {
      // Bars 1-19 at price=1, bar 20 at price=1 → SMA=1
      feedN(engine, 20, 1);
      // Now push one bar at price=21 → window becomes [bars 2-20 at 1, bar 21 at 21]
      // But all existing bars are at 1, so the window is 19×1 + 21 = 40 → avg = 2
      engine.update(makeBar(21));
      expect(engine.snapshot().midPriceSma20).toBeCloseTo(2, 10);
    });
  });

  // ── EMA(20) ───────────────────────────────────────────────────────────────

  describe("EMA(20)", () => {
    const ALPHA = 2 / (20 + 1); // ≈ 0.09524

    test("first bar initialises EMA to midPrice", () => {
      engine.update(makeBar(0.6));
      expect(engine.snapshot().midPriceEma20).toBeCloseTo(0.6, 10);
    });

    test("second bar applies EMA formula", () => {
      engine.update(makeBar(0.6));
      engine.update(makeBar(0.7));
      const expected = ALPHA * 0.7 + (1 - ALPHA) * 0.6;
      expect(engine.snapshot().midPriceEma20).toBeCloseTo(expected, 10);
    });

    test("returns 0 before any bar is fed", () => {
      expect(engine.snapshot().midPriceEma20).toBe(0);
    });

    test("EMA on constant price converges to that price", () => {
      feedN(engine, 100, 0.5);
      expect(engine.snapshot().midPriceEma20).toBeCloseTo(0.5, 6);
    });

    test("EMA tracks a price shift upward", () => {
      feedN(engine, 40, 0.5); // warm up at 0.5
      const emaBefore = engine.snapshot().midPriceEma20;
      feedN(engine, 10, 0.9); // shift price higher
      expect(engine.snapshot().midPriceEma20).toBeGreaterThan(emaBefore);
    });
  });

  // ── RSI(14) ───────────────────────────────────────────────────────────────

  describe("RSI(14)", () => {
    /**
     * RSI seeding:
     *   - Bar 1: stored as prevMid (rsiBarCount=1), no change yet.
     *   - Bars 2-15: 14 changes accumulate in _seedGains/_seedLosses.
     *     When _seedGains.length reaches RSI_PERIOD (14), avgGain/avgLoss are set.
     *   So RSI is first available after feeding 15 bars total.
     */

    test("returns 0 when fewer than 15 bars have been fed", () => {
      feedN(engine, 14, 0.5);
      expect(engine.snapshot().rsi14).toBe(0);
    });

    test("returns 0 after exactly 14 bars (not yet seeded)", () => {
      feedN(engine, 14, 0.5);
      expect(engine.snapshot().rsi14).toBe(0);
    });

    test("returns a value after 15 bars (seeded)", () => {
      feedN(engine, 15, 0.5);
      // When all prices are equal, gain=loss=0 → avgLoss=0 → returns 100 by convention
      // (avgLoss===0 branch returns 100)
      expect(engine.snapshot().rsi14).toBe(100);
    });

    test("returns 100 when all 14 changes are gains (only upward moves)", () => {
      // First bar as baseline, then 14 strictly increasing bars
      engine.update(makeBar(0.50));
      for (let i = 1; i <= 14; i++) {
        engine.update(makeBar(0.50 + i * 0.01)); // 0.51, 0.52, ... 0.64
      }
      // All gains → avgLoss=0 → RSI=100
      expect(engine.snapshot().rsi14).toBe(100);
    });

    test("returns 0 when all 14 changes are losses (only downward moves)", () => {
      // First bar as baseline, then 14 strictly decreasing bars
      engine.update(makeBar(0.64));
      for (let i = 1; i <= 14; i++) {
        engine.update(makeBar(0.64 - i * 0.01)); // 0.63, 0.62, ... 0.50
      }
      // All losses → avgGain=0 → RS=0 → RSI = 100 - 100/(1+0) = 0
      expect(engine.snapshot().rsi14).toBe(0);
    });

    test("RSI is between 0 and 100 for mixed moves", () => {
      // Alternating up/down
      let price = 0.5;
      for (let i = 0; i < 30; i++) {
        price += i % 2 === 0 ? 0.01 : -0.005;
        engine.update(makeBar(price));
      }
      const rsi = engine.snapshot().rsi14;
      expect(rsi).toBeGreaterThanOrEqual(0);
      expect(rsi).toBeLessThanOrEqual(100);
    });

    test("RSI > 50 after sustained upward trend", () => {
      engine.update(makeBar(0.50));
      for (let i = 1; i <= 30; i++) {
        engine.update(makeBar(0.50 + i * 0.01));
      }
      expect(engine.snapshot().rsi14).toBeGreaterThan(50);
    });

    test("RSI < 50 after sustained downward trend", () => {
      engine.update(makeBar(0.80));
      for (let i = 1; i <= 30; i++) {
        engine.update(makeBar(0.80 - i * 0.01));
      }
      expect(engine.snapshot().rsi14).toBeLessThan(50);
    });
  });

  // ── Realized Volatility ───────────────────────────────────────────────────

  describe("Realized Volatility", () => {
    test("returns 0 before 50 bars have been fed", () => {
      feedN(engine, 49, 0.5);
      expect(engine.snapshot().realizedVol).toBe(0);
    });

    test("returns 0 at exactly 49 bars", () => {
      feedN(engine, 49, 0.5);
      expect(engine.snapshot().realizedVol).toBe(0);
    });

    test("returns 0 at exactly 50 bars with constant price (zero variance)", () => {
      feedN(engine, 50, 0.5);
      expect(engine.snapshot().realizedVol).toBeCloseTo(0, 10);
    });

    test("returns non-zero after 50 bars with varying prices", () => {
      // Alternate between two prices to generate non-zero returns
      for (let i = 0; i < 50; i++) {
        engine.update(makeBar(i % 2 === 0 ? 0.5 : 0.55));
      }
      expect(engine.snapshot().realizedVol).toBeGreaterThan(0);
    });

    test("higher price variance → higher realized vol", () => {
      const engineLow = new IndicatorEngine();
      const engineHigh = new IndicatorEngine();

      // Low variance: alternates between 0.50 and 0.51
      for (let i = 0; i < 50; i++) {
        engineLow.update(makeBar(i % 2 === 0 ? 0.50 : 0.51));
      }

      // High variance: alternates between 0.30 and 0.70
      for (let i = 0; i < 50; i++) {
        engineHigh.update(makeBar(i % 2 === 0 ? 0.30 : 0.70));
      }

      expect(engineHigh.snapshot().realizedVol).toBeGreaterThan(
        engineLow.snapshot().realizedVol,
      );
    });
  });

  // ── Spread SMA(10) ────────────────────────────────────────────────────────

  describe("Spread SMA(10)", () => {
    test("returns 0 when fewer than 10 bars have been fed", () => {
      feedN(engine, 9, 0.5, 0.02);
      expect(engine.snapshot().spreadSma10).toBe(0);
    });

    test("returns 0 after exactly 9 bars", () => {
      feedN(engine, 9, 0.5, 0.02);
      expect(engine.snapshot().spreadSma10).toBe(0);
    });

    test("returns correct average at exactly 10 bars (all same spread)", () => {
      feedN(engine, 10, 0.5, 0.02);
      expect(engine.snapshot().spreadSma10).toBeCloseTo(0.02, 10);
    });

    test("returns correct average with varying spreads", () => {
      // 5 bars with spread=0.01, 5 bars with spread=0.03 → average=0.02
      feedN(engine, 5, 0.5, 0.01);
      feedN(engine, 5, 0.5, 0.03);
      expect(engine.snapshot().spreadSma10).toBeCloseTo(0.02, 10);
    });

    test("slides correctly after more than 10 bars", () => {
      // 10 bars at spread=0.01, then 1 bar at spread=0.11
      // The window after 11 bars is [9 bars×0.01, 1 bar×0.11] → avg=(0.09+0.11)/10=0.02
      feedN(engine, 10, 0.5, 0.01);
      engine.update(makeBar(0.5, 0.11));
      expect(engine.snapshot().spreadSma10).toBeCloseTo(0.02, 10);
    });
  });

  // ── reset() ───────────────────────────────────────────────────────────────

  describe("reset()", () => {
    test("clears the buffer so SMA returns 0 after reset", () => {
      feedN(engine, 20, 0.5);
      engine.reset();
      expect(engine.snapshot().midPriceSma20).toBe(0);
    });

    test("clears the EMA state so it returns 0 after reset", () => {
      feedN(engine, 5, 0.5);
      engine.reset();
      expect(engine.snapshot().midPriceEma20).toBe(0);
    });

    test("clears RSI state so it returns 0 after reset", () => {
      feedN(engine, 20, 0.5);
      engine.reset();
      expect(engine.snapshot().rsi14).toBe(0);
    });

    test("clears realized vol so it returns 0 after reset", () => {
      for (let i = 0; i < 50; i++) {
        engine.update(makeBar(i % 2 === 0 ? 0.5 : 0.55));
      }
      engine.reset();
      expect(engine.snapshot().realizedVol).toBe(0);
    });

    test("clears spread SMA so it returns 0 after reset", () => {
      feedN(engine, 10, 0.5, 0.03);
      engine.reset();
      expect(engine.snapshot().spreadSma10).toBe(0);
    });

    test("after reset(), re-feeding 20 bars restores SMA correctly", () => {
      feedN(engine, 20, 0.5);
      engine.reset();
      feedN(engine, 20, 0.8);
      expect(engine.snapshot().midPriceSma20).toBeCloseTo(0.8, 10);
    });

    test("after reset(), EMA re-initialises to the first new bar", () => {
      feedN(engine, 10, 0.5);
      engine.reset();
      engine.update(makeBar(0.9));
      expect(engine.snapshot().midPriceEma20).toBeCloseTo(0.9, 10);
    });
  });
});
