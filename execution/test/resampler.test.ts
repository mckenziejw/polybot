import { describe, test, expect, beforeEach } from "bun:test";
import { Resampler } from "../src/data/resampler.ts";
import { OrderBook } from "../src/data/orderbook.ts";
import type { OrderBookState } from "../src/types.ts";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeBook(
  bids: { price: number; size: number }[],
  asks: { price: number; size: number }[],
  assetId = "asset-1",
): OrderBook {
  const book = new OrderBook();
  const state: OrderBookState = {
    assetId,
    timestamp: Date.now(),
    hash: "h",
    bids,
    asks,
  };
  book.applySnapshot(state);
  return book;
}

/** Standard book: bid=0.55, ask=0.60 → mid=0.575, spread=0.05 */
function standardBook(assetId = "asset-1"): OrderBook {
  return makeBook(
    [{ price: 0.55, size: 100 }],
    [{ price: 0.60, size: 80 }],
    assetId,
  );
}

// ---------------------------------------------------------------------------
// Resampler
// ---------------------------------------------------------------------------

describe("Resampler", () => {
  let resampler: Resampler;

  beforeEach(() => {
    // Use a 100 ms interval for all tests
    resampler = new Resampler(100);
  });

  // ── tick + emit ───────────────────────────────────────────────────────────

  describe("tick() + emit()", () => {
    test("emit returns one bar per asset after a tick", () => {
      resampler.tick("asset-1", standardBook());
      const bars = resampler.emit();
      expect(bars).toHaveLength(1);
    });

    test("bar contains correct midPrice derived from book", () => {
      resampler.tick("asset-1", standardBook());
      const [bar] = resampler.emit();
      expect(bar!.midPrice).toBeCloseTo(0.575, 10);
    });

    test("bar contains correct spread derived from book", () => {
      resampler.tick("asset-1", standardBook());
      const [bar] = resampler.emit();
      expect(bar!.spread).toBeCloseTo(0.05, 10);
    });

    test("bar assetId matches the asset fed via tick()", () => {
      resampler.tick("my-asset", standardBook("my-asset"));
      const [bar] = resampler.emit();
      expect(bar!.assetId).toBe("my-asset");
    });

    test("bar has stalenessMs=0 immediately after a tick", () => {
      resampler.tick("asset-1", standardBook());
      const [bar] = resampler.emit();
      expect(bar!.stalenessMs).toBe(0);
    });

    test("bar bookImbalance reflects book state (bid-heavy → positive)", () => {
      const book = makeBook(
        [{ price: 0.55, size: 1000 }],
        [{ price: 0.60, size: 100 }],
      );
      resampler.tick("asset-1", book);
      const [bar] = resampler.emit();
      expect(bar!.bookImbalance).toBeGreaterThan(0);
    });

    test("bar bidPrice / askPrice arrays have 5 entries", () => {
      resampler.tick("asset-1", standardBook());
      const [bar] = resampler.emit();
      expect(bar!.bidPrice).toHaveLength(5);
      expect(bar!.askPrice).toHaveLength(5);
    });

    test("missing levels are padded with zeros in bidPrice", () => {
      resampler.tick("asset-1", standardBook()); // only 1 bid level
      const [bar] = resampler.emit();
      // Level 0 should be 0.55; levels 1-4 should be 0
      expect(bar!.bidPrice[0]).toBe(0.55);
      expect(bar!.bidPrice[1]).toBe(0);
      expect(bar!.bidPrice[4]).toBe(0);
    });

    test("emit returns nothing when no ticks have been fed", () => {
      const bars = resampler.emit();
      expect(bars).toHaveLength(0);
    });

    test("emit resets hasNewTick so next emit without a tick forward-fills", () => {
      resampler.tick("asset-1", standardBook());
      resampler.emit(); // consumes tick
      const bars = resampler.emit(); // should forward-fill
      expect(bars).toHaveLength(1);
      expect(bars[0]!.stalenessMs).toBeGreaterThan(0);
    });
  });

  // ── Forward-fill / staleness ───────────────────────────────────────────────

  describe("forward-fill and staleness", () => {
    test("staleness increments by intervalMs on each missed tick", () => {
      resampler.tick("asset-1", standardBook());
      resampler.emit(); // stalenessMs reset to 0

      const [bar1] = resampler.emit(); // 1st forward-fill
      expect(bar1!.stalenessMs).toBe(100);

      const [bar2] = resampler.emit(); // 2nd forward-fill
      expect(bar2!.stalenessMs).toBe(200);
    });

    test("staleness resets to 0 after a new real tick", () => {
      resampler.tick("asset-1", standardBook());
      resampler.emit(); // consume tick, staleness=0
      resampler.emit(); // forward-fill, staleness=100

      // New real tick arrives
      resampler.tick("asset-1", standardBook());
      const [bar] = resampler.emit();
      expect(bar!.stalenessMs).toBe(0);
    });

    test("staleness is capped at 5000 ms", () => {
      resampler.tick("asset-1", standardBook());
      resampler.emit(); // consume tick

      // Drive staleness well past the cap (5000 / 100 = 50 intervals → need 51 more)
      for (let i = 0; i < 60; i++) {
        resampler.emit();
      }

      const [bar] = resampler.emit();
      expect(bar!.stalenessMs).toBe(5_000);
    });

    test("forward-fill preserves midPrice from last known tick", () => {
      const book = makeBook(
        [{ price: 0.70, size: 50 }],
        [{ price: 0.80, size: 50 }],
      );
      resampler.tick("asset-1", book);
      resampler.emit(); // consume tick (mid=0.75)

      const [bar] = resampler.emit(); // forward-fill
      expect(bar!.midPrice).toBeCloseTo(0.75, 10);
    });
  });

  // ── reset() ───────────────────────────────────────────────────────────────

  describe("reset()", () => {
    test("clears all tracked assets so emit returns nothing", () => {
      resampler.tick("asset-1", standardBook());
      resampler.emit();
      resampler.reset();
      const bars = resampler.emit();
      expect(bars).toHaveLength(0);
    });

    test("after reset(), new ticks are tracked fresh", () => {
      resampler.tick("asset-1", standardBook());
      resampler.emit();
      resampler.reset();

      resampler.tick("asset-1", standardBook());
      const bars = resampler.emit();
      expect(bars).toHaveLength(1);
      expect(bars[0]!.stalenessMs).toBe(0);
    });

    test("staleness accumulated before reset does not carry over", () => {
      resampler.tick("asset-1", standardBook());
      resampler.emit();
      resampler.emit(); // staleness=100
      resampler.emit(); // staleness=200
      resampler.reset();

      resampler.tick("asset-1", standardBook());
      resampler.emit(); // consumes tick, staleness=0
      const [bar] = resampler.emit(); // first forward-fill after reset
      expect(bar!.stalenessMs).toBe(100); // not 300
    });
  });

  // ── Multiple assets ────────────────────────────────────────────────────────

  describe("multiple assets tracked independently", () => {
    test("emit returns a bar for each tracked asset", () => {
      resampler.tick("asset-a", standardBook("asset-a"));
      resampler.tick("asset-b", standardBook("asset-b"));
      const bars = resampler.emit();
      expect(bars).toHaveLength(2);
      const ids = bars.map((b) => b.assetId).sort();
      expect(ids).toEqual(["asset-a", "asset-b"]);
    });

    test("staleness accumulates independently per asset", () => {
      resampler.tick("asset-a", standardBook("asset-a"));
      resampler.tick("asset-b", standardBook("asset-b"));
      resampler.emit(); // both staleness=0

      // Only feed asset-a next interval
      resampler.tick("asset-a", standardBook("asset-a"));
      const bars = resampler.emit();

      const barA = bars.find((b) => b.assetId === "asset-a")!;
      const barB = bars.find((b) => b.assetId === "asset-b")!;

      expect(barA.stalenessMs).toBe(0);   // fresh tick
      expect(barB.stalenessMs).toBe(100); // forward-filled
    });

    test("midPrice is tracked per asset independently", () => {
      const bookA = makeBook(
        [{ price: 0.40, size: 10 }],
        [{ price: 0.50, size: 10 }],
        "asset-a",
      );
      const bookB = makeBook(
        [{ price: 0.70, size: 10 }],
        [{ price: 0.80, size: 10 }],
        "asset-b",
      );

      resampler.tick("asset-a", bookA);
      resampler.tick("asset-b", bookB);
      const bars = resampler.emit();

      const barA = bars.find((b) => b.assetId === "asset-a")!;
      const barB = bars.find((b) => b.assetId === "asset-b")!;

      expect(barA.midPrice).toBeCloseTo(0.45, 10);
      expect(barB.midPrice).toBeCloseTo(0.75, 10);
    });
  });
});
