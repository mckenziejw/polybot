import { describe, test, expect, beforeEach } from "bun:test";
import { OrderBook, OrderBookManager } from "../src/data/orderbook.ts";
import type { OrderBookState, PriceChangeEntry } from "../src/types.ts";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeSnapshot(overrides: Partial<OrderBookState> = {}): OrderBookState {
  return {
    assetId: "asset-1",
    timestamp: 1_000,
    hash: "hash-abc",
    bids: [
      { price: 0.55, size: 100 },
      { price: 0.50, size: 200 },
      { price: 0.45, size: 150 },
    ],
    asks: [
      { price: 0.60, size: 80 },
      { price: 0.65, size: 120 },
      { price: 0.70, size: 90 },
    ],
    ...overrides,
  };
}

function makeChange(overrides: Partial<PriceChangeEntry>): PriceChangeEntry {
  return {
    assetId: "asset-1",
    price: 0.55,
    size: 50,
    side: "BUY",
    hash: "hash-xyz",
    bestBid: 0.55,
    bestAsk: 0.60,
    ...overrides,
  };
}

// ---------------------------------------------------------------------------
// OrderBook
// ---------------------------------------------------------------------------

describe("OrderBook", () => {
  let book: OrderBook;

  beforeEach(() => {
    book = new OrderBook();
  });

  // ── applySnapshot ─────────────────────────────────────────────────────────

  describe("applySnapshot", () => {
    test("replaces all bids and asks", () => {
      book.applySnapshot(makeSnapshot());
      expect(book.bids).toHaveLength(3);
      expect(book.asks).toHaveLength(3);
    });

    test("sets timestamp and hash", () => {
      book.applySnapshot(makeSnapshot({ timestamp: 9_999, hash: "testhash" }));
      expect(book.timestamp).toBe(9_999);
      // hash is private; we verify indirectly via snapshot()
      expect(book.snapshot().hash).toBe("testhash");
    });

    test("sets assetId", () => {
      book.applySnapshot(makeSnapshot({ assetId: "my-asset" }));
      expect(book.assetId).toBe("my-asset");
    });

    test("bids are sorted descending after snapshot", () => {
      // Supply bids in wrong order
      book.applySnapshot(
        makeSnapshot({
          bids: [
            { price: 0.45, size: 10 },
            { price: 0.55, size: 20 },
            { price: 0.50, size: 30 },
          ],
        }),
      );
      const prices = book.bids.map((l) => l.price);
      expect(prices).toEqual([0.55, 0.50, 0.45]);
    });

    test("asks are sorted ascending after snapshot", () => {
      book.applySnapshot(
        makeSnapshot({
          asks: [
            { price: 0.70, size: 10 },
            { price: 0.60, size: 20 },
            { price: 0.65, size: 30 },
          ],
        }),
      );
      const prices = book.asks.map((l) => l.price);
      expect(prices).toEqual([0.60, 0.65, 0.70]);
    });

    test("replacing an existing book clears old data", () => {
      book.applySnapshot(makeSnapshot());
      book.applySnapshot(
        makeSnapshot({
          bids: [{ price: 0.99, size: 1 }],
          asks: [{ price: 1.00, size: 1 }],
        }),
      );
      expect(book.bids).toHaveLength(1);
      expect(book.asks).toHaveLength(1);
      expect(book.bids[0]!.price).toBe(0.99);
    });
  });

  // ── applyPriceChange ──────────────────────────────────────────────────────

  describe("applyPriceChange", () => {
    beforeEach(() => {
      book.applySnapshot(makeSnapshot());
    });

    test("inserts a new bid level in sorted position", () => {
      // New bid between 0.55 and 0.50
      book.applyPriceChange(makeChange({ price: 0.52, size: 75, side: "BUY" }), 2_000);
      const prices = book.bids.map((l) => l.price);
      expect(prices).toEqual([0.55, 0.52, 0.50, 0.45]);
    });

    test("inserts a new ask level in sorted position", () => {
      // New ask between 0.60 and 0.65
      book.applyPriceChange(makeChange({ price: 0.62, size: 50, side: "SELL" }), 2_000);
      const prices = book.asks.map((l) => l.price);
      expect(prices).toEqual([0.60, 0.62, 0.65, 0.70]);
    });

    test("updates an existing bid level", () => {
      book.applyPriceChange(makeChange({ price: 0.55, size: 999, side: "BUY" }), 2_000);
      const level = book.bids.find((l) => l.price === 0.55);
      expect(level?.size).toBe(999);
    });

    test("updates an existing ask level", () => {
      book.applyPriceChange(makeChange({ price: 0.60, size: 777, side: "SELL" }), 2_000);
      const level = book.asks.find((l) => l.price === 0.60);
      expect(level?.size).toBe(777);
    });

    test("removes a bid level when size === 0", () => {
      book.applyPriceChange(makeChange({ price: 0.50, size: 0, side: "BUY" }), 2_000);
      const prices = book.bids.map((l) => l.price);
      expect(prices).not.toContain(0.50);
      expect(book.bids).toHaveLength(2);
    });

    test("removes an ask level when size === 0", () => {
      book.applyPriceChange(makeChange({ price: 0.65, size: 0, side: "SELL" }), 2_000);
      const prices = book.asks.map((l) => l.price);
      expect(prices).not.toContain(0.65);
      expect(book.asks).toHaveLength(2);
    });

    test("removing a non-existent level is a no-op", () => {
      const bidsBefore = book.bids.length;
      book.applyPriceChange(makeChange({ price: 0.99, size: 0, side: "BUY" }), 2_000);
      expect(book.bids).toHaveLength(bidsBefore);
    });

    test("updates timestamp and hash", () => {
      book.applyPriceChange(
        makeChange({ hash: "new-hash" }),
        5_555,
      );
      expect(book.timestamp).toBe(5_555);
      expect(book.snapshot().hash).toBe("new-hash");
    });

    test("bids remain sorted descending after multiple changes", () => {
      book.applyPriceChange(makeChange({ price: 0.57, size: 40, side: "BUY" }), 2_000);
      book.applyPriceChange(makeChange({ price: 0.43, size: 20, side: "BUY" }), 2_001);
      book.applyPriceChange(makeChange({ price: 0.51, size: 60, side: "BUY" }), 2_002);
      const prices = book.bids.map((l) => l.price);
      expect(prices).toEqual([...prices].sort((a, b) => b - a));
    });

    test("asks remain sorted ascending after multiple changes", () => {
      book.applyPriceChange(makeChange({ price: 0.63, size: 40, side: "SELL" }), 2_000);
      book.applyPriceChange(makeChange({ price: 0.58, size: 10, side: "SELL" }), 2_001);
      book.applyPriceChange(makeChange({ price: 0.75, size: 30, side: "SELL" }), 2_002);
      const prices = book.asks.map((l) => l.price);
      expect(prices).toEqual([...prices].sort((a, b) => a - b));
    });
  });

  // ── Computed getters ──────────────────────────────────────────────────────

  describe("computed getters", () => {
    beforeEach(() => {
      book.applySnapshot(makeSnapshot());
      // After snapshot: bestBid=0.55, bestAsk=0.60
    });

    test("bestBid returns highest bid price", () => {
      expect(book.bestBid).toBe(0.55);
    });

    test("bestAsk returns lowest ask price", () => {
      expect(book.bestAsk).toBe(0.60);
    });

    test("mid returns average of bestBid and bestAsk", () => {
      expect(book.mid).toBeCloseTo(0.575, 10);
    });

    test("spread returns bestAsk minus bestBid", () => {
      expect(book.spread).toBeCloseTo(0.05, 10);
    });

    test("bestBid returns 0 when no bids", () => {
      book.applySnapshot(makeSnapshot({ bids: [] }));
      expect(book.bestBid).toBe(0);
    });

    test("bestAsk returns Infinity when no asks", () => {
      book.applySnapshot(makeSnapshot({ asks: [] }));
      expect(book.bestAsk).toBe(Infinity);
    });

    test("mid returns 0 when bids empty", () => {
      book.applySnapshot(makeSnapshot({ bids: [] }));
      expect(book.mid).toBe(0);
    });

    test("mid returns 0 when asks empty", () => {
      book.applySnapshot(makeSnapshot({ asks: [] }));
      expect(book.mid).toBe(0);
    });

    test("spread returns 0 when bids empty", () => {
      book.applySnapshot(makeSnapshot({ bids: [] }));
      expect(book.spread).toBe(0);
    });

    test("imbalance is bid-heavy when bids dominate", () => {
      book.applySnapshot(
        makeSnapshot({
          bids: [{ price: 0.55, size: 1000 }],
          asks: [{ price: 0.60, size: 100 }],
        }),
      );
      expect(book.imbalance).toBeGreaterThan(0);
    });

    test("imbalance is ask-heavy when asks dominate", () => {
      book.applySnapshot(
        makeSnapshot({
          bids: [{ price: 0.55, size: 100 }],
          asks: [{ price: 0.60, size: 1000 }],
        }),
      );
      expect(book.imbalance).toBeLessThan(0);
    });

    test("imbalance is 0 when book is empty", () => {
      book.applySnapshot(makeSnapshot({ bids: [], asks: [] }));
      expect(book.imbalance).toBe(0);
    });

    test("imbalance is symmetric (equal sizes → 0)", () => {
      book.applySnapshot(
        makeSnapshot({
          bids: [{ price: 0.55, size: 500 }],
          asks: [{ price: 0.60, size: 500 }],
        }),
      );
      expect(book.imbalance).toBe(0);
    });
  });

  // ── topLevels ─────────────────────────────────────────────────────────────

  describe("topLevels(5)", () => {
    test("returns exactly 5 entries per side when book has ≥ 5 levels", () => {
      book.applySnapshot(
        makeSnapshot({
          bids: [
            { price: 0.55, size: 10 },
            { price: 0.54, size: 20 },
            { price: 0.53, size: 30 },
            { price: 0.52, size: 40 },
            { price: 0.51, size: 50 },
            { price: 0.50, size: 60 },
          ],
          asks: [
            { price: 0.60, size: 10 },
            { price: 0.61, size: 20 },
            { price: 0.62, size: 30 },
            { price: 0.63, size: 40 },
            { price: 0.64, size: 50 },
            { price: 0.65, size: 60 },
          ],
        }),
      );
      const top = book.topLevels(5);
      expect(top.bidPrice).toHaveLength(5);
      expect(top.bidSize).toHaveLength(5);
      expect(top.askPrice).toHaveLength(5);
      expect(top.askSize).toHaveLength(5);
    });

    test("pads with zeros when fewer than 5 levels exist", () => {
      book.applySnapshot(
        makeSnapshot({
          bids: [{ price: 0.55, size: 100 }],
          asks: [{ price: 0.60, size: 80 }],
        }),
      );
      const top = book.topLevels(5);
      expect(top.bidPrice).toHaveLength(5);
      expect(top.bidPrice[0]).toBe(0.55);
      expect(top.bidPrice[1]).toBe(0);
      expect(top.bidPrice[4]).toBe(0);
      expect(top.bidSize[1]).toBe(0);
      expect(top.askPrice[0]).toBe(0.60);
      expect(top.askPrice[1]).toBe(0);
    });

    test("returns all zeros when book is empty", () => {
      book.applySnapshot(makeSnapshot({ bids: [], asks: [] }));
      const top = book.topLevels(5);
      expect(top.bidPrice.every((p) => p === 0)).toBeTrue();
      expect(top.bidSize.every((s) => s === 0)).toBeTrue();
      expect(top.askPrice.every((p) => p === 0)).toBeTrue();
      expect(top.askSize.every((s) => s === 0)).toBeTrue();
    });

    test("top levels are in correct sorted order", () => {
      book.applySnapshot(makeSnapshot());
      const top = book.topLevels(3);
      // Bids descending
      expect(top.bidPrice[0]).toBeGreaterThan(top.bidPrice[1]!);
      // Asks ascending
      expect(top.askPrice[0]).toBeLessThan(top.askPrice[1]!);
    });
  });

  // ── clear ─────────────────────────────────────────────────────────────────

  describe("clear", () => {
    test("resets all state to defaults", () => {
      book.applySnapshot(makeSnapshot());
      book.clear();
      expect(book.bids).toHaveLength(0);
      expect(book.asks).toHaveLength(0);
      expect(book.timestamp).toBe(0);
      expect(book.assetId).toBe("");
      expect(book.snapshot().hash).toBe("");
    });
  });
});

// ---------------------------------------------------------------------------
// OrderBookManager
// ---------------------------------------------------------------------------

describe("OrderBookManager", () => {
  let manager: OrderBookManager;

  beforeEach(() => {
    manager = new OrderBookManager();
  });

  test("get() creates a new OrderBook for an unknown assetId", () => {
    const book = manager.get("new-asset");
    expect(book).toBeInstanceOf(OrderBook);
  });

  test("get() returns the same instance for the same assetId", () => {
    const a = manager.get("asset-x");
    const b = manager.get("asset-x");
    expect(a).toBe(b);
  });

  test("get() returns different instances for different assetIds", () => {
    const a = manager.get("asset-x");
    const b = manager.get("asset-y");
    expect(a).not.toBe(b);
  });

  test("allSnapshots() returns snapshots for all tracked books", () => {
    manager.get("asset-x").applySnapshot(makeSnapshot({ assetId: "asset-x" }));
    manager.get("asset-y").applySnapshot(makeSnapshot({ assetId: "asset-y" }));
    const snaps = manager.allSnapshots();
    expect(Object.keys(snaps)).toHaveLength(2);
    expect(snaps["asset-x"]).toBeDefined();
    expect(snaps["asset-y"]).toBeDefined();
  });

  test("allSnapshots() preserves book data", () => {
    manager.get("asset-x").applySnapshot(makeSnapshot({ assetId: "asset-x", hash: "snap-hash" }));
    const snaps = manager.allSnapshots();
    expect(snaps["asset-x"]!.hash).toBe("snap-hash");
  });

  test("clear() removes all books", () => {
    manager.get("asset-x");
    manager.get("asset-y");
    manager.clear();
    const snaps = manager.allSnapshots();
    expect(Object.keys(snaps)).toHaveLength(0);
  });

  test("after clear(), get() creates a fresh book", () => {
    const before = manager.get("asset-x");
    before.applySnapshot(makeSnapshot({ assetId: "asset-x" }));
    manager.clear();
    const after = manager.get("asset-x");
    expect(after.bids).toHaveLength(0);
  });
});
