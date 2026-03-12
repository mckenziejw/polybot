import { describe, test, expect, beforeEach } from "bun:test";
import { PositionStore } from "../src/positions/position-store.ts";
import type { OpenOrder, UserOrderEvent, UserTradeEvent } from "../src/types.ts";

// Helpers for building test events

function makeOpenOrder(overrides: Partial<OpenOrder> = {}): OpenOrder {
  return {
    orderId: "order-1",
    assetId: "asset-abc",
    side: "BUY",
    price: 0.5,
    originalSize: 100,
    remainingSize: 100,
    placedAt: Date.now(),
    ...overrides,
  };
}

function makeTradeEvent(overrides: Partial<UserTradeEvent> = {}): UserTradeEvent {
  return {
    eventType: "trade",
    id: "trade-1",
    takerOrderId: "order-1",
    market: "market-1",
    assetId: "asset-abc",
    side: "BUY",
    size: 50,
    price: 0.5,
    feeRateBps: 0,
    status: "MATCHED",
    matchTime: new Date().toISOString(),
    lastUpdate: new Date().toISOString(),
    outcome: "YES",
    owner: "0xowner",
    tradeOwner: "0xowner",
    makerAddress: "0xmaker",
    transactionHash: "0xtx",
    bucketIndex: 0,
    makerOrders: [],
    traderSide: "TAKER",
    timestamp: Date.now(),
    ...overrides,
  };
}

function makeOrderEvent(overrides: Partial<UserOrderEvent> = {}): UserOrderEvent {
  return {
    eventType: "order",
    id: "order-1",
    owner: "0xowner",
    market: "market-1",
    assetId: "asset-abc",
    side: "BUY",
    originalSize: 100,
    sizeMatched: 0,
    price: 0.5,
    associateTrades: null,
    outcome: "YES",
    orderEventType: "PLACEMENT",
    createdAt: new Date().toISOString(),
    expiration: "",
    orderType: "GTC",
    status: "LIVE",
    makerAddress: "0xmaker",
    timestamp: Date.now(),
    ...overrides,
  };
}

describe("PositionStore", () => {
  let store: PositionStore;

  beforeEach(() => {
    store = new PositionStore();
  });

  // ── addOrder ──────────────────────────────────────────────────────────────

  describe("addOrder", () => {
    test("adds order to openOrders", () => {
      const order = makeOpenOrder({ orderId: "ord-1" });
      store.addOrder(order);
      const orders = store.getOpenOrders();
      expect(orders).toHaveLength(1);
      expect(orders[0]!.orderId).toBe("ord-1");
    });

    test("stores a copy so external mutation does not affect store", () => {
      const order = makeOpenOrder({ orderId: "ord-2", remainingSize: 100 });
      store.addOrder(order);
      order.remainingSize = 0; // mutate original
      const stored = store.getOpenOrders()[0]!;
      expect(stored.remainingSize).toBe(100);
    });

    test("adding multiple orders tracks all of them", () => {
      store.addOrder(makeOpenOrder({ orderId: "ord-a" }));
      store.addOrder(makeOpenOrder({ orderId: "ord-b" }));
      expect(store.getOpenOrders()).toHaveLength(2);
    });
  });

  // ── applyFill ─────────────────────────────────────────────────────────────

  describe("applyFill", () => {
    test("creates a new position on first fill", () => {
      store.addOrder(makeOpenOrder({ orderId: "order-1", originalSize: 50, remainingSize: 50 }));
      store.applyFill(makeTradeEvent({ side: "BUY", size: 50, price: 0.5 }));
      const positions = store.getPositions();
      expect(positions).toHaveLength(1);
      expect(positions[0]!.side).toBe("BUY");
      expect(positions[0]!.size).toBe(50);
      expect(positions[0]!.avgEntryPrice).toBe(0.5);
      expect(positions[0]!.realizedPnl).toBe(0);
    });

    test("updates avgEntryPrice on same-side fill (weighted average)", () => {
      store.addOrder(makeOpenOrder({ orderId: "order-1", originalSize: 100, remainingSize: 100 }));
      // First fill: 50 @ 0.4
      store.applyFill(makeTradeEvent({ id: "t1", side: "BUY", size: 50, price: 0.4 }));
      // Second fill: 50 @ 0.6 — avg should be 0.5
      store.applyFill(makeTradeEvent({ id: "t2", side: "BUY", size: 50, price: 0.6 }));
      const pos = store.getPositions()[0]!;
      expect(pos.size).toBe(100);
      expect(pos.avgEntryPrice).toBeCloseTo(0.5, 10);
    });

    test("calculates realizedPnl on opposing fill (BUY then SELL)", () => {
      store.addOrder(makeOpenOrder({ orderId: "order-1", originalSize: 100, remainingSize: 100 }));
      store.addOrder(makeOpenOrder({ orderId: "order-2", originalSize: 60, remainingSize: 60 }));
      // Enter long 100 @ 0.4
      store.applyFill(makeTradeEvent({ side: "BUY", size: 100, price: 0.4 }));
      // Close 60 @ 0.6 → pnl = (0.6 - 0.4) * 60 = 12
      store.applyFill(
        makeTradeEvent({ takerOrderId: "order-2", side: "SELL", size: 60, price: 0.6, id: "t2" })
      );
      const pos = store.getPositions()[0]!;
      expect(pos.realizedPnl).toBeCloseTo(12, 10);
      expect(pos.size).toBe(40);
      expect(pos.side).toBe("BUY");
    });

    test("calculates realizedPnl on opposing fill (SELL then BUY)", () => {
      store.addOrder(makeOpenOrder({ orderId: "order-1", originalSize: 100, remainingSize: 100 }));
      store.addOrder(makeOpenOrder({ orderId: "order-2", originalSize: 40, remainingSize: 40 }));
      // Enter short 100 @ 0.6
      store.applyFill(makeTradeEvent({ side: "SELL", size: 100, price: 0.6 }));
      // Close 40 @ 0.4 → pnl = (0.6 - 0.4) * 40 = 8
      store.applyFill(
        makeTradeEvent({ takerOrderId: "order-2", side: "BUY", size: 40, price: 0.4, id: "t2" })
      );
      const pos = store.getPositions()[0]!;
      expect(pos.realizedPnl).toBeCloseTo(8, 10);
      expect(pos.size).toBe(60);
    });

    test("processes fill events with MINED status (trades can skip MATCHED)", () => {
      store.addOrder(makeOpenOrder({ orderId: "order-1" }));
      store.applyFill(makeTradeEvent({ status: "MINED", size: 25, price: 0.5 }));
      const positions = store.getPositions();
      expect(positions).toHaveLength(1);
      expect(positions[0]!.size).toBe(25);
    });

    test("ignores fill events with RETRYING/FAILED status", () => {
      store.addOrder(makeOpenOrder({ orderId: "order-1" }));
      store.applyFill(makeTradeEvent({ status: "RETRYING" }));
      expect(store.getPositions()).toHaveLength(0);
      store.applyFill(makeTradeEvent({ id: "t-fail", status: "FAILED" }));
      expect(store.getPositions()).toHaveLength(0);
    });

    test("processes fill with CONFIRMED status", () => {
      store.addOrder(makeOpenOrder({ orderId: "order-1" }));
      store.applyFill(makeTradeEvent({ id: "t-confirmed", status: "CONFIRMED", size: 30, price: 0.5 }));
      const positions = store.getPositions();
      expect(positions).toHaveLength(1);
      expect(positions[0]!.size).toBe(30);
    });

    test("deduplicates same trade ID across status updates", () => {
      store.addOrder(makeOpenOrder({ orderId: "order-1" }));
      store.applyFill(makeTradeEvent({ id: "t-dup", status: "MATCHED", size: 50, price: 0.5 }));
      store.applyFill(makeTradeEvent({ id: "t-dup", status: "CONFIRMED", size: 50, price: 0.5 }));
      const pos = store.getPositions()[0]!;
      // Should only count once despite two events
      expect(pos.size).toBe(50);
    });

    test("ignores trade events for orders we did not place", () => {
      // Do NOT call addOrder — simulate a trade from another user
      store.applyFill(makeTradeEvent({ side: "BUY", size: 100, price: 0.5 }));
      expect(store.getPositions()).toHaveLength(0);
    });
  });

  // ── Partial fill ──────────────────────────────────────────────────────────

  describe("partial fill", () => {
    test("remainingSize decreases correctly after partial taker fill", () => {
      store.addOrder(makeOpenOrder({ orderId: "order-1", originalSize: 100, remainingSize: 100 }));
      store.applyFill(
        makeTradeEvent({ takerOrderId: "order-1", size: 40 })
      );
      const orders = store.getOpenOrders();
      expect(orders).toHaveLength(1);
      expect(orders[0]!.remainingSize).toBe(60);
    });

    test("order is removed when fully filled via taker", () => {
      store.addOrder(makeOpenOrder({ orderId: "order-1", originalSize: 100, remainingSize: 100 }));
      store.applyFill(
        makeTradeEvent({ takerOrderId: "order-1", size: 100 })
      );
      expect(store.getOpenOrders()).toHaveLength(0);
    });

    test("remainingSize decreases correctly for maker order", () => {
      store.addOrder(makeOpenOrder({ orderId: "maker-ord", originalSize: 200, remainingSize: 200 }));
      store.applyFill(
        makeTradeEvent({
          id: "t-maker-partial",
          traderSide: "MAKER",
          takerOrderId: "other-order",
          makerOrders: [
            {
              orderId: "maker-ord",
              owner: "0xowner",
              makerAddress: "0xmaker",
              matchedAmount: 80,
              price: 0.5,
              feeRateBps: 0,
              assetId: "asset-abc",
              outcome: "YES",
              side: "BUY",
            },
          ],
        })
      );
      const orders = store.getOpenOrders();
      expect(orders).toHaveLength(1);
      expect(orders[0]!.remainingSize).toBe(120);
    });

    test("maker order removed when fully matched", () => {
      store.addOrder(makeOpenOrder({ orderId: "maker-ord", originalSize: 100, remainingSize: 100 }));
      store.applyFill(
        makeTradeEvent({
          id: "t-maker-full",
          traderSide: "MAKER",
          takerOrderId: "other-order",
          makerOrders: [
            {
              orderId: "maker-ord",
              owner: "0xowner",
              makerAddress: "0xmaker",
              matchedAmount: 100,
              price: 0.5,
              feeRateBps: 0,
              assetId: "asset-abc",
              outcome: "YES",
              side: "BUY",
            },
          ],
        })
      );
      expect(store.getOpenOrders()).toHaveLength(0);
    });

    test("maker fill uses maker price/side, not taker top-level fields", () => {
      // We placed a BUY limit order at 0.38
      store.addOrder(makeOpenOrder({ orderId: "our-maker", assetId: "down-token", side: "BUY", price: 0.38, originalSize: 5, remainingSize: 5 }));
      // Trade event comes from taker's perspective: taker is SELLING at 0.43
      store.applyFill(
        makeTradeEvent({
          id: "t-maker-price",
          traderSide: "MAKER",
          takerOrderId: "taker-ord",
          assetId: "down-token",
          side: "SELL",       // taker's side
          size: 5,            // taker's size
          price: 0.43,        // taker's price (NOT our fill price)
          makerOrders: [
            {
              orderId: "our-maker",
              owner: "0xowner",
              makerAddress: "0xmaker",
              matchedAmount: 5,
              price: 0.38,      // OUR fill price
              feeRateBps: 0,
              assetId: "down-token",
              outcome: "NO",
              side: "BUY",      // OUR side
            },
          ],
        })
      );
      const pos = store.getPositions()[0]!;
      expect(pos.assetId).toBe("down-token");
      expect(pos.side).toBe("BUY");
      expect(pos.size).toBe(5);
      // Must use maker price (0.38), not taker price (0.43)
      expect(pos.avgEntryPrice).toBe(0.38);
    });

    test("taker fill uses top-level fields when we are the taker", () => {
      // We placed an order that became the taker
      store.addOrder(makeOpenOrder({ orderId: "our-taker", assetId: "up-token", side: "BUY", price: 0.58, originalSize: 5, remainingSize: 5 }));
      store.applyFill(
        makeTradeEvent({
          id: "t-taker-fields",
          traderSide: "TAKER",
          takerOrderId: "our-taker",
          assetId: "up-token",
          side: "BUY",
          size: 5,
          price: 0.58,
          makerOrders: [
            {
              orderId: "other-maker",
              owner: "0xother",
              makerAddress: "0xothermaker",
              matchedAmount: 5,
              price: 0.58,
              feeRateBps: 0,
              assetId: "up-token",
              outcome: "YES",
              side: "SELL",
            },
          ],
        })
      );
      const pos = store.getPositions()[0]!;
      expect(pos.side).toBe("BUY");
      expect(pos.size).toBe(5);
      expect(pos.avgEntryPrice).toBe(0.58);
    });

    test("does not double-count when we are both taker and maker (edge case)", () => {
      // Unlikely but defensive: our taker order matches our own maker order
      // In practice, the exchange sends two separate events: one as TAKER, one as MAKER
      store.addOrder(makeOpenOrder({ orderId: "our-taker", assetId: "asset-abc", side: "BUY", price: 0.5, originalSize: 10, remainingSize: 10 }));
      store.addOrder(makeOpenOrder({ orderId: "our-maker", assetId: "asset-abc", side: "SELL", price: 0.5, originalSize: 10, remainingSize: 10 }));
      // Event 1: we are the taker (BUY 10)
      store.applyFill(
        makeTradeEvent({
          id: "self-trade-taker",
          traderSide: "TAKER",
          takerOrderId: "our-taker",
          assetId: "asset-abc",
          side: "BUY",
          size: 10,
          price: 0.5,
          makerOrders: [
            {
              orderId: "our-maker",
              owner: "0xowner",
              makerAddress: "0xmaker",
              matchedAmount: 10,
              price: 0.5,
              feeRateBps: 0,
              assetId: "asset-abc",
              outcome: "YES",
              side: "SELL",
            },
          ],
        })
      );
      // Event 2: we are the maker (SELL 10)
      store.applyFill(
        makeTradeEvent({
          id: "self-trade-maker",
          traderSide: "MAKER",
          takerOrderId: "our-taker",
          assetId: "asset-abc",
          side: "BUY",
          size: 10,
          price: 0.5,
          makerOrders: [
            {
              orderId: "our-maker",
              owner: "0xowner",
              makerAddress: "0xmaker",
              matchedAmount: 10,
              price: 0.5,
              feeRateBps: 0,
              assetId: "asset-abc",
              outcome: "YES",
              side: "SELL",
            },
          ],
        })
      );
      // Taker BUY 10 + Maker SELL 10 = net zero position
      expect(store.getPositions()).toHaveLength(0);
      expect(store.getOpenOrders()).toHaveLength(0);
    });

    test("multiple maker fills in a single trade event", () => {
      store.addOrder(makeOpenOrder({ orderId: "maker-a", assetId: "up-token", side: "BUY", price: 0.50, originalSize: 10, remainingSize: 10 }));
      store.addOrder(makeOpenOrder({ orderId: "maker-b", assetId: "up-token", side: "BUY", price: 0.48, originalSize: 10, remainingSize: 10 }));
      store.applyFill(
        makeTradeEvent({
          id: "t-multi-maker",
          traderSide: "MAKER",
          takerOrderId: "external-taker",
          assetId: "up-token",
          side: "SELL",
          size: 15,
          price: 0.50,
          makerOrders: [
            {
              orderId: "maker-a",
              owner: "0xowner",
              makerAddress: "0xmaker",
              matchedAmount: 10,
              price: 0.50,
              feeRateBps: 0,
              assetId: "up-token",
              outcome: "YES",
              side: "BUY",
            },
            {
              orderId: "maker-b",
              owner: "0xowner",
              makerAddress: "0xmaker",
              matchedAmount: 5,
              price: 0.48,
              feeRateBps: 0,
              assetId: "up-token",
              outcome: "YES",
              side: "BUY",
            },
          ],
        })
      );
      const pos = store.getPositions()[0]!;
      expect(pos.side).toBe("BUY");
      expect(pos.size).toBe(15);
      // Weighted avg: (0.50 * 10 + 0.48 * 5) / 15 ≈ 0.4933
      expect(pos.avgEntryPrice).toBeCloseTo(0.4933, 3);
      // maker-a fully filled, maker-b partially filled
      const orders = store.getOpenOrders();
      expect(orders).toHaveLength(1);
      expect(orders[0]!.orderId).toBe("maker-b");
      expect(orders[0]!.remainingSize).toBe(5);
    });
  });

  // ── Race condition: order deleted before trade event ──────────────────────

  describe("order event / trade event race condition", () => {
    test("maker fill still tracks position when order already deleted by UPDATE event", () => {
      // 1. Place order
      store.addOrder(makeOpenOrder({ orderId: "race-ord", assetId: "down-token", side: "BUY", price: 0.52, originalSize: 5, remainingSize: 5 }));
      // 2. Order UPDATE arrives and deletes the order (fully matched)
      store.applyOrderEvent(makeOrderEvent({
        id: "race-ord",
        assetId: "down-token",
        orderEventType: "UPDATE",
        originalSize: 5,
        sizeMatched: 5,
      }));
      expect(store.getOpenOrders()).toHaveLength(0); // order is gone

      // 3. Trade event arrives AFTER order deletion — this used to fall through
      //    to the dangerous fallback path. Now uses traderSide + knownOrderIds.
      store.applyFill(
        makeTradeEvent({
          id: "race-trade",
          traderSide: "MAKER",
          takerOrderId: "external-taker",
          assetId: "down-token",
          side: "SELL",       // taker's side (irrelevant for us)
          size: 100,          // taker's total size (NOT our fill)
          price: 0.60,        // taker's price (NOT our fill price)
          makerOrders: [
            {
              orderId: "race-ord",
              owner: "0xowner",
              makerAddress: "0xmaker",
              matchedAmount: 5,   // OUR actual fill size
              price: 0.52,        // OUR actual fill price
              feeRateBps: 0,
              assetId: "down-token",
              outcome: "NO",
              side: "BUY",        // OUR side
            },
          ],
        })
      );
      const pos = store.getPositions()[0]!;
      // Must use maker fill details, NOT the taker's top-level fields
      expect(pos.side).toBe("BUY");
      expect(pos.size).toBe(5);          // not 100
      expect(pos.avgEntryPrice).toBe(0.52); // not 0.60
    });

    test("partial fills from multiple takers track correctly", () => {
      // Our order gets filled by two different takers
      store.addOrder(makeOpenOrder({ orderId: "multi-fill", assetId: "down-token", side: "BUY", price: 0.52, originalSize: 5, remainingSize: 5 }));

      // First taker fills 3 of our 5
      store.applyFill(
        makeTradeEvent({
          id: "multi-t1",
          traderSide: "MAKER",
          takerOrderId: "taker-1",
          assetId: "down-token",
          side: "SELL",
          size: 10,           // taker's total order
          price: 0.55,
          makerOrders: [
            {
              orderId: "multi-fill",
              owner: "0xowner",
              makerAddress: "0xmaker",
              matchedAmount: 3,
              price: 0.52,
              feeRateBps: 0,
              assetId: "down-token",
              outcome: "NO",
              side: "BUY",
            },
          ],
        })
      );

      // Second taker fills remaining 2
      store.applyFill(
        makeTradeEvent({
          id: "multi-t2",
          traderSide: "MAKER",
          takerOrderId: "taker-2",
          assetId: "down-token",
          side: "SELL",
          size: 20,           // different taker's total order
          price: 0.53,
          makerOrders: [
            {
              orderId: "multi-fill",
              owner: "0xowner",
              makerAddress: "0xmaker",
              matchedAmount: 2,
              price: 0.52,
              feeRateBps: 0,
              assetId: "down-token",
              outcome: "NO",
              side: "BUY",
            },
          ],
        })
      );

      const pos = store.getPositions()[0]!;
      expect(pos.side).toBe("BUY");
      expect(pos.size).toBe(5);            // 3 + 2, not 10 + 20
      expect(pos.avgEntryPrice).toBe(0.52); // all at same maker price
      // Order should be fully consumed
      expect(store.getOpenOrders()).toHaveLength(0);
    });
  });

  // ── Position flip ─────────────────────────────────────────────────────────
  //
  // Note: in applyFill, closedSize = Math.min(pos.size, fillSize) so pos.size
  // can never go below 0 after subtraction.  The `pos.size < 0` branch is
  // therefore unreachable; an opposing fill larger than the position reduces
  // the position to 0 (full close) rather than flipping the side.

  describe("position flip", () => {
    test("opposing fill larger than position closes it to zero (no side flip)", () => {
      store.addOrder(makeOpenOrder({ orderId: "flip-o1", originalSize: 50, remainingSize: 50 }));
      store.addOrder(makeOpenOrder({ orderId: "flip-o2", originalSize: 80, remainingSize: 80 }));
      // Enter long 50 @ 0.4
      store.applyFill(makeTradeEvent({ id: "flip-t1", takerOrderId: "flip-o1", side: "BUY", size: 50, price: 0.4 }));
      // Sell 80: closedSize = min(50,80) = 50, pos.size becomes 0
      store.applyFill(
        makeTradeEvent({ id: "flip-t2", takerOrderId: "flip-o2", side: "SELL", size: 80, price: 0.7 })
      );
      // Position closed to zero, so it should not appear in getPositions
      expect(store.getPositions()).toHaveLength(0);
    });

    test("realizedPnl is correct when oversized opposing fill closes position", () => {
      store.addOrder(makeOpenOrder({ orderId: "pnl-o1", originalSize: 50, remainingSize: 50 }));
      store.addOrder(makeOpenOrder({ orderId: "pnl-o2", originalSize: 80, remainingSize: 80 }));
      store.addOrder(makeOpenOrder({ orderId: "pnl-o3", originalSize: 10, remainingSize: 10 }));
      // Enter long 50 @ 0.4
      store.applyFill(makeTradeEvent({ id: "pnl-t1", takerOrderId: "pnl-o1", side: "BUY", size: 50, price: 0.4 }));
      // Sell 80 @ 0.7: closedSize = min(50, 80) = 50 → pnl = (0.7 - 0.4) * 50 = 15
      store.applyFill(
        makeTradeEvent({ id: "pnl-t2", takerOrderId: "pnl-o2", side: "SELL", size: 80, price: 0.7 })
      );
      // Position size is now 0; getPositions() filters it out.
      // Re-open the position on the same assetId to read the accumulated pnl.
      store.applyFill(
        makeTradeEvent({ id: "pnl-t3", takerOrderId: "pnl-o3", side: "SELL", size: 10, price: 0.6 })
      );
      const pos = store.getPositions()[0]!;
      // realizedPnl accumulates on the position object — it was 15 before re-open
      expect(pos.realizedPnl).toBeCloseTo(15, 10);
    });

    test("full close sets avgEntryPrice to 0 and position hidden from getPositions", () => {
      store.addOrder(makeOpenOrder({ orderId: "close-o1", originalSize: 50, remainingSize: 50 }));
      store.addOrder(makeOpenOrder({ orderId: "close-o2", originalSize: 50, remainingSize: 50 }));
      store.applyFill(makeTradeEvent({ id: "close-t1", takerOrderId: "close-o1", side: "BUY", size: 50, price: 0.5 }));
      store.applyFill(
        makeTradeEvent({ id: "close-t2", takerOrderId: "close-o2", side: "SELL", size: 50, price: 0.6 })
      );
      expect(store.getPositions()).toHaveLength(0);
    });

    test("partial opposing fill reduces position size without flipping", () => {
      store.addOrder(makeOpenOrder({ orderId: "part-o1", originalSize: 100, remainingSize: 100 }));
      store.addOrder(makeOpenOrder({ orderId: "part-o2", originalSize: 40, remainingSize: 40 }));
      // Enter long 100 @ 0.5
      store.applyFill(makeTradeEvent({ id: "partial-t1", takerOrderId: "part-o1", side: "BUY", size: 100, price: 0.5 }));
      // Sell 40 @ 0.7 → remaining long 60
      store.applyFill(
        makeTradeEvent({ id: "partial-t2", takerOrderId: "part-o2", side: "SELL", size: 40, price: 0.7 })
      );
      const pos = store.getPositions()[0]!;
      expect(pos.side).toBe("BUY");
      expect(pos.size).toBe(60);
    });
  });

  // ── applyOrderEvent ───────────────────────────────────────────────────────

  describe("applyOrderEvent", () => {
    test("ignores order events for orders not in knownOrderIds", () => {
      // Do NOT call addOrder — simulate another user's order event
      store.applyOrderEvent(
        makeOrderEvent({
          id: "foreign-ord",
          assetId: "asset-xyz",
          side: "SELL",
          orderEventType: "PLACEMENT",
          originalSize: 200,
          sizeMatched: 0,
          price: 0.7,
        })
      );
      expect(store.getOpenOrders()).toHaveLength(0);
    });

    test("PLACEMENT: updates remaining size on known order", () => {
      // First register via addOrder (simulates CLOB response)
      store.addOrder(makeOpenOrder({
        orderId: "evt-ord-1",
        assetId: "asset-xyz",
        side: "SELL",
        price: 0.7,
        originalSize: 200,
        remainingSize: 200,
      }));
      // Then PLACEMENT confirmation arrives with some already matched
      store.applyOrderEvent(
        makeOrderEvent({
          id: "evt-ord-1",
          assetId: "asset-xyz",
          side: "SELL",
          orderEventType: "PLACEMENT",
          originalSize: 200,
          sizeMatched: 30,
          price: 0.7,
        })
      );
      const o = store.getOpenOrders()[0]!;
      expect(o.orderId).toBe("evt-ord-1");
      expect(o.remainingSize).toBe(170);
    });

    test("UPDATE: adjusts remainingSize based on new sizeMatched", () => {
      store.addOrder(makeOpenOrder({
        orderId: "upd-ord",
        originalSize: 100,
        remainingSize: 100,
      }));
      store.applyOrderEvent(
        makeOrderEvent({
          id: "upd-ord",
          orderEventType: "UPDATE",
          originalSize: 100,
          sizeMatched: 40,
        })
      );
      const o = store.getOpenOrders()[0]!;
      expect(o.remainingSize).toBe(60);
    });

    test("UPDATE: removes order when fully matched", () => {
      store.addOrder(makeOpenOrder({
        orderId: "upd-ord-full",
        originalSize: 100,
        remainingSize: 100,
      }));
      store.applyOrderEvent(
        makeOrderEvent({
          id: "upd-ord-full",
          orderEventType: "UPDATE",
          originalSize: 100,
          sizeMatched: 100,
        })
      );
      expect(store.getOpenOrders()).toHaveLength(0);
    });

    test("CANCELLATION: removes order", () => {
      store.addOrder(makeOpenOrder({ orderId: "can-ord", originalSize: 50, remainingSize: 50 }));
      expect(store.getOpenOrders()).toHaveLength(1);
      store.applyOrderEvent(
        makeOrderEvent({ id: "can-ord", orderEventType: "CANCELLATION" })
      );
      expect(store.getOpenOrders()).toHaveLength(0);
    });

    test("CANCELLATION: no-op when order not in knownOrderIds", () => {
      // Should not throw
      expect(() =>
        store.applyOrderEvent(
          makeOrderEvent({ id: "ghost", orderEventType: "CANCELLATION" })
        )
      ).not.toThrow();
      expect(store.getOpenOrders()).toHaveLength(0);
    });
  });

  // ── getPositions ──────────────────────────────────────────────────────────

  describe("getPositions", () => {
    test("only returns positions with non-zero size", () => {
      store.addOrder(makeOpenOrder({ orderId: "gp-o1", assetId: "a1", originalSize: 50, remainingSize: 50 }));
      store.addOrder(makeOpenOrder({ orderId: "gp-o2", assetId: "a1", originalSize: 50, remainingSize: 50 }));
      store.addOrder(makeOpenOrder({ orderId: "gp-o3", assetId: "a2", originalSize: 30, remainingSize: 30 }));
      // Open then fully close a position
      store.applyFill(makeTradeEvent({ id: "gp-t1", takerOrderId: "gp-o1", assetId: "a1", side: "BUY", size: 50, price: 0.5 }));
      store.applyFill(makeTradeEvent({ id: "gp-t2", takerOrderId: "gp-o2", assetId: "a1", side: "SELL", size: 50, price: 0.5 }));
      // Open a second
      store.applyFill(makeTradeEvent({ id: "gp-t3", takerOrderId: "gp-o3", assetId: "a2", side: "BUY", size: 30, price: 0.4 }));
      const positions = store.getPositions();
      expect(positions).toHaveLength(1);
      expect(positions[0]!.assetId).toBe("a2");
    });

    test("returns empty array when no positions", () => {
      expect(store.getPositions()).toHaveLength(0);
    });
  });

  // ── reset ─────────────────────────────────────────────────────────────────

  describe("reset", () => {
    test("clears all positions and open orders", () => {
      store.addOrder(makeOpenOrder({ orderId: "o1" }));
      store.applyFill(makeTradeEvent({ takerOrderId: "o1", side: "BUY", size: 100, price: 0.5 }));
      expect(store.getPositions()).toHaveLength(1);
      expect(store.getOpenOrders()).toHaveLength(0); // order consumed by fill

      store.addOrder(makeOpenOrder({ orderId: "o2" })); // add a fresh one
      store.reset();

      expect(store.getPositions()).toHaveLength(0);
      expect(store.getOpenOrders()).toHaveLength(0);
    });

    test("store is usable after reset", () => {
      store.addOrder(makeOpenOrder({ orderId: "order-1" }));
      store.applyFill(makeTradeEvent({ id: "reset-t1", side: "BUY", size: 100, price: 0.5 }));
      store.reset();
      store.addOrder(makeOpenOrder({ orderId: "order-after" }));
      store.applyFill(makeTradeEvent({ id: "reset-t2", takerOrderId: "order-after", side: "SELL", size: 20, price: 0.6 }));
      const positions = store.getPositions();
      expect(positions).toHaveLength(1);
      expect(positions[0]!.side).toBe("SELL");
      expect(positions[0]!.size).toBe(20);
    });
  });

  // ── snapshot ──────────────────────────────────────────────────────────────

  describe("snapshot", () => {
    test("returns consistent view of positions and open orders", () => {
      store.addOrder(makeOpenOrder({ orderId: "snap-ord" }));
      store.addOrder(makeOpenOrder({ orderId: "snap-fill-ord", originalSize: 30, remainingSize: 30 }));
      store.applyFill(makeTradeEvent({ id: "snap-t1", side: "BUY", size: 30, price: 0.4, takerOrderId: "snap-fill-ord" }));
      const snap = store.snapshot();
      expect(snap.positions).toHaveLength(1);
      expect(snap.openOrders).toHaveLength(1); // snap-ord still open, snap-fill-ord consumed
    });
  });
});
