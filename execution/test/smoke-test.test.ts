import { describe, it, expect, beforeEach } from "bun:test";
import { SmokeTestStrategy } from "../src/strategy/smoke-test.ts";
import type { MarketState, MarketInfo, OrderBookState, Position, OpenOrder } from "../src/types.ts";

function makeMarket(): MarketInfo {
  return {
    slug: "btc-up-5m-test",
    conditionId: "0xcond",
    upTokenId: "up-token",
    downTokenId: "down-token",
    endTime: new Date(Date.now() + 300_000),
    tickSize: 0.01,
  };
}

function makeBook(assetId: string, bestBid: number, bestAsk: number): OrderBookState {
  return {
    assetId,
    bids: bestBid > 0 ? [{ price: bestBid, size: 100 }] : [],
    asks: bestAsk > 0 ? [{ price: bestAsk, size: 100 }] : [],
    timestamp: Date.now(),
    hash: "h",
  };
}

function makeState(
  overrides: {
    upBid?: number;
    upAsk?: number;
    downBid?: number;
    downAsk?: number;
    positions?: Position[];
    openOrders?: OpenOrder[];
  } = {}
): MarketState {
  const market = makeMarket();
  const {
    upBid = 0.48,
    upAsk = 0.52,
    downBid = 0.48,
    downAsk = 0.52,
    positions = [],
    openOrders = [],
  } = overrides;

  return {
    market,
    books: {
      [market.upTokenId]: makeBook(market.upTokenId, upBid, upAsk),
      [market.downTokenId]: makeBook(market.downTokenId, downBid, downAsk),
    },
    positions,
    openOrders,
    timeRemainingMs: 250_000,
    timestamp: Date.now(),
  };
}

describe("SmokeTestStrategy", () => {
  let strategy: SmokeTestStrategy;

  beforeEach(async () => {
    strategy = new SmokeTestStrategy();
    await strategy.onMarketOpen(makeState());
  });

  it("has correct id and dataMode", () => {
    expect(strategy.id).toBe("smoke-test");
    expect(strategy.dataMode).toBe("tick");
  });

  describe("WAITING phase", () => {
    it("does nothing when mid is below 0.4", async () => {
      const actions = await strategy.evaluate(makeState({ upBid: 0.30, upAsk: 0.35 }));
      expect(actions).toEqual([{ type: "NOOP" }]);
    });

    it("does nothing when mid is above 0.6", async () => {
      const actions = await strategy.evaluate(makeState({ upBid: 0.65, upAsk: 0.70 }));
      expect(actions).toEqual([{ type: "NOOP" }]);
    });

    it("does nothing when up book has no bids", async () => {
      const actions = await strategy.evaluate(makeState({ upBid: 0, upAsk: 0.50 }));
      expect(actions).toEqual([{ type: "NOOP" }]);
    });

    it("does nothing when down book has no bids", async () => {
      const actions = await strategy.evaluate(makeState({ downBid: 0 }));
      expect(actions).toEqual([{ type: "NOOP" }]);
    });

    it("places two limit buys when mid is in [0.4, 0.6]", async () => {
      const actions = await strategy.evaluate(makeState({ upBid: 0.48, upAsk: 0.52, downBid: 0.47 }));
      expect(actions).toHaveLength(2);
      expect(actions[0]).toMatchObject({
        type: "PLACE_ORDER",
        assetId: "up-token",
        side: "BUY",
        price: 0.48,
        size: 5,
      });
      expect(actions[1]).toMatchObject({
        type: "PLACE_ORDER",
        assetId: "down-token",
        side: "BUY",
        price: 0.47,
        size: 5,
      });
    });

    it("transitions to POSTED after placing orders", async () => {
      await strategy.evaluate(makeState());
      // Second call should not place new orders (now in POSTED phase)
      const actions = await strategy.evaluate(makeState());
      expect(actions).toEqual([{ type: "NOOP" }]);
    });
  });

  describe("POSTED phase — both fill", () => {
    it("transitions to BOTH_HELD when both positions appear", async () => {
      // Place orders
      await strategy.evaluate(makeState());

      // Both sides fill
      const actions = await strategy.evaluate(
        makeState({
          positions: [
            { assetId: "up-token", side: "BUY", size: 5, avgEntryPrice: 0.48, realizedPnl: 0 },
            { assetId: "down-token", side: "BUY", size: 5, avgEntryPrice: 0.48, realizedPnl: 0 },
          ],
        })
      );
      // Should cancel any remaining orders or noop
      const types = actions.map((a) => a.type);
      expect(types.every((t) => t === "CANCEL_ALL" || t === "NOOP")).toBe(true);

      // Subsequent calls should noop (holding to expiry)
      const next = await strategy.evaluate(makeState({
        positions: [
          { assetId: "up-token", side: "BUY", size: 5, avgEntryPrice: 0.48, realizedPnl: 0 },
          { assetId: "down-token", side: "BUY", size: 5, avgEntryPrice: 0.48, realizedPnl: 0 },
        ],
      }));
      expect(next).toEqual([{ type: "NOOP" }]);
    });
  });

  describe("POSTED phase — timeout", () => {
    it("cancels all and goes to DONE when no fills after 3s", async () => {
      // Place orders
      await strategy.evaluate(makeState());

      // Simulate 3s passing by manipulating postedAt
      (strategy as any).postedAt = Date.now() - 3001;

      const actions = await strategy.evaluate(makeState({ openOrders: [{ orderId: "o1", assetId: "up-token", side: "BUY", price: 0.48, originalSize: 5, remainingSize: 5, placedAt: 0 }] }));
      expect(actions).toEqual([{ type: "CANCEL_ALL" }]);

      // Should be DONE now
      const next = await strategy.evaluate(makeState());
      expect(next).toEqual([{ type: "NOOP" }]);
    });

    it("transitions to ONE_HELD when one side fills before timeout", async () => {
      await strategy.evaluate(makeState());
      (strategy as any).postedAt = Date.now() - 3001;

      const actions = await strategy.evaluate(
        makeState({
          positions: [
            { assetId: "up-token", side: "BUY", size: 5, avgEntryPrice: 0.48, realizedPnl: 0 },
          ],
        })
      );
      expect(actions).toEqual([{ type: "CANCEL_ALL" }]);

      // Now in ONE_HELD, should monitor mid price
      const held = await strategy.evaluate(makeState({ upBid: 0.49, upAsk: 0.51 }));
      expect(held).toEqual([{ type: "NOOP" }]); // deviation < 0.1
    });
  });

  describe("ONE_HELD phase — exit conditions", () => {
    async function setupOneHeld(strat: SmokeTestStrategy): Promise<void> {
      await strat.evaluate(makeState());
      (strat as any).postedAt = Date.now() - 3001;
      await strat.evaluate(
        makeState({
          positions: [
            { assetId: "up-token", side: "BUY", size: 5, avgEntryPrice: 0.50, realizedPnl: 0 },
          ],
        })
      );
    }

    it("sells when mid moves up > 0.1 from cost basis (take profit)", async () => {
      await setupOneHeld(strategy);

      const actions = await strategy.evaluate(makeState({ upBid: 0.61, upAsk: 0.63 }));
      expect(actions).toHaveLength(1);
      expect(actions[0]).toMatchObject({
        type: "PLACE_ORDER",
        assetId: "up-token",
        side: "SELL",
        price: 0.61,
      });
    });

    it("sells when mid moves down > 0.1 from cost basis (stop loss)", async () => {
      await setupOneHeld(strategy);

      const actions = await strategy.evaluate(makeState({ upBid: 0.38, upAsk: 0.40 }));
      expect(actions).toHaveLength(1);
      expect(actions[0]).toMatchObject({
        type: "PLACE_ORDER",
        assetId: "up-token",
        side: "SELL",
        price: 0.38,
      });
    });

    it("holds when deviation is within threshold", async () => {
      await setupOneHeld(strategy);

      const actions = await strategy.evaluate(makeState({ upBid: 0.52, upAsk: 0.56 }));
      expect(actions).toEqual([{ type: "NOOP" }]);
    });

    it("transitions to DONE after selling", async () => {
      await setupOneHeld(strategy);

      await strategy.evaluate(makeState({ upBid: 0.61, upAsk: 0.63 }));
      const next = await strategy.evaluate(makeState());
      expect(next).toEqual([{ type: "NOOP" }]);
    });
  });

  describe("onMarketClose", () => {
    it("cancels all orders", async () => {
      const actions = await strategy.onMarketClose(makeState());
      expect(actions).toEqual([{ type: "CANCEL_ALL" }]);
    });
  });

  describe("once-per-market guarantee", () => {
    it("does not trade again after DONE", async () => {
      // Place and timeout with no fills
      await strategy.evaluate(makeState());
      (strategy as any).postedAt = Date.now() - 3001;
      await strategy.evaluate(makeState());

      // Now DONE — should never place orders again even with perfect conditions
      const actions = await strategy.evaluate(makeState({ upBid: 0.49, upAsk: 0.51, downBid: 0.49 }));
      expect(actions).toEqual([{ type: "NOOP" }]);
    });

    it("resets on new market open", async () => {
      // Go to DONE
      await strategy.evaluate(makeState());
      (strategy as any).postedAt = Date.now() - 3001;
      await strategy.evaluate(makeState());

      // New market
      await strategy.onMarketOpen(makeState());

      // Should be back in WAITING and able to trade
      const actions = await strategy.evaluate(makeState());
      expect(actions).toHaveLength(2);
      expect(actions[0]!.type).toBe("PLACE_ORDER");
    });
  });
});
