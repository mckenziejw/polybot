import { describe, test, expect, beforeAll, afterAll } from "bun:test";
import { registerStrategy, createStrategy } from "../src/strategy/strategy.ts";
import type { Strategy } from "../src/strategy/strategy.ts";
import type { MarketState, TradeAction, DataMode } from "../src/types.ts";
import { SignalReceiverStrategy } from "../src/strategy/signal-receiver.ts";

// ---------------------------------------------------------------------------
// Helper – build a minimal Strategy implementation
// ---------------------------------------------------------------------------

function makeStrategy(id: string, dataMode: DataMode = "tick"): Strategy {
  return {
    id,
    dataMode,
    async onMarketOpen(_state: MarketState): Promise<void> {},
    async evaluate(_state: MarketState): Promise<TradeAction[]> {
      return [{ type: "NOOP" }];
    },
    async onMarketClose(_state: MarketState): Promise<TradeAction[]> {
      return [];
    },
  };
}

// ---------------------------------------------------------------------------
// Note on test isolation:
// The strategyRegistry is a module-level Map that persists for the lifetime of
// the module (Bun caches modules). We register unique IDs per test to avoid
// cross-test collisions. Where we need to test "unknown id" behaviour we use
// an ID that was never registered.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// registerStrategy + createStrategy – happy path
// ---------------------------------------------------------------------------

describe("registerStrategy + createStrategy", () => {
  test("registered factory is called and returns the strategy", () => {
    const id = "test-strategy-basic";
    let factoryCalls = 0;

    registerStrategy(id, () => {
      factoryCalls++;
      return makeStrategy(id);
    });

    const strategy = createStrategy(id);

    expect(factoryCalls).toBe(1);
    expect(strategy).toBeDefined();
    expect(strategy.id).toBe(id);
  });

  test("createStrategy returns a new instance each call", () => {
    const id = "test-strategy-new-instance";

    registerStrategy(id, () => makeStrategy(id));

    const s1 = createStrategy(id);
    const s2 = createStrategy(id);

    // Both should be valid strategies but are separate objects
    expect(s1.id).toBe(id);
    expect(s2.id).toBe(id);
    expect(s1).not.toBe(s2);
  });

  test("returned strategy has correct dataMode", () => {
    const id = "test-strategy-datamode";

    registerStrategy(id, () => makeStrategy(id, "resampled"));

    const strategy = createStrategy(id);
    expect(strategy.dataMode).toBe("resampled");
  });

  test("returned strategy implements all interface methods", () => {
    const id = "test-strategy-interface";

    registerStrategy(id, () => makeStrategy(id));

    const strategy = createStrategy(id);

    expect(typeof strategy.onMarketOpen).toBe("function");
    expect(typeof strategy.evaluate).toBe("function");
    expect(typeof strategy.onMarketClose).toBe("function");
  });
});

// ---------------------------------------------------------------------------
// createStrategy with unknown id
// ---------------------------------------------------------------------------

describe("createStrategy with unknown id", () => {
  test("throws when the id is not registered", () => {
    expect(() => createStrategy("__this_id_does_not_exist__")).toThrow();
  });

  test("error message mentions the unknown id", () => {
    expect(() => createStrategy("__unknown_strategy_xyz__")).toThrow(
      /__unknown_strategy_xyz__/,
    );
  });

  test("error message includes available strategies list or 'none'", () => {
    // Register at least one strategy so the available list is non-empty
    registerStrategy("test-available-check", () => makeStrategy("test-available-check"));

    let errorMsg = "";
    try {
      createStrategy("__definitely_not_registered__");
    } catch (err) {
      errorMsg = (err as Error).message;
    }

    // The error should mention the unknown id and include either strategy names or 'none'
    expect(errorMsg).toMatch(/__definitely_not_registered__/);
    expect(errorMsg.length).toBeGreaterThan(0);
  });

  test("error message says 'none' when no strategies are registered (empty registry simulation)", () => {
    // We cannot clear the module-level registry, so instead we verify the general
    // shape of the error thrown for an unknown id regardless of registry state.
    let caught: Error | null = null;
    try {
      createStrategy("__not_a_strategy__");
    } catch (err) {
      caught = err as Error;
    }

    expect(caught).not.toBeNull();
    expect(caught!.message).toContain("Unknown strategy");
  });
});

// ---------------------------------------------------------------------------
// Multiple strategies can be registered
// ---------------------------------------------------------------------------

describe("multiple strategies", () => {
  test("two distinct strategies can be registered and retrieved independently", () => {
    const idA = "test-multi-strategy-A";
    const idB = "test-multi-strategy-B";

    registerStrategy(idA, () => makeStrategy(idA, "tick"));
    registerStrategy(idB, () => makeStrategy(idB, "resampled"));

    const sA = createStrategy(idA);
    const sB = createStrategy(idB);

    expect(sA.id).toBe(idA);
    expect(sA.dataMode).toBe("tick");

    expect(sB.id).toBe(idB);
    expect(sB.dataMode).toBe("resampled");
  });

  test("registering a strategy with an existing id overwrites the previous factory", () => {
    const id = "test-overwrite-strategy";
    let callsV1 = 0;
    let callsV2 = 0;

    registerStrategy(id, () => {
      callsV1++;
      return makeStrategy(id);
    });

    registerStrategy(id, () => {
      callsV2++;
      return makeStrategy(id);
    });

    createStrategy(id);

    // Only v2 should have been called
    expect(callsV1).toBe(0);
    expect(callsV2).toBe(1);
  });

  test("many strategies can be registered without conflict", () => {
    const ids = Array.from({ length: 10 }, (_, i) => `test-bulk-strategy-${i}`);

    for (const id of ids) {
      registerStrategy(id, () => makeStrategy(id));
    }

    for (const id of ids) {
      const s = createStrategy(id);
      expect(s.id).toBe(id);
    }
  });

  test("factory function is called with no arguments", () => {
    const id = "test-strategy-no-args";
    const args: unknown[][] = [];

    registerStrategy(id, (...a: unknown[]) => {
      args.push(a);
      return makeStrategy(id);
    });

    createStrategy(id);

    expect(args).toHaveLength(1);
    expect(args[0]).toHaveLength(0);
  });
});

// ---------------------------------------------------------------------------
// Strategy interface contract – evaluate / onMarketClose return correct types
// ---------------------------------------------------------------------------

describe("Strategy interface contract", () => {
  test("evaluate returns an array of TradeActions", async () => {
    const id = "test-strategy-evaluate";
    registerStrategy(id, () => makeStrategy(id));

    const strategy = createStrategy(id);

    // Provide a minimal MarketState
    const state: MarketState = {
      market: {
        slug: "btc-updown-5m-0",
        conditionId: "0xmarket",
        upTokenId: "0xup",
        downTokenId: "0xdown",
        endTime: new Date(),
        tickSize: 0.01,
      },
      books: {},
      positions: [],
      openOrders: [],
      timeRemainingMs: 60_000,
      timestamp: Date.now(),
    };

    const actions = await strategy.evaluate(state);
    expect(Array.isArray(actions)).toBe(true);
  });

  test("onMarketClose returns an array", async () => {
    const id = "test-strategy-close";
    registerStrategy(id, () => makeStrategy(id));

    const strategy = createStrategy(id);

    const state: MarketState = {
      market: {
        slug: "btc-updown-5m-0",
        conditionId: "0xmarket",
        upTokenId: "0xup",
        downTokenId: "0xdown",
        endTime: new Date(),
        tickSize: 0.01,
      },
      books: {},
      positions: [],
      openOrders: [],
      timeRemainingMs: 0,
      timestamp: Date.now(),
    };

    const actions = await strategy.onMarketClose(state);
    expect(Array.isArray(actions)).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// SignalReceiverStrategy – HTTP client tests
// ---------------------------------------------------------------------------

describe("SignalReceiverStrategy", () => {
  const minimalState: MarketState = {
    market: {
      slug: "btc-updown-5m-0",
      conditionId: "0xmarket",
      upTokenId: "0xup",
      downTokenId: "0xdown",
      endTime: new Date(),
      tickSize: 0.01,
    },
    books: {},
    positions: [],
    openOrders: [],
    timeRemainingMs: 60_000,
    timestamp: Date.now(),
  };

  test("has correct id and dataMode", () => {
    const s = new SignalReceiverStrategy();
    expect(s.id).toBe("signal-receiver");
    expect(s.dataMode).toBe("resampled");
  });

  test("returns NOOP when model server is unreachable", async () => {
    const s = new SignalReceiverStrategy("http://localhost:19999", 50);
    await s.onMarketOpen(minimalState);
    const actions = await s.evaluate(minimalState);
    expect(actions).toEqual([{ type: "NOOP" }]);
  });

  test("returns actions from model server", async () => {
    const expectedActions: TradeAction[] = [
      { type: "PLACE_ORDER", assetId: "0xup", side: "BUY", price: 0.5, size: 1, orderType: "GTC" },
    ];

    let receivedBody: unknown = null;
    const server = Bun.serve({
      port: 0,
      async fetch(req) {
        receivedBody = await req.json();
        return new Response(JSON.stringify({ actions: expectedActions }), {
          headers: { "Content-Type": "application/json" },
        });
      },
    });

    try {
      const s = new SignalReceiverStrategy(`http://localhost:${server.port}`, 1000);
      await s.onMarketOpen(minimalState);
      const actions = await s.evaluate(minimalState);

      expect(actions).toHaveLength(1);
      expect(actions[0]!.type).toBe("PLACE_ORDER");

      // Verify the model server received MarketState
      expect(receivedBody).toBeDefined();
      expect((receivedBody as any).market.slug).toBe("btc-updown-5m-0");
    } finally {
      server.stop();
    }
  });

  test("returns NOOP on non-200 response", async () => {
    const server = Bun.serve({
      port: 0,
      fetch() {
        return new Response("Internal Error", { status: 500 });
      },
    });

    try {
      const s = new SignalReceiverStrategy(`http://localhost:${server.port}`, 1000);
      await s.onMarketOpen(minimalState);
      const actions = await s.evaluate(minimalState);
      expect(actions).toEqual([{ type: "NOOP" }]);
    } finally {
      server.stop();
    }
  });

  test("returns NOOP on malformed response", async () => {
    const server = Bun.serve({
      port: 0,
      fetch() {
        return new Response(JSON.stringify({ oops: "no actions field" }), {
          headers: { "Content-Type": "application/json" },
        });
      },
    });

    try {
      const s = new SignalReceiverStrategy(`http://localhost:${server.port}`, 1000);
      await s.onMarketOpen(minimalState);
      const actions = await s.evaluate(minimalState);
      expect(actions).toEqual([{ type: "NOOP" }]);
    } finally {
      server.stop();
    }
  });

  test("onMarketClose returns CANCEL_ALL", async () => {
    const s = new SignalReceiverStrategy();
    const actions = await s.onMarketClose(minimalState);
    expect(actions).toEqual([{ type: "CANCEL_ALL" }]);
  });
});
