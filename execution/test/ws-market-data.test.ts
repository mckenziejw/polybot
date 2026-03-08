import { describe, test, expect } from "bun:test";
import { WsMarketData } from "../src/data/ws-market-data.ts";
import type { MarketDataHandler } from "../src/data/market-data-source.ts";
import type { AppConfig, MarketDataEvent, MarketInfo } from "../src/types.ts";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeConfig(overrides: Partial<AppConfig> = {}): AppConfig {
  return {
    polymarket: {
      host: "https://clob.polymarket.com",
      chainId: 137,
      privateKey: "0xdeadbeef",
      apiKey: "test-key",
      apiSecret: "test-secret",
      apiPassphrase: "test-passphrase",
      proxyWallet: "0x0000",
    },
    marketMaker: {
      orderSize: 10,
      cancelWindowS: 30,
      minRemainingS: 60,
      maxPositions: 3,
      priceOffset: 0.01,
    },
    execution: {
      dataSource: "websocket",
      redisUrl: "redis://localhost:6379",
      strategyId: "test-strategy",
      metricsDir: "/tmp/metrics",
      rpcUrl: "https://polygon-rpc.com",
      resampleIntervalMs: 100,
      externalFeeds: [],
    },
    ...overrides,
  };
}

function makeMarket(id = "0xmarket1"): MarketInfo {
  return {
    slug: "btc-updown-5m-1700000000",
    conditionId: id,
    upTokenId: "0xup",
    downTokenId: "0xdown",
    endTime: new Date(Date.now() + 300_000),
    tickSize: 0.01,
  };
}

// ---------------------------------------------------------------------------
// Instantiation
// ---------------------------------------------------------------------------

describe("WsMarketData – instantiation", () => {
  test("can be constructed with a valid AppConfig", () => {
    const src = new WsMarketData(makeConfig());
    expect(src).toBeDefined();
    expect(src).toBeInstanceOf(WsMarketData);
  });

  test("accepts different config shapes without throwing", () => {
    const cfg = makeConfig();
    cfg.execution.resampleIntervalMs = 500;
    const src = new WsMarketData(cfg);
    expect(src).toBeInstanceOf(WsMarketData);
  });
});

// ---------------------------------------------------------------------------
// on / off handler registration
// ---------------------------------------------------------------------------

describe("WsMarketData – on/off handler registration", () => {
  test("on('data') registers a handler without throwing", () => {
    const src = new WsMarketData(makeConfig());
    const handler: MarketDataHandler = () => {};
    expect(() => src.on("data", handler)).not.toThrow();
  });

  test("multiple distinct handlers can be registered", () => {
    const src = new WsMarketData(makeConfig());
    const events1: MarketDataEvent[] = [];
    const events2: MarketDataEvent[] = [];

    const h1: MarketDataHandler = (e) => events1.push(e);
    const h2: MarketDataHandler = (e) => events2.push(e);

    expect(() => {
      src.on("data", h1);
      src.on("data", h2);
    }).not.toThrow();

    // No events emitted without a live WS
    expect(events1).toHaveLength(0);
    expect(events2).toHaveLength(0);
  });

  test("off('data') removes a registered handler without throwing", () => {
    const src = new WsMarketData(makeConfig());
    const handler: MarketDataHandler = () => {};

    src.on("data", handler);
    expect(() => src.off("data", handler)).not.toThrow();
  });

  test("off('data') with an unregistered handler does not throw", () => {
    const src = new WsMarketData(makeConfig());
    const handler: MarketDataHandler = () => {};
    expect(() => src.off("data", handler)).not.toThrow();
  });

  test("same handler can be added and removed repeatedly", () => {
    const src = new WsMarketData(makeConfig());
    const handler: MarketDataHandler = () => {};

    expect(() => {
      src.on("data", handler);
      src.off("data", handler);
      src.on("data", handler);
      src.off("data", handler);
    }).not.toThrow();
  });

  test("removing one handler does not affect others", () => {
    const src = new WsMarketData(makeConfig());
    const toRemove: MarketDataHandler = () => {};
    const toKeep: MarketDataHandler = () => {};

    src.on("data", toRemove);
    src.on("data", toKeep);
    src.off("data", toRemove);

    // Instance should still be intact
    expect(src).toBeInstanceOf(WsMarketData);
  });
});

// ---------------------------------------------------------------------------
// subscribe() – market tracking (without a live WS)
// ---------------------------------------------------------------------------

describe("WsMarketData – subscribe", () => {
  test("subscribe() resolves without error when WS is not connected", async () => {
    const src = new WsMarketData(makeConfig());
    await expect(src.subscribe(makeMarket())).resolves.toBeUndefined();
  });

  test("subscribe() with the same conditionId is idempotent", async () => {
    const src = new WsMarketData(makeConfig());
    const market = makeMarket("0xdup");
    await src.subscribe(market);
    await expect(src.subscribe(market)).resolves.toBeUndefined();
  });

  test("multiple distinct markets can be subscribed", async () => {
    const src = new WsMarketData(makeConfig());
    await src.subscribe(makeMarket("0xa"));
    await src.subscribe(makeMarket("0xb"));
    await expect(src.subscribe(makeMarket("0xc"))).resolves.toBeUndefined();
  });
});

// ---------------------------------------------------------------------------
// stop() without start() – should be safe
// ---------------------------------------------------------------------------

describe("WsMarketData – stop without start", () => {
  test("stop() resolves without error when never started", async () => {
    const src = new WsMarketData(makeConfig());
    await expect(src.stop()).resolves.toBeUndefined();
  });

  test("stop() is idempotent", async () => {
    const src = new WsMarketData(makeConfig());
    await src.stop();
    await expect(src.stop()).resolves.toBeUndefined();
  });
});
