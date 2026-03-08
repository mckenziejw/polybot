import { describe, test, expect, beforeEach, afterEach } from "bun:test";
import { writeFileSync, mkdirSync, rmSync, existsSync } from "node:fs";
import { join } from "node:path";
import { loadConfig } from "../src/config.ts";

// Helpers

function makeTempDir(): string {
  const rand = Math.random().toString(36).slice(2, 10);
  const dir = `/tmp/test-config-${rand}`;
  mkdirSync(dir, { recursive: true });
  return dir;
}

function writeConfig(dir: string, obj: unknown): string {
  const filePath = join(dir, "config.json");
  writeFileSync(filePath, JSON.stringify(obj), "utf-8");
  return filePath;
}

/** Minimal valid config object. */
const VALID_BASE = {
  polymarket: {
    private_key: "0xdeadbeef",
    host: "https://clob.polymarket.com",
    chain_id: 137,
    api_key: "my-api-key",
    api_secret: "my-api-secret",
    api_passphrase: "my-passphrase",
    proxy_wallet: "0xproxy",
  },
  market_maker: {
    order_size: 10,
    cancel_window_s: 5,
    min_remaining_s: 20,
    max_positions: 2,
    price_offset: 0.01,
  },
  execution: {
    data_source: "redis",
    redis_url: "redis://localhost:6380",
    strategy_id: "momentum",
    metrics_dir: "/tmp/metrics",
    rpc_url: "https://my-rpc.example.com",
    resample_interval_ms: 250,
    external_feeds: ["binance"],
  },
};

describe("loadConfig", () => {
  let tmpDir: string;

  beforeEach(() => {
    tmpDir = makeTempDir();
  });

  afterEach(() => {
    if (existsSync(tmpDir)) {
      rmSync(tmpDir, { recursive: true, force: true });
    }
  });

  // ── Valid config ───────────────────────────────────────────────────────────

  describe("valid config", () => {
    test("loads polymarket fields correctly", () => {
      const configPath = writeConfig(tmpDir, VALID_BASE);
      const config = loadConfig(configPath);

      expect(config.polymarket.privateKey).toBe("0xdeadbeef");
      expect(config.polymarket.host).toBe("https://clob.polymarket.com");
      expect(config.polymarket.chainId).toBe(137);
      expect(config.polymarket.apiKey).toBe("my-api-key");
      expect(config.polymarket.apiSecret).toBe("my-api-secret");
      expect(config.polymarket.apiPassphrase).toBe("my-passphrase");
      expect(config.polymarket.proxyWallet).toBe("0xproxy");
    });

    test("loads market_maker fields correctly", () => {
      const configPath = writeConfig(tmpDir, VALID_BASE);
      const config = loadConfig(configPath);

      expect(config.marketMaker.orderSize).toBe(10);
      expect(config.marketMaker.cancelWindowS).toBe(5);
      expect(config.marketMaker.minRemainingS).toBe(20);
      expect(config.marketMaker.maxPositions).toBe(2);
      expect(config.marketMaker.priceOffset).toBe(0.01);
    });

    test("loads execution fields correctly (snake_case keys)", () => {
      const configPath = writeConfig(tmpDir, VALID_BASE);
      const config = loadConfig(configPath);

      expect(config.execution.dataSource).toBe("redis");
      expect(config.execution.redisUrl).toBe("redis://localhost:6380");
      expect(config.execution.strategyId).toBe("momentum");
      expect(config.execution.metricsDir).toBe("/tmp/metrics");
      expect(config.execution.rpcUrl).toBe("https://my-rpc.example.com");
      expect(config.execution.resampleIntervalMs).toBe(250);
      expect(config.execution.externalFeeds).toEqual(["binance"]);
    });
  });

  // ── Missing required fields ───────────────────────────────────────────────

  describe("missing required fields", () => {
    test("throws when polymarket.private_key is missing", () => {
      const cfg = {
        ...VALID_BASE,
        polymarket: { ...VALID_BASE.polymarket, private_key: undefined },
      };
      const configPath = writeConfig(tmpDir, cfg);
      expect(() => loadConfig(configPath)).toThrow(/private_key/);
    });

    test("throws when polymarket.host is missing", () => {
      const cfg = {
        ...VALID_BASE,
        polymarket: { ...VALID_BASE.polymarket, host: undefined },
      };
      const configPath = writeConfig(tmpDir, cfg);
      expect(() => loadConfig(configPath)).toThrow(/host/);
    });

    test("throws when the file does not exist", () => {
      expect(() => loadConfig("/tmp/nonexistent-config-file-xyz.json")).toThrow();
    });

    test("throws when the file is not valid JSON", () => {
      const configPath = join(tmpDir, "bad.json");
      writeFileSync(configPath, "not json {{{{", "utf-8");
      expect(() => loadConfig(configPath)).toThrow();
    });
  });

  // ── Defaults ──────────────────────────────────────────────────────────────

  describe("defaults", () => {
    test("execution section uses defaults when omitted entirely", () => {
      const cfg = { polymarket: VALID_BASE.polymarket };
      const configPath = writeConfig(tmpDir, cfg);
      const config = loadConfig(configPath);

      expect(config.execution.dataSource).toBe("websocket");
      expect(config.execution.redisUrl).toBe("redis://localhost:6379");
      expect(config.execution.strategyId).toBe("market-maker");
      expect(config.execution.metricsDir).toBe("data/metrics");
      expect(config.execution.rpcUrl).toBe("https://polygon-rpc.com");
      expect(config.execution.resampleIntervalMs).toBe(100);
      expect(config.execution.externalFeeds).toEqual([]);
    });

    test("market_maker section uses defaults when omitted entirely", () => {
      const cfg = { polymarket: VALID_BASE.polymarket };
      const configPath = writeConfig(tmpDir, cfg);
      const config = loadConfig(configPath);

      expect(config.marketMaker.orderSize).toBe(5.0);
      expect(config.marketMaker.cancelWindowS).toBe(6.0);
      expect(config.marketMaker.minRemainingS).toBe(30.0);
      expect(config.marketMaker.maxPositions).toBe(1);
      expect(config.marketMaker.priceOffset).toBe(0.0);
    });

    test("polymarket.chainId defaults to 137 when absent", () => {
      const { chain_id: _, ...polymarketWithoutChainId } = VALID_BASE.polymarket;
      const cfg = { polymarket: polymarketWithoutChainId };
      const configPath = writeConfig(tmpDir, cfg);
      const config = loadConfig(configPath);
      expect(config.polymarket.chainId).toBe(137);
    });

    test("externalFeeds defaults to empty array when key is absent", () => {
      const cfg = {
        polymarket: VALID_BASE.polymarket,
        execution: {
          data_source: "websocket",
          // no external_feeds key
        },
      };
      const configPath = writeConfig(tmpDir, cfg);
      const config = loadConfig(configPath);
      expect(config.execution.externalFeeds).toEqual([]);
    });
  });

  // ── snake_case and camelCase both accepted for execution fields ───────────

  describe("snake_case and camelCase keys for execution", () => {
    test("camelCase keys are accepted", () => {
      const cfg = {
        polymarket: VALID_BASE.polymarket,
        execution: {
          dataSource: "redis",
          redisUrl: "redis://camel:6379",
          strategyId: "camel-strategy",
          metricsDir: "/camel/metrics",
          rpcUrl: "https://camel-rpc.example.com",
          resampleIntervalMs: 500,
          externalFeeds: ["feed-a", "feed-b"],
        },
      };
      const configPath = writeConfig(tmpDir, cfg);
      const config = loadConfig(configPath);

      expect(config.execution.dataSource).toBe("redis");
      expect(config.execution.redisUrl).toBe("redis://camel:6379");
      expect(config.execution.strategyId).toBe("camel-strategy");
      expect(config.execution.metricsDir).toBe("/camel/metrics");
      expect(config.execution.rpcUrl).toBe("https://camel-rpc.example.com");
      expect(config.execution.resampleIntervalMs).toBe(500);
      expect(config.execution.externalFeeds).toEqual(["feed-a", "feed-b"]);
    });

    test("snake_case keys are accepted", () => {
      const cfg = {
        polymarket: VALID_BASE.polymarket,
        execution: {
          data_source: "websocket",
          redis_url: "redis://snake:6379",
          strategy_id: "snake-strategy",
          metrics_dir: "/snake/metrics",
          rpc_url: "https://snake-rpc.example.com",
          resample_interval_ms: 200,
          external_feeds: ["feed-x"],
        },
      };
      const configPath = writeConfig(tmpDir, cfg);
      const config = loadConfig(configPath);

      expect(config.execution.dataSource).toBe("websocket");
      expect(config.execution.redisUrl).toBe("redis://snake:6379");
      expect(config.execution.strategyId).toBe("snake-strategy");
      expect(config.execution.metricsDir).toBe("/snake/metrics");
      expect(config.execution.rpcUrl).toBe("https://snake-rpc.example.com");
      expect(config.execution.resampleIntervalMs).toBe(200);
      expect(config.execution.externalFeeds).toEqual(["feed-x"]);
    });

    test("camelCase takes precedence over snake_case when both are present", () => {
      const cfg = {
        polymarket: VALID_BASE.polymarket,
        execution: {
          dataSource: "redis",       // camelCase wins
          data_source: "websocket",  // should be ignored
          redisUrl: "redis://camel-wins:6379",
          redis_url: "redis://snake-loses:6379",
        },
      };
      const configPath = writeConfig(tmpDir, cfg);
      const config = loadConfig(configPath);

      expect(config.execution.dataSource).toBe("redis");
      expect(config.execution.redisUrl).toBe("redis://camel-wins:6379");
    });
  });
});
