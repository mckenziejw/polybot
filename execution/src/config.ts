import path from "path";
import { readFileSync } from "node:fs";
import type {
  AppConfig,
  PolymarketConfig,
  MarketMakerConfig,
  ExecutionConfig,
} from "./types.ts";

interface RawConfig {
  polymarket?: Record<string, unknown>;
  market_maker?: Record<string, unknown>;
  execution?: Record<string, unknown>;
}

/**
 * Load and parse the application config from a JSON file.
 * Converts snake_case JSON keys to camelCase TypeScript interface properties.
 * Provides sensible defaults for the execution section.
 *
 * @param configPath - Path to config.json. Defaults to ../../config.json relative to this file.
 * @returns Validated AppConfig object
 * @throws Error if required fields are missing or file cannot be read
 */
export function loadConfig(configPath?: string): AppConfig {
  const resolvedPath =
    configPath || path.resolve(import.meta.dir, "../../config.json");

  let rawConfig: RawConfig;
  try {
    const fileContent = readFileSync(resolvedPath, "utf-8");
    rawConfig = JSON.parse(fileContent) as RawConfig;
  } catch (error) {
    throw new Error(
      `Failed to load config from ${resolvedPath}: ${error instanceof Error ? error.message : String(error)}`
    );
  }

  // Parse polymarket config
  const polymarketRaw = rawConfig.polymarket || {};
  if (!polymarketRaw.private_key) {
    throw new Error(
      "Missing required field: polymarket.private_key in config"
    );
  }
  if (!polymarketRaw.host) {
    throw new Error("Missing required field: polymarket.host in config");
  }

  const polymarket: PolymarketConfig = {
    host: String(polymarketRaw.host),
    chainId: Number(polymarketRaw.chain_id) || 137,
    privateKey: String(polymarketRaw.private_key),
    apiKey: String(polymarketRaw.api_key || ""),
    apiSecret: String(polymarketRaw.api_secret || ""),
    apiPassphrase: String(polymarketRaw.api_passphrase || ""),
    proxyWallet: String(polymarketRaw.proxy_wallet || ""),
  };

  // Parse market maker config
  const marketMakerRaw = rawConfig.market_maker || {};
  const marketMaker: MarketMakerConfig = {
    orderSize: Number(marketMakerRaw.order_size) || 5.0,
    cancelWindowS: Number(marketMakerRaw.cancel_window_s) || 6.0,
    minRemainingS: Number(marketMakerRaw.min_remaining_s) || 30.0,
    maxPositions: Number(marketMakerRaw.max_positions) || 1,
    priceOffset: Number(marketMakerRaw.price_offset) || 0.0,
  };

  // Parse execution config with defaults
  const executionRaw = rawConfig.execution || {};
  const execution: ExecutionConfig = {
    dataSource: (String(executionRaw.dataSource ?? executionRaw.data_source ?? "websocket")) as "redis" | "websocket",
    redisUrl: String(executionRaw.redisUrl ?? executionRaw.redis_url ?? "redis://localhost:6379"),
    strategyId: String(executionRaw.strategyId ?? executionRaw.strategy_id ?? "market-maker"),
    metricsDir: String(executionRaw.metricsDir ?? executionRaw.metrics_dir ?? "data/metrics"),
    rpcUrl: String(executionRaw.rpcUrl ?? executionRaw.rpc_url ?? "https://polygon-rpc.com"),
    resampleIntervalMs: Number(executionRaw.resampleIntervalMs ?? executionRaw.resample_interval_ms ?? 100),
    externalFeeds: (Array.isArray(executionRaw.externalFeeds) ? executionRaw.externalFeeds
      : Array.isArray(executionRaw.external_feeds) ? executionRaw.external_feeds
      : []) as string[],
    betDollars: Number(executionRaw.betDollars ?? executionRaw.bet_dollars ?? 5),
  };

  return {
    polymarket,
    marketMaker,
    execution,
  };
}
