import path from "path";
import { readFileSync } from "node:fs";
import type {
  AppConfig,
  PolymarketConfig,
  MarketMakerConfig,
  MarketMakerMode,
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
    configPath || process.env.CONFIG || path.resolve(import.meta.dir, "../../config.json");

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
  const modeRaw = String(marketMakerRaw.mode || "mint-and-sell");
  const mode = (modeRaw === "traditional" ? "traditional" : "mint-and-sell") as MarketMakerMode;
  const pricingRaw = String(marketMakerRaw.pricing || "inventory-skew");
  const pricing = (pricingRaw === "whale-front" ? "whale-front" : "inventory-skew") as import("./types.ts").PricingMode;
  const marketMaker: MarketMakerConfig = {
    mode,
    pricing,
    initialPairs: Number(marketMakerRaw.initial_pairs ?? 50),
    maxTotalInventory: Number(marketMakerRaw.max_total_inventory ?? marketMakerRaw.max_inventory_per_side ?? 200),
    maxUnhedgedExposure: Number(marketMakerRaw.max_unhedged_exposure ?? 50),
    skewK: Number(marketMakerRaw.skew_k ?? 0.005),
    requoteThreshold: Number(marketMakerRaw.requote_threshold ?? 0.005),
    replenishThreshold: Number(marketMakerRaw.replenish_threshold ?? 10),
    whaleThreshold: Number(marketMakerRaw.whale_threshold ?? 1000),
    whalePullCooldownMs: Number(marketMakerRaw.whale_pull_cooldown_ms ?? 3000),
    whaleMoveTicks: Number(marketMakerRaw.whale_move_ticks ?? 3),
    // Legacy fields for other strategies
    orderSize: Number(marketMakerRaw.order_size) || 5.0,
    cancelWindowS: Number(marketMakerRaw.cancel_window_s) || 6.0,
    minRemainingS: Number(marketMakerRaw.min_remaining_s) || 30.0,
    maxPositions: Number(marketMakerRaw.max_positions) || 1,
    priceOffset: Number(marketMakerRaw.price_offset) || 0.0,
  };

  // Parse execution config with defaults
  const executionRaw = rawConfig.execution || {};
  const execution: ExecutionConfig = {
    asset: String(executionRaw.asset ?? "btc"),
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
    modelServerUrl: String(executionRaw.modelServerUrl ?? executionRaw.model_server_url ?? "http://127.0.0.1:8000"),
    dashboardPort: executionRaw.dashboard_port ? Number(executionRaw.dashboard_port) : undefined,
    marketDuration: String(executionRaw.market_duration ?? "5m"),
  };

  return {
    polymarket,
    marketMaker,
    execution,
  };
}
