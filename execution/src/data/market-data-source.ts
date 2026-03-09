import type { AppConfig, MarketDataEvent, MarketInfo } from "../types.ts";
import { RedisMarketData } from "./redis-market-data.ts";
import { WsMarketData } from "./ws-market-data.ts";

// ---------------------------------------------------------------------------
// Handler type & interface
// ---------------------------------------------------------------------------

export type MarketDataHandler = (event: MarketDataEvent) => void;

export interface MarketDataSource {
  on(event: "data", handler: MarketDataHandler): void;
  off(event: "data", handler: MarketDataHandler): void;
  subscribe(market: MarketInfo): Promise<void>;
  start(): Promise<void>;
  stop(): Promise<void>;
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

/**
 * Return the appropriate MarketDataSource implementation based on config.
 * - "websocket" → WsMarketData (direct Polymarket CLOB WebSocket connection)
 * - "redis"     → RedisMarketData (Redis Streams consumer, default)
 */
export function createMarketDataSource(config: AppConfig): MarketDataSource {
  if (config.execution.dataSource === "websocket") {
    return new WsMarketData(config);
  }

  return new RedisMarketData(config);
}
