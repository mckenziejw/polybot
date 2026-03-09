import type { ExternalDataPoint } from "../types.ts";
import { BinanceFeed } from "./binance-feed.ts";

export type ExternalFeedHandler = (point: ExternalDataPoint) => void;

export interface ExternalFeed {
  readonly name: string;
  on(event: "data", handler: ExternalFeedHandler): void;
  off(event: "data", handler: ExternalFeedHandler): void;
  start(): Promise<void>;
  stop(): Promise<void>;
}

export function createExternalFeed(name: string): ExternalFeed {
  // Support any Binance book ticker feed: "binance-btcusdt", "binance-ethusdt", etc.
  if (name.startsWith("binance-")) {
    const symbol = name.slice("binance-".length);
    return new BinanceFeed(symbol);
  }
  throw new Error(`Unknown external feed: "${name}"`);
}
