import Redis from "ioredis";
import type {
  AppConfig,
  BookEvent,
  MarketCloseEvent,
  MarketDataEvent,
  MarketInfo,
  MarketOpenEvent,
  OrderBookState,
  PriceChangeEntry,
  PriceChangeEvent,
  PriceLevel,
} from "../types.ts";
import type { MarketDataHandler, MarketDataSource } from "./market-data-source.ts";
import { decodeMarketEvent } from "./proto.ts";

// ---------------------------------------------------------------------------
// Constants — must match redis_publisher.py
// ---------------------------------------------------------------------------

const MARKET_EVENTS_STREAM = "polymarket:market_events";
const CONTROL_STREAM = "polymarket:control";
const CONSUMER_GROUP = "polymarket:execution";
const CONSUMER_NAME = "execution-1";
const BLOCK_MS = 1_000;
const READ_COUNT = 100;

// ---------------------------------------------------------------------------
// Implementation
// ---------------------------------------------------------------------------

export class RedisMarketData implements MarketDataSource {
  private readonly handlers = new Set<MarketDataHandler>();
  private readonly redis: Redis;
  private running = false;

  constructor(private readonly config: AppConfig) {
    this.redis = new Redis(config.execution.redisUrl, {
      lazyConnect: true,
      // Reconnect automatically on connection loss
      retryStrategy: (times) => Math.min(times * 500, 10_000),
    });
  }

  // -------------------------------------------------------------------------
  // MarketDataSource interface
  // -------------------------------------------------------------------------

  on(_event: "data", handler: MarketDataHandler): void {
    this.handlers.add(handler);
  }

  off(_event: "data", handler: MarketDataHandler): void {
    this.handlers.delete(handler);
  }

  /**
   * No-op for Redis mode — market subscriptions are driven entirely by the
   * control stream published by the ingest component.
   */
  async subscribe(_market: MarketInfo): Promise<void> {
    return;
  }

  async start(): Promise<void> {
    await this.redis.connect();
    await this.ensureConsumerGroups();
    await this.bootstrapFromControlStream();
    this.running = true;
    // Run the consume loop asynchronously so start() returns promptly
    this.consumeLoop().catch((err) =>
      console.error("[RedisMarketData] Consume loop crashed:", err)
    );
  }

  async stop(): Promise<void> {
    this.running = false;
    await this.redis.quit();
  }

  // -------------------------------------------------------------------------
  // Consumer group setup
  // -------------------------------------------------------------------------

  private async ensureConsumerGroups(): Promise<void> {
    const streams = [MARKET_EVENTS_STREAM, CONTROL_STREAM];

    for (const stream of streams) {
      try {
        // MKSTREAM creates the stream if it does not yet exist
        await this.redis.xgroup("CREATE", stream, CONSUMER_GROUP, "0", "MKSTREAM");
        console.log(`[RedisMarketData] Created consumer group on ${stream}`);
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : String(err);
        if (msg.includes("BUSYGROUP")) {
          // Already exists — fine
        } else {
          throw err;
        }
      }
    }
  }

  // -------------------------------------------------------------------------
  // Bootstrap: scan control stream history to find active market
  // -------------------------------------------------------------------------

  private async bootstrapFromControlStream(): Promise<void> {
    console.log("[RedisMarketData] Bootstrapping from control stream...");

    let lastOpenFields: Record<string, string> | null = null;
    let lastOpenSlug: string | null = null;
    let lastCloseSlug: string | null = null;

    try {
      // XRANGE returns all messages: [id, fields[]]
      const entries = await this.redis.xrange(CONTROL_STREAM, "-", "+");

      for (const [, fields] of entries) {
        // ioredis returns fields as a flat string array: [key, val, key, val, ...]
        const parsed = flatFieldsToRecord(fields);
        const eventType = parsed["event_type"] ?? "";
        const slug = parsed["slug"] ?? "";

        if (eventType === "market_open") {
          lastOpenSlug = slug;
          lastOpenFields = parsed;
        } else if (eventType === "market_close") {
          lastCloseSlug = slug;
        }
      }

      if (lastOpenSlug && lastOpenSlug !== lastCloseSlug && lastOpenFields) {
        console.log(`[RedisMarketData] Resuming mid-window: ${lastOpenSlug}`);
        const marketInfo = buildMarketInfo(lastOpenFields);
        if (marketInfo) {
          this.emit({ type: "market_open", market: marketInfo });
        }
      } else {
        console.log("[RedisMarketData] No active market window — waiting for market_open");
      }
    } catch (err) {
      console.error("[RedisMarketData] Bootstrap failed:", err);
    }
  }

  // -------------------------------------------------------------------------
  // Main consume loop
  // -------------------------------------------------------------------------

  private async consumeLoop(): Promise<void> {
    const streams: Record<string, string> = {
      [MARKET_EVENTS_STREAM]: ">",
      [CONTROL_STREAM]: ">",
    };

    // Build args array for XREADGROUP:
    // XREADGROUP GROUP <group> <consumer> COUNT <n> BLOCK <ms> STREAMS <key ...> <id ...>
    while (this.running) {
      try {
        const streamKeys = Object.keys(streams);
        const streamIds = Object.values(streams);

        const results = await (this.redis as any).xreadgroup(
          "GROUP",
          CONSUMER_GROUP,
          CONSUMER_NAME,
          "COUNT",
          READ_COUNT,
          "BLOCK",
          BLOCK_MS,
          "STREAMS",
          ...streamKeys,
          ...streamIds
        );

        if (!results) {
          // Timeout — no new messages, loop again
          continue;
        }

        // results: [[streamName, [[id, fields], ...]], ...]
        for (const [streamName, messages] of results as [string, [string, string[]][]][]) {
          for (const [msgId, rawFields] of messages) {
            try {
              await this.handleMessage(streamName, rawFields);
              await this.redis.xack(streamName, CONSUMER_GROUP, msgId);
            } catch (err) {
              console.error(
                `[RedisMarketData] Failed to process message ${msgId} on ${streamName}:`,
                err
              );
            }
          }
        }
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : String(err);
        // Swallow "Connection is closed" noise during shutdown
        if (!this.running && msg.includes("Connection is closed")) break;
        console.error("[RedisMarketData] Error in consume loop:", err);
        // Brief pause to avoid a tight error loop
        await sleep(1_000);
      }
    }
  }

  // -------------------------------------------------------------------------
  // Message dispatch
  // -------------------------------------------------------------------------

  private async handleMessage(streamName: string, rawFields: string[]): Promise<void> {
    const fields = flatFieldsToRecord(rawFields);

    if (streamName === CONTROL_STREAM) {
      this.handleControl(fields);
    } else if (streamName === MARKET_EVENTS_STREAM) {
      this.handleMarketEvent(fields);
    }
  }

  // -------------------------------------------------------------------------
  // Control stream
  // -------------------------------------------------------------------------

  private handleControl(fields: Record<string, string>): void {
    const eventType = fields["event_type"] ?? "";

    if (eventType === "market_open") {
      const market = buildMarketInfo(fields);
      if (!market) {
        console.warn("[RedisMarketData] market_open missing required fields:", fields);
        return;
      }
      console.log(`[RedisMarketData] market_open: ${market.slug}`);
      const event: MarketOpenEvent = { type: "market_open", market };
      this.emit(event);
    } else if (eventType === "market_close") {
      const conditionId = fields["market"] ?? fields["condition_id"] ?? "";
      const slug = fields["slug"] ?? "";
      console.log(`[RedisMarketData] market_close: ${slug}`);
      const event: MarketCloseEvent = { type: "market_close", conditionId };
      this.emit(event);
    } else {
      console.warn(`[RedisMarketData] Unknown control event_type: ${eventType}`);
    }
  }

  // -------------------------------------------------------------------------
  // Market events stream
  // -------------------------------------------------------------------------

  private handleMarketEvent(fields: Record<string, string>): void {
    const dataStr = fields["data"];
    if (!dataStr) {
      console.warn("[RedisMarketData] market_events message missing data field");
      return;
    }

    // ioredis returns binary data as a latin1 string when not in Buffer mode;
    // convert to Buffer for protobuf decoding.
    const buf = Buffer.from(dataStr, "binary");
    const decoded = decodeMarketEvent(buf);

    if (decoded.type === "book") {
      const { book } = decoded;

      const bids: PriceLevel[] = book.bids
        .map((b) => ({ price: b.price, size: b.size }))
        .sort((a, b) => b.price - a.price);

      const asks: PriceLevel[] = book.asks
        .map((a) => ({ price: a.price, size: a.size }))
        .sort((a, b) => a.price - b.price);

      const orderBook: OrderBookState = {
        assetId: book.assetId,
        bids,
        asks,
        timestamp: book.exchangeTimestamp,
        hash: book.hash,
      };

      const event: BookEvent = { type: "book", book: orderBook };
      this.emit(event);
    } else if (decoded.type === "trade") {
      const { trade } = decoded;

      const changes: PriceChangeEntry[] = trade.priceChanges.map((pc) => ({
        assetId: pc.assetId,
        price: pc.price,
        size: pc.size,
        side: pc.side,
        hash: pc.hash,
        bestBid: pc.bestBid,
        bestAsk: pc.bestAsk,
      }));

      const event: PriceChangeEvent = {
        type: "price_change",
        market: trade.market,
        timestamp: trade.exchangeTimestamp,
        changes,
      };
      this.emit(event);
    }
  }

  // -------------------------------------------------------------------------
  // Event emission
  // -------------------------------------------------------------------------

  private emit(event: MarketDataEvent): void {
    for (const handler of this.handlers) {
      try {
        handler(event);
      } catch (err) {
        console.error("[RedisMarketData] Handler error:", err);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * ioredis returns stream message fields as a flat string array:
 * ["key1", "val1", "key2", "val2", ...]
 * Convert to a plain Record for convenient access.
 */
function flatFieldsToRecord(fields: string[]): Record<string, string> {
  const result: Record<string, string> = {};
  for (let i = 0; i + 1 < fields.length; i += 2) {
    result[fields[i]!] = fields[i + 1]!;
  }
  return result;
}

/**
 * Build a MarketInfo from a control stream market_open field map.
 * Returns null if required fields are absent.
 */
function buildMarketInfo(fields: Record<string, string>): MarketInfo | null {
  const slug = fields["slug"] ?? "";
  const conditionId = fields["market"] ?? fields["condition_id"] ?? "";
  const upTokenId = fields["up_token_id"] ?? "";
  const downTokenId = fields["down_token_id"] ?? "";
  const endTimeStr = fields["end_time"] ?? "";

  if (!slug || !conditionId || !upTokenId || !downTokenId || !endTimeStr) {
    return null;
  }

  return {
    slug,
    conditionId,
    upTokenId,
    downTokenId,
    endTime: new Date(endTimeStr),
    // tick_size is not present in control messages; callers may update later
    tickSize: 0,
  };
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
