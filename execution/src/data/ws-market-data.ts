import type {
  AppConfig,
  BookEvent,
  MarketDataEvent,
  MarketInfo,
  OrderBookState,
  PriceChangeEntry,
  PriceChangeEvent,
  PriceLevel,
  Side,
} from "../types.ts";
import type { MarketDataHandler, MarketDataSource } from "./market-data-source.ts";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market";
const PING_INTERVAL_MS = 10_000;
const RECONNECT_BASE_MS = 1_000;
const RECONNECT_MAX_MS = 30_000;

// ---------------------------------------------------------------------------
// Raw wire types (all numeric fields arrive as strings from Polymarket)
// ---------------------------------------------------------------------------

interface RawPriceLevel {
  price: string;
  size: string;
}

interface RawBookMessage {
  event_type: "book";
  market: string;
  asset_id: string;
  timestamp: string;
  hash: string;
  bids: RawPriceLevel[];
  asks: RawPriceLevel[];
}

interface RawPriceChange {
  asset_id: string;
  price: string;
  size: string;
  side: string;
  hash: string;
  best_bid: string;
  best_ask: string;
}

interface RawPriceChangeMessage {
  event_type: "price_change";
  market: string;
  timestamp: string;
  price_changes: RawPriceChange[];
}

interface RawIgnoredMessage {
  event_type: "last_trade_price" | "tick_size_change";
}

type RawMessage = RawBookMessage | RawPriceChangeMessage | RawIgnoredMessage;

// ---------------------------------------------------------------------------
// Implementation
// ---------------------------------------------------------------------------

export class WsMarketData implements MarketDataSource {
  private readonly handlers = new Set<MarketDataHandler>();
  private ws: WebSocket | null = null;
  private pingTimer: ReturnType<typeof setInterval> | null = null;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private reconnectDelay = RECONNECT_BASE_MS;
  private stopped = false;

  // Markets to (re-)subscribe on connect
  private subscribedMarkets: MarketInfo[] = [];

  constructor(private readonly config: AppConfig) {}

  // -------------------------------------------------------------------------
  // MarketDataSource interface
  // -------------------------------------------------------------------------

  on(_event: "data", handler: MarketDataHandler): void {
    this.handlers.add(handler);
  }

  off(_event: "data", handler: MarketDataHandler): void {
    this.handlers.delete(handler);
  }

  async subscribe(market: MarketInfo): Promise<void> {
    // Track for reconnect re-subscriptions
    if (!this.subscribedMarkets.find((m) => m.conditionId === market.conditionId)) {
      this.subscribedMarkets.push(market);
    }

    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.sendSubscription(market, "subscribe");
    }
    // If not yet connected, the subscription will be sent in _onOpen
  }

  async start(): Promise<void> {
    this.stopped = false;
    this.connect();
  }

  async stop(): Promise<void> {
    this.stopped = true;
    this.clearPingTimer();
    this.clearReconnectTimer();
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  // -------------------------------------------------------------------------
  // WebSocket lifecycle
  // -------------------------------------------------------------------------

  private connect(): void {
    if (this.stopped) return;

    console.log(`[WsMarketData] Connecting to ${WS_URL}`);
    const ws = new WebSocket(WS_URL);

    ws.addEventListener("open", () => this.onOpen(ws));
    ws.addEventListener("message", (ev) => this.onMessage(ev));
    ws.addEventListener("error", (ev) => this.onError(ev));
    ws.addEventListener("close", (ev) => this.onClose(ev));

    this.ws = ws;
  }

  private onOpen(ws: WebSocket): void {
    console.log("[WsMarketData] Connection opened");
    this.reconnectDelay = RECONNECT_BASE_MS; // reset backoff on successful connect

    // Subscribe to all tracked markets
    for (const market of this.subscribedMarkets) {
      this.sendSubscription(market, "subscribe");
    }

    // Send initial subscription without operation field if no tracked markets
    // (the caller must call subscribe() before or after start())
    this.startPingTimer(ws);
  }

  private onMessage(ev: MessageEvent): void {
    const raw = ev.data as string;

    if (raw === "PONG") {
      return;
    }

    let parsed: RawMessage | RawMessage[];
    try {
      parsed = JSON.parse(raw);
    } catch (err) {
      console.error("[WsMarketData] Failed to parse JSON:", err, raw.slice(0, 200));
      return;
    }

    const messages: RawMessage[] = Array.isArray(parsed) ? parsed : [parsed];

    for (const msg of messages) {
      try {
        this.handleMessage(msg);
      } catch (err) {
        console.error("[WsMarketData] Failed to handle message:", err);
      }
    }
  }

  private onError(ev: Event): void {
    console.error("[WsMarketData] WebSocket error:", ev);
  }

  private onClose(ev: CloseEvent): void {
    console.warn(`[WsMarketData] Connection closed (code=${ev.code}, reason=${ev.reason})`);
    this.clearPingTimer();

    if (!this.stopped) {
      this.scheduleReconnect();
    }
  }

  // -------------------------------------------------------------------------
  // Reconnection
  // -------------------------------------------------------------------------

  private scheduleReconnect(): void {
    const delay = this.reconnectDelay;
    console.log(`[WsMarketData] Reconnecting in ${delay}ms`);

    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      // Exponential backoff: double the delay up to max
      this.reconnectDelay = Math.min(this.reconnectDelay * 2, RECONNECT_MAX_MS);
      this.connect();
    }, delay);
  }

  // -------------------------------------------------------------------------
  // Ping
  // -------------------------------------------------------------------------

  private startPingTimer(ws: WebSocket): void {
    this.clearPingTimer();
    this.pingTimer = setInterval(() => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send("PING");
      } else {
        this.clearPingTimer();
      }
    }, PING_INTERVAL_MS);
  }

  // -------------------------------------------------------------------------
  // Subscription helpers
  // -------------------------------------------------------------------------

  private sendSubscription(market: MarketInfo, operation?: "subscribe"): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;

    const msg: Record<string, unknown> = {
      assets_ids: [market.upTokenId, market.downTokenId],
      type: "market",
    };
    if (operation) {
      msg.operation = operation;
    }

    this.ws.send(JSON.stringify(msg));
  }

  // -------------------------------------------------------------------------
  // Message parsing
  // -------------------------------------------------------------------------

  private handleMessage(msg: RawMessage): void {
    switch (msg.event_type) {
      case "book":
        this.emit(this.parseBook(msg as RawBookMessage));
        break;

      case "price_change":
        this.emit(this.parsePriceChange(msg as RawPriceChangeMessage));
        break;

      case "last_trade_price":
      case "tick_size_change":
        // Intentionally ignored
        break;

      default: {
        const unknownType = (msg as { event_type: string }).event_type;
        console.warn(`[WsMarketData] Unknown event_type: ${unknownType}`);
      }
    }
  }

  private parseBook(raw: RawBookMessage): BookEvent {
    const bids: PriceLevel[] = (raw.bids ?? [])
      .map((b) => ({ price: Number(b.price), size: Number(b.size) }))
      .sort((a, b) => b.price - a.price); // descending

    const asks: PriceLevel[] = (raw.asks ?? [])
      .map((a) => ({ price: Number(a.price), size: Number(a.size) }))
      .sort((a, b) => a.price - b.price); // ascending

    const book: OrderBookState = {
      assetId: raw.asset_id,
      bids,
      asks,
      timestamp: Number(raw.timestamp),
      hash: raw.hash,
    };

    return { type: "book", book };
  }

  private parsePriceChange(raw: RawPriceChangeMessage): PriceChangeEvent {
    const changes: PriceChangeEntry[] = (raw.price_changes ?? []).map((pc) => ({
      assetId: pc.asset_id,
      price: Number(pc.price),
      size: Number(pc.size),
      side: pc.side as Side,
      hash: pc.hash,
      bestBid: Number(pc.best_bid),
      bestAsk: Number(pc.best_ask),
    }));

    return {
      type: "price_change",
      market: raw.market,
      timestamp: Number(raw.timestamp),
      changes,
    };
  }

  // -------------------------------------------------------------------------
  // Event emission
  // -------------------------------------------------------------------------

  private emit(event: MarketDataEvent): void {
    for (const handler of this.handlers) {
      try {
        handler(event);
      } catch (err) {
        console.error("[WsMarketData] Handler error:", err);
      }
    }
  }

  // -------------------------------------------------------------------------
  // Timer cleanup helpers
  // -------------------------------------------------------------------------

  private clearPingTimer(): void {
    if (this.pingTimer !== null) {
      clearInterval(this.pingTimer);
      this.pingTimer = null;
    }
  }

  private clearReconnectTimer(): void {
    if (this.reconnectTimer !== null) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
  }
}
