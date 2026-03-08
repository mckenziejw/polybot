import type {
  UserOrderEvent,
  UserTradeEvent,
  UserChannelEvent,
  Side,
  OrderEventType,
  OrderStatus,
  OrderType,
  TradeStatus,
  TraderSide,
  MakerOrderInfo,
} from "../types.ts";

const WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/user";
const PING_INTERVAL_MS = 10_000;
const RECONNECT_BASE_MS = 1_000;
const RECONNECT_MAX_MS = 30_000;

export type UserChannelHandler = (event: UserChannelEvent) => void;

interface UserChannelConfig {
  apiKey: string;
  apiSecret: string;
  apiPassphrase: string;
}

/**
 * WebSocket subscription to Polymarket user channel for real-time
 * order placements, updates, cancellations, and trade fills.
 *
 * Auth: credentials are passed directly in the subscription message.
 * These should be derived L2 credentials (from ClobClient.createOrDeriveApiKey),
 * not raw config values.
 */
export class UserChannel {
  private ws: WebSocket | null = null;
  private handlers: UserChannelHandler[] = [];
  private pingInterval: ReturnType<typeof setInterval> | null = null;
  private reconnectAttempt = 0;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private running = false;
  private subscribedMarkets: string[] = [];

  constructor(private config: UserChannelConfig) {}

  /** Update credentials (e.g. with derived L2 creds from OrderManager). */
  setCredentials(creds: UserChannelConfig): void {
    this.config = creds;
  }

  on(event: "data", handler: UserChannelHandler): void {
    this.handlers.push(handler);
  }

  off(event: "data", handler: UserChannelHandler): void {
    this.handlers = this.handlers.filter((h) => h !== handler);
  }

  private emit(event: UserChannelEvent): void {
    for (const handler of this.handlers) {
      try {
        handler(event);
      } catch (err) {
        console.error("[UserChannel] Handler error:", err);
      }
    }
  }

  /** Subscribe to order/trade events for a market (by conditionId). */
  subscribeMarket(conditionId: string): void {
    if (!this.subscribedMarkets.includes(conditionId)) {
      this.subscribedMarkets.push(conditionId);
    }

    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(
        JSON.stringify({
          operation: "subscribe",
          markets: [conditionId],
        })
      );
    }
  }

  /** Unsubscribe from a market. */
  unsubscribeMarket(conditionId: string): void {
    this.subscribedMarkets = this.subscribedMarkets.filter((m) => m !== conditionId);

    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(
        JSON.stringify({
          operation: "unsubscribe",
          markets: [conditionId],
        })
      );
    }
  }

  async start(): Promise<void> {
    this.running = true;
    return this.connect();
  }

  async stop(): Promise<void> {
    this.running = false;
    if (this.pingInterval) {
      clearInterval(this.pingInterval);
      this.pingInterval = null;
    }
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  private connect(): Promise<void> {
    return new Promise<void>((resolve, reject) => {
      const ws = new WebSocket(WS_URL);
      this.ws = ws;

      ws.onopen = () => {
        console.log("[UserChannel] Connected");
        this.reconnectAttempt = 0;

        // Send authenticated subscription
        const subscribeMsg = {
          auth: {
            apiKey: this.config.apiKey,
            secret: this.config.apiSecret,
            passphrase: this.config.apiPassphrase,
          },
          type: "user",
          markets: this.subscribedMarkets.length > 0 ? this.subscribedMarkets : undefined,
        };
        ws.send(JSON.stringify(subscribeMsg));

        // Start ping
        this.pingInterval = setInterval(() => {
          if (ws.readyState === WebSocket.OPEN) {
            ws.send("PING");
          }
        }, PING_INTERVAL_MS);

        resolve();
      };

      ws.onmessage = (event) => {
        const data = typeof event.data === "string" ? event.data : "";

        // Ignore PONG
        if (data === "PONG") return;

        try {
          const parsed = JSON.parse(data);
          // Can be a single event or array of events
          const events = Array.isArray(parsed) ? parsed : [parsed];
          for (const raw of events) {
            const evt = this.parseEvent(raw);
            if (evt) this.emit(evt);
          }
        } catch (err) {
          console.error("[UserChannel] Parse error:", err, data.slice(0, 200));
        }
      };

      ws.onerror = (err) => {
        console.error("[UserChannel] WebSocket error:", err);
        if (this.reconnectAttempt === 0) reject(err);
      };

      ws.onclose = () => {
        console.log("[UserChannel] Disconnected");
        if (this.pingInterval) {
          clearInterval(this.pingInterval);
          this.pingInterval = null;
        }
        if (this.running) {
          this.scheduleReconnect();
        }
      };
    });
  }

  private scheduleReconnect(): void {
    const delay = Math.min(
      RECONNECT_BASE_MS * Math.pow(2, this.reconnectAttempt),
      RECONNECT_MAX_MS
    );
    console.log(`[UserChannel] Reconnecting in ${delay}ms (attempt ${this.reconnectAttempt + 1})`);
    this.reconnectAttempt++;
    this.reconnectTimer = setTimeout(() => {
      this.connect().catch((err) => {
        console.error("[UserChannel] Reconnect failed:", err);
      });
    }, delay);
  }

  private parseEvent(raw: Record<string, unknown>): UserChannelEvent | null {
    const eventType = raw.event_type;

    if (eventType === "order") {
      return this.parseOrderEvent(raw);
    }
    if (eventType === "trade") {
      return this.parseTradeEvent(raw);
    }

    // Unknown event type — ignore
    return null;
  }

  private parseOrderEvent(raw: Record<string, unknown>): UserOrderEvent {
    return {
      eventType: "order",
      id: String(raw.id ?? ""),
      owner: String(raw.owner ?? ""),
      market: String(raw.market ?? ""),
      assetId: String(raw.asset_id ?? ""),
      side: String(raw.side ?? "BUY") as Side,
      originalSize: Number(raw.original_size ?? 0),
      sizeMatched: Number(raw.size_matched ?? 0),
      price: Number(raw.price ?? 0),
      associateTrades: raw.associate_trades as string[] | null ?? null,
      outcome: String(raw.outcome ?? ""),
      orderEventType: String(raw.type ?? "PLACEMENT") as OrderEventType,
      createdAt: String(raw.created_at ?? ""),
      expiration: String(raw.expiration ?? ""),
      orderType: String(raw.order_type ?? "GTC") as OrderType,
      status: String(raw.status ?? "LIVE") as OrderStatus,
      makerAddress: String(raw.maker_address ?? ""),
      timestamp: Number(raw.timestamp ?? 0),
    };
  }

  private parseTradeEvent(raw: Record<string, unknown>): UserTradeEvent {
    const makerOrders = Array.isArray(raw.maker_orders)
      ? (raw.maker_orders as Record<string, unknown>[]).map(
          (mo): MakerOrderInfo => ({
            orderId: String(mo.order_id ?? ""),
            owner: String(mo.owner ?? ""),
            makerAddress: String(mo.maker_address ?? ""),
            matchedAmount: Number(mo.matched_amount ?? 0),
            price: Number(mo.price ?? 0),
            feeRateBps: Number(mo.fee_rate_bps ?? 0),
            assetId: String(mo.asset_id ?? ""),
            outcome: String(mo.outcome ?? ""),
            side: String(mo.side ?? "BUY") as Side,
          })
        )
      : [];

    return {
      eventType: "trade",
      id: String(raw.id ?? ""),
      takerOrderId: String(raw.taker_order_id ?? ""),
      market: String(raw.market ?? ""),
      assetId: String(raw.asset_id ?? ""),
      side: String(raw.side ?? "BUY") as Side,
      size: Number(raw.size ?? 0),
      price: Number(raw.price ?? 0),
      feeRateBps: Number(raw.fee_rate_bps ?? 0),
      status: String(raw.status ?? "MATCHED") as TradeStatus,
      matchTime: String(raw.matchtime ?? ""),
      lastUpdate: String(raw.last_update ?? ""),
      outcome: String(raw.outcome ?? ""),
      owner: String(raw.owner ?? ""),
      tradeOwner: String(raw.trade_owner ?? ""),
      makerAddress: String(raw.maker_address ?? ""),
      transactionHash: String(raw.transaction_hash ?? ""),
      bucketIndex: Number(raw.bucket_index ?? 0),
      makerOrders,
      traderSide: String(raw.trader_side ?? "TAKER") as TraderSide,
      timestamp: Number(raw.timestamp ?? 0),
    };
  }
}
