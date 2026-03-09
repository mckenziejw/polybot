import type { ExternalDataPoint } from "../types.ts";
import type { ExternalFeed, ExternalFeedHandler } from "./external-feed.ts";

const BACKOFF_STEPS_MS = [1_000, 2_000, 4_000, 8_000, 16_000, 30_000];

interface BookTickerMessage {
  u: number;   // order book updateId
  s: string;   // symbol
  b: string;   // best bid price
  B: string;   // best bid qty
  a: string;   // best ask price
  A: string;   // best ask qty
  T: number;   // transaction time
}

export class BinanceFeed implements ExternalFeed {
  readonly name: string;
  private readonly wsUrl: string;

  private handlers = new Set<ExternalFeedHandler>();
  private ws: WebSocket | null = null;
  private running = false;
  private reconnectAttempt = 0;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;

  constructor(symbol: string = "btcusdt") {
    const sym = symbol.toLowerCase();
    this.name = `binance-${sym}`;
    this.wsUrl = `wss://stream.binance.com:9443/ws/${sym}@bookTicker`;
  }

  on(event: "data", handler: ExternalFeedHandler): void {
    if (event === "data") {
      this.handlers.add(handler);
    }
  }

  off(event: "data", handler: ExternalFeedHandler): void {
    if (event === "data") {
      this.handlers.delete(handler);
    }
  }

  async start(): Promise<void> {
    if (this.running) return;
    this.running = true;
    this.reconnectAttempt = 0;
    this.connect();
  }

  async stop(): Promise<void> {
    this.running = false;
    if (this.reconnectTimer !== null) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    if (this.ws !== null) {
      this.ws.close();
      this.ws = null;
    }
  }

  private connect(): void {
    if (!this.running) return;

    const ws = new WebSocket(this.wsUrl);
    this.ws = ws;

    ws.onopen = () => {
      this.reconnectAttempt = 0;
    };

    ws.onmessage = (event: MessageEvent) => {
      try {
        const msg = JSON.parse(event.data as string) as BookTickerMessage;
        const bid = Number(msg.b);
        const ask = Number(msg.a);
        const point: ExternalDataPoint = {
          feed: this.name,
          timestamp: Date.now(),
          price: (bid + ask) / 2,
          bid,
          ask,
        };
        for (const handler of this.handlers) {
          handler(point);
        }
      } catch {
        // Malformed message — silently skip.
      }
    };

    ws.onerror = (_event: Event) => {
      // onclose will fire immediately after onerror; reconnect logic lives there.
    };

    ws.onclose = (_event: CloseEvent) => {
      this.ws = null;
      if (!this.running) return;
      this.scheduleReconnect();
    };
  }

  private scheduleReconnect(): void {
    const delayMs =
      BACKOFF_STEPS_MS[Math.min(this.reconnectAttempt, BACKOFF_STEPS_MS.length - 1)];
    this.reconnectAttempt++;
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      this.connect();
    }, delayMs);
  }
}
