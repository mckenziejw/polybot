import type { Server } from "bun";
import type {
  AppConfig,
  ExternalDataPoint,
  MarketInfo,
  OrderBookState,
  PriceLevel,
} from "../types.ts";
import type { OrderBookManager } from "../data/orderbook.ts";
import type { PositionStore } from "../positions/position-store.ts";
import type { OrderManager } from "../orders/order-manager.ts";
import type { Minter } from "../positions/minter.ts";
import type { Strategy } from "../strategy/strategy.ts";
import { DASHBOARD_HTML } from "./html.ts";

export interface DashboardRefs {
  bookManager: OrderBookManager;
  positionStore: PositionStore;
  orderManager: OrderManager;
  minter: Minter;
  externalData: Record<string, ExternalDataPoint>;
  getCurrentMarket: () => MarketInfo | null;
  getStrategy: () => Strategy;
  config: AppConfig;
}

interface HistoryPoint {
  t: number;
  p: number;
}

const MAX_HISTORY = 7200; // 1 hour at 500ms

export class DashboardServer {
  private server: Server | null = null;
  private readonly port: number;
  private readonly refs: DashboardRefs;
  private btcHistory: HistoryPoint[] = [];
  private upMidHistory: HistoryPoint[] = [];
  private lastBtcTs = 0;
  private lastUpTs = 0;

  /** When true, the bot is in killed state — strategy evaluate() is blocked. */
  killed = false;

  constructor(port: number, refs: DashboardRefs) {
    this.port = port;
    this.refs = refs;
  }

  start(): void {
    const self = this;

    this.server = Bun.serve({
      port: this.port,
      hostname: "0.0.0.0",
      async fetch(req) {
        const url = new URL(req.url);

        if (url.pathname === "/" || url.pathname === "/index.html") {
          return new Response(DASHBOARD_HTML, {
            headers: { "Content-Type": "text/html; charset=utf-8" },
          });
        }

        if (url.pathname === "/api/state") {
          return Response.json(self.snapshot());
        }

        if (url.pathname === "/api/kill" && req.method === "POST") {
          return Response.json(await self.handleKill());
        }

        if (url.pathname === "/api/sse") {
          const stream = new ReadableStream({
            start(controller) {
              const encoder = new TextEncoder();
              const interval = setInterval(() => {
                try {
                  const data = JSON.stringify(self.snapshot());
                  controller.enqueue(encoder.encode(`data: ${data}\n\n`));
                } catch {
                  clearInterval(interval);
                }
              }, 500);

              req.signal.addEventListener("abort", () => {
                clearInterval(interval);
              });
            },
          });

          return new Response(stream, {
            headers: {
              "Content-Type": "text/event-stream",
              "Cache-Control": "no-cache",
              Connection: "keep-alive",
            },
          });
        }

        return new Response("Not found", { status: 404 });
      },
    });

    console.log(`[Dashboard] http://127.0.0.1:${this.port}`);
  }

  /**
   * Kill switch: cancel all orders, merge matched pairs back to USDC.e,
   * then market-sell any remaining unhedged tokens.
   * Sets `killed = true` so the orchestrator blocks further strategy evaluations.
   */
  private async handleKill(): Promise<Record<string, unknown>> {
    const steps: string[] = [];

    try {
      this.killed = true;
      steps.push("Strategy blocked");
      console.log("[KILL] Kill switch activated — blocking strategy");

      // Step 1: Cancel all open orders
      const cancelOk = await this.refs.orderManager.cancelAll();
      steps.push(cancelOk ? "Orders cancelled" : "Cancel failed (will try nonce)");
      console.log(`[KILL] Cancel all: ${cancelOk ? "OK" : "FAILED"}`);

      // Step 2: Read on-chain token balances
      const market = this.refs.getCurrentMarket();
      if (!market) {
        steps.push("No active market — nothing to flatten");
        return { success: true, steps };
      }

      const [upBalance, downBalance] = await Promise.all([
        this.refs.minter.getTokenBalance(market.upTokenId),
        this.refs.minter.getTokenBalance(market.downTokenId),
      ]);
      const upTokens = Number(upBalance) / 1_000_000;
      const downTokens = Number(downBalance) / 1_000_000;
      steps.push(`On-chain balances: Up=${upTokens.toFixed(2)} Down=${downTokens.toFixed(2)}`);
      console.log(`[KILL] On-chain: Up=${upTokens.toFixed(2)} Down=${downTokens.toFixed(2)}`);

      // Step 3: Merge matched pairs back to USDC.e (no slippage, no fees)
      const pairs = Math.floor(Math.min(upTokens, downTokens));
      if (pairs >= 1) {
        const mergeAmount = BigInt(pairs) * 1_000_000n;
        const mergeResult = await this.refs.minter.merge(market.conditionId, mergeAmount);
        steps.push(`Merge ${pairs} pairs → ${mergeResult.success ? "OK" : (mergeResult.error || "FAILED")}`);
        console.log(`[KILL] Merge ${pairs} pairs: ${mergeResult.success ? "OK" : (mergeResult.error || "FAILED")}`);
      } else {
        steps.push("No matched pairs to merge");
      }

      // Step 4: Market-sell any remaining unhedged tokens
      // Re-read balances after merge to get accurate remainders
      const [upBalanceAfter, downBalanceAfter] = await Promise.all([
        this.refs.minter.getTokenBalance(market.upTokenId),
        this.refs.minter.getTokenBalance(market.downTokenId),
      ]);
      const remainUp = Number(upBalanceAfter) / 1_000_000;
      const remainDown = Number(downBalanceAfter) / 1_000_000;

      const DUMP_PRICE = 0.01;

      if (remainUp >= 5) {
        const result = await this.refs.orderManager.placeOrder({
          type: "PLACE_ORDER",
          assetId: market.upTokenId,
          side: "SELL",
          price: DUMP_PRICE,
          size: remainUp,
          orderType: "GTC",
        });
        steps.push(`Sell Up remainder: ${remainUp.toFixed(2)} @ ${DUMP_PRICE} → ${result.success ? "OK" : (result.errorMsg || "FAILED")}`);
        console.log(`[KILL] Sell Up ${remainUp.toFixed(2)} @ ${DUMP_PRICE}: ${result.success ? "OK" : (result.errorMsg || "FAILED")}`);
      }

      if (remainDown >= 5) {
        const result = await this.refs.orderManager.placeOrder({
          type: "PLACE_ORDER",
          assetId: market.downTokenId,
          side: "SELL",
          price: DUMP_PRICE,
          size: remainDown,
          orderType: "GTC",
        });
        steps.push(`Sell Down remainder: ${remainDown.toFixed(2)} @ ${DUMP_PRICE} → ${result.success ? "OK" : (result.errorMsg || "FAILED")}`);
        console.log(`[KILL] Sell Down ${remainDown.toFixed(2)} @ ${DUMP_PRICE}: ${result.success ? "OK" : (result.errorMsg || "FAILED")}`);
      }

      return { success: true, steps };
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      steps.push(`Error: ${msg}`);
      console.error(`[KILL] Error: ${msg}`);
      return { success: false, steps, error: msg };
    }
  }

  stop(): void {
    this.server?.stop(true);
    this.server = null;
  }

  private snapshot(): Record<string, unknown> {
    const market = this.refs.getCurrentMarket();
    const strategy = this.refs.getStrategy();
    const { positions, openOrders } = this.refs.positionStore.snapshot();
    const now = Date.now();

    // Update BTC history
    const btcPoint = this.refs.externalData["binance-btcusdt"];
    if (btcPoint && btcPoint.timestamp > this.lastBtcTs) {
      this.lastBtcTs = btcPoint.timestamp;
      this.btcHistory.push({ t: btcPoint.timestamp, p: btcPoint.price });
      if (this.btcHistory.length > MAX_HISTORY) {
        this.btcHistory = this.btcHistory.slice(-MAX_HISTORY);
      }
    }

    // Update Up token mid price history
    if (market) {
      const upBook = this.refs.bookManager.get(market.upTokenId);
      const upMid = upBook.mid;
      if (upMid > 0 && now > this.lastUpTs + 400) {
        this.lastUpTs = now;
        this.upMidHistory.push({ t: now, p: upMid });
        if (this.upMidHistory.length > MAX_HISTORY) {
          this.upMidHistory = this.upMidHistory.slice(-MAX_HISTORY);
        }
      }
    }

    // Build orderbook data
    let bookUp: Record<string, unknown> | null = null;
    let bookDown: Record<string, unknown> | null = null;
    if (market) {
      const upBook = this.refs.bookManager.get(market.upTokenId);
      const downBook = this.refs.bookManager.get(market.downTokenId);
      bookUp = {
        bids: upBook.bids.slice(0, 10),
        asks: upBook.asks.slice(0, 10),
        bestBid: upBook.bestBid,
        bestAsk: upBook.bestAsk,
        mid: upBook.mid,
        spread: upBook.spread,
      };
      bookDown = {
        bids: downBook.bids.slice(0, 10),
        asks: downBook.asks.slice(0, 10),
        bestBid: downBook.bestBid,
        bestAsk: downBook.bestAsk,
        mid: downBook.mid,
        spread: downBook.spread,
      };
    }

    return {
      timestamp: now,
      market: market
        ? {
            slug: market.slug,
            conditionId: market.conditionId,
            upTokenId: market.upTokenId,
            downTokenId: market.downTokenId,
            endTime: market.endTime.getTime(),
            timeRemainingMs: Math.max(
              0,
              market.endTime.getTime() - now
            ),
          }
        : null,
      bookUp,
      bookDown,
      positions,
      openOrders,
      btcPrice: btcPoint ?? null,
      btcHistory: this.btcHistory,
      upMidHistory: this.upMidHistory,
      strategy: strategy.getState ? strategy.getState() : null,
      killed: this.killed,
    };
  }

  /** Reset price histories on market rotation. */
  resetHistories(): void {
    this.upMidHistory = [];
    this.lastUpTs = 0;
  }
}
