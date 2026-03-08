/**
 * Paper trading harness for the market maker strategy.
 *
 * Connects to live Polymarket WebSocket + Binance feed, runs the strategy,
 * and simulates fills when limit orders cross the book. No CLOB API
 * credentials needed — this is read-only + simulation.
 *
 * Logs three CSV files to data/paper/:
 *   - ticks.csv:   per-tick market state snapshot (sampled at ~1/sec to avoid bloat)
 *   - trades.csv:  every order placement, fill, and cancellation
 *   - markets.csv: per-market summary with resolution outcome
 *
 * Usage: bun run src/paper-trade.ts [--markets N] [--strategy ID]
 */

import { appendFileSync, mkdirSync, statSync } from "node:fs";
import { join } from "node:path";
import { WsMarketData } from "./data/ws-market-data.ts";
import { OrderBookManager } from "./data/orderbook.ts";
import { BinanceFeed } from "./data/binance-feed.ts";
import { MarketRotation } from "./rotation/market-rotation.ts";
import { createStrategy } from "./strategy/strategy.ts";
import type {
  MarketInfo,
  MarketState,
  MarketDataEvent,
  TradeAction,
  ExternalDataPoint,
  PlaceOrderAction,
  Position,
  OpenOrder,
  OrderBookState,
  Side,
} from "./types.ts";

// Register strategies
import "./strategy/market-maker.ts";
import "./strategy/signal-receiver.ts";
import "./strategy/smoke-test.ts";
import "./strategy/accumulator.ts";
import "./strategy/gap-trader.ts";
import "./strategy/momentum.ts";
import "./strategy/mid-drift.ts";

// ── Normal CDF (same as strategy, duplicated to avoid coupling) ──

function normalCdf(x: number): number {
  if (x < -8) return 0;
  if (x > 8) return 1;
  const a1 = 0.319381530, a2 = -0.356563782, a3 = 1.781477937;
  const a4 = -1.821255978, a5 = 1.330274429, p = 0.2316419;
  const sign = x < 0 ? -1 : 1;
  const absX = Math.abs(x);
  const t = 1 / (1 + p * absX);
  const pdf = Math.exp(-0.5 * absX * absX) / Math.sqrt(2 * Math.PI);
  const cdf = 1 - pdf * t * (a1 + t * (a2 + t * (a3 + t * (a4 + t * a5))));
  return sign < 0 ? 1 - cdf : cdf;
}

// ── Rolling vol tracker (same logic as strategy) ──

class VolTracker {
  private ticks: { t: number; p: number }[] = [];
  private windowMs: number;

  constructor(windowMs: number = 30_000) {
    this.windowMs = windowMs;
  }

  reset(): void { this.ticks = []; }

  push(timestamp: number, price: number): void {
    this.ticks.push({ t: timestamp, p: price });
    const cutoff = timestamp - this.windowMs;
    while (this.ticks.length > 0 && this.ticks[0]!.t < cutoff) {
      this.ticks.shift();
    }
    if (this.ticks.length > 6000) {
      this.ticks = this.ticks.slice(-6000);
    }
  }

  get tickCount(): number { return this.ticks.length; }

  realizedVol(scaleDurationMs: number): number {
    if (this.ticks.length < 10) return 0;
    let sumSqReturns = 0;
    for (let i = 1; i < this.ticks.length; i++) {
      const r = Math.log(this.ticks[i]!.p / this.ticks[i - 1]!.p);
      sumSqReturns += r * r;
    }
    const windowDurationMs = this.ticks[this.ticks.length - 1]!.t - this.ticks[0]!.t;
    if (windowDurationMs <= 0) return 0;
    return Math.sqrt((sumSqReturns / windowDurationMs) * scaleDurationMs);
  }
}

// ── CSV Logger ──

class PaperCsvLogger {
  private ticksPath: string;
  private tradesPath: string;
  private marketsPath: string;

  constructor(outDir: string) {
    mkdirSync(outDir, { recursive: true });
    const sessionId = new Date().toISOString().replace(/[:.]/g, "").slice(0, 15);
    this.ticksPath = join(outDir, `ticks_${sessionId}.csv`);
    this.tradesPath = join(outDir, `trades_${sessionId}.csv`);
    this.marketsPath = join(outDir, `markets_${sessionId}.csv`);

    this.initFile(this.ticksPath, [
      "timestamp", "market_slug", "time_elapsed_s", "time_remaining_s",
      "btc_price", "btc_return_pct", "realized_vol_pct",
      "fair_up", "fair_down", "z_score",
      "up_best_bid", "up_best_ask", "up_mid", "up_spread",
      "down_best_bid", "down_best_ask", "down_mid", "down_spread",
      "up_position", "down_position", "net_inventory",
      "open_order_count", "cumulative_fills", "cumulative_pnl",
    ].join(","));

    this.initFile(this.tradesPath, [
      "timestamp", "market_slug", "event_type",
      "order_id", "asset_id", "token_label", "side",
      "price", "size", "order_type",
      "fill_price", "fill_size",
      "position_after", "avg_entry_after", "realized_pnl",
      "btc_price", "fair_value", "time_remaining_s",
    ].join(","));

    this.initFile(this.marketsPath, [
      "market_slug", "condition_id", "up_token_id", "down_token_id",
      "start_time", "end_time",
      "btc_open", "btc_close", "btc_return_pct",
      "total_ticks", "total_orders", "total_fills", "total_cancels",
      "realized_pnl_trading",
      "up_position_at_close", "up_avg_entry",
      "down_position_at_close", "down_avg_entry",
      "resolution_winner", "resolution_pnl", "total_pnl",
    ].join(","));
  }

  private initFile(path: string, header: string): void {
    try { statSync(path); } catch {
      appendFileSync(path, header + "\n");
    }
  }

  private esc(val: unknown): string {
    const s = String(val ?? "");
    return s.includes(",") || s.includes('"') || s.includes("\n")
      ? '"' + s.replace(/"/g, '""') + '"'
      : s;
  }

  logTick(fields: Record<string, unknown>): void {
    const row = [
      fields.timestamp, fields.market_slug, fields.time_elapsed_s, fields.time_remaining_s,
      fields.btc_price, fields.btc_return_pct, fields.realized_vol_pct,
      fields.fair_up, fields.fair_down, fields.z_score,
      fields.up_best_bid, fields.up_best_ask, fields.up_mid, fields.up_spread,
      fields.down_best_bid, fields.down_best_ask, fields.down_mid, fields.down_spread,
      fields.up_position, fields.down_position, fields.net_inventory,
      fields.open_order_count, fields.cumulative_fills, fields.cumulative_pnl,
    ].map(this.esc).join(",");
    appendFileSync(this.ticksPath, row + "\n");
  }

  logTrade(fields: Record<string, unknown>): void {
    const row = [
      fields.timestamp, fields.market_slug, fields.event_type,
      fields.order_id, fields.asset_id, fields.token_label, fields.side,
      fields.price, fields.size, fields.order_type,
      fields.fill_price, fields.fill_size,
      fields.position_after, fields.avg_entry_after, fields.realized_pnl,
      fields.btc_price, fields.fair_value, fields.time_remaining_s,
    ].map(this.esc).join(",");
    appendFileSync(this.tradesPath, row + "\n");
  }

  logMarket(fields: Record<string, unknown>): void {
    const row = [
      fields.market_slug, fields.condition_id, fields.up_token_id, fields.down_token_id,
      fields.start_time, fields.end_time,
      fields.btc_open, fields.btc_close, fields.btc_return_pct,
      fields.total_ticks, fields.total_orders, fields.total_fills, fields.total_cancels,
      fields.realized_pnl_trading,
      fields.up_position_at_close, fields.up_avg_entry,
      fields.down_position_at_close, fields.down_avg_entry,
      fields.resolution_winner, fields.resolution_pnl, fields.total_pnl,
    ].map(this.esc).join(",");
    appendFileSync(this.marketsPath, row + "\n");
  }

  get paths() {
    return { ticks: this.ticksPath, trades: this.tradesPath, markets: this.marketsPath };
  }
}

// ── Paper position / order tracking ──

interface PaperOrder {
  orderId: string;
  assetId: string;
  side: Side;
  price: number;
  size: number;
  remainingSize: number;
  placedAt: number;
  orderType: string;
}

interface PaperPosition {
  assetId: string;
  side: Side;
  size: number;
  avgEntryPrice: number;
  realizedPnl: number;
}

interface PaperFill {
  orderId: string;
  assetId: string;
  side: Side;
  price: number;
  size: number;
}

class PaperTrader {
  private orders = new Map<string, PaperOrder>();
  private positions = new Map<string, PaperPosition>();
  private nextOrderId = 1;
  totalPnl = 0;
  fillCount = 0;
  orderCount = 0;
  cancelCount = 0;

  placeOrder(action: PlaceOrderAction): string {
    const orderId = `paper-${this.nextOrderId++}`;
    this.orders.set(orderId, {
      orderId,
      assetId: action.assetId,
      side: action.side,
      price: action.price,
      size: action.size,
      remainingSize: action.size,
      placedAt: Date.now(),
      orderType: action.orderType,
    });
    this.orderCount++;
    return orderId;
  }

  cancelOrder(orderId: string): boolean {
    if (this.orders.delete(orderId)) {
      this.cancelCount++;
      return true;
    }
    return false;
  }

  cancelAll(): number {
    const count = this.orders.size;
    this.cancelCount += count;
    this.orders.clear();
    return count;
  }

  checkFills(books: Record<string, OrderBookState>): PaperFill[] {
    const fills: PaperFill[] = [];
    const now = Date.now();
    for (const [orderId, order] of this.orders) {
      const book = books[order.assetId];
      if (!book) continue;

      // We're a maker — fill triggers when the market crosses through our price,
      // but we fill at OUR order price (that's how we capture spread)
      let filled = false;
      if (order.side === "BUY") {
        // Our bid gets hit when someone sells at or below our price
        if (book.asks.length > 0 && order.price >= book.asks[0]!.price) filled = true;
      } else {
        // Our ask gets lifted when someone buys at or above our price
        if (book.bids.length > 0 && order.price <= book.bids[0]!.price) filled = true;
      }

      if (filled) {
        this.applyFill(order);
        fills.push({ orderId, assetId: order.assetId, side: order.side, price: order.price, size: order.remainingSize });
        this.orders.delete(orderId);
        this.fillCount++;
      }
    }
    return fills;
  }

  private applyFill(order: PaperOrder): void {
    const existing = this.positions.get(order.assetId);
    if (order.side === "BUY") {
      if (existing) {
        const totalCost = existing.avgEntryPrice * existing.size + order.price * order.remainingSize;
        existing.size += order.remainingSize;
        existing.avgEntryPrice = totalCost / existing.size;
      } else {
        this.positions.set(order.assetId, {
          assetId: order.assetId, side: "BUY", size: order.remainingSize,
          avgEntryPrice: order.price, realizedPnl: 0,
        });
      }
    } else {
      if (existing && existing.size > 0) {
        const sellSize = Math.min(order.remainingSize, existing.size);
        const pnl = (order.price - existing.avgEntryPrice) * sellSize;
        existing.realizedPnl += pnl;
        existing.size -= sellSize;
        this.totalPnl += pnl;
        if (existing.size <= 0) this.positions.delete(order.assetId);
      }
    }
  }

  resolveMarket(winningAssetId: string): number {
    let resolutionPnl = 0;
    for (const [assetId, pos] of this.positions) {
      if (pos.size <= 0) continue;
      const payout = assetId === winningAssetId ? 1.0 : 0.0;
      const pnl = (payout - pos.avgEntryPrice) * pos.size;
      resolutionPnl += pnl;
      this.totalPnl += pnl;
    }
    this.positions.clear();
    return resolutionPnl;
  }

  getOpenOrders(): OpenOrder[] {
    return [...this.orders.values()].map((o) => ({
      orderId: o.orderId, assetId: o.assetId, side: o.side, price: o.price,
      originalSize: o.size, remainingSize: o.remainingSize, placedAt: o.placedAt,
    }));
  }

  getPositions(): Position[] {
    return [...this.positions.values()]
      .filter((p) => p.size > 0)
      .map((p) => ({
        assetId: p.assetId, side: p.side, size: p.size,
        avgEntryPrice: p.avgEntryPrice, realizedPnl: p.realizedPnl, settled: true,
      }));
  }

  getPosition(assetId: string): PaperPosition | undefined {
    return this.positions.get(assetId);
  }

  get openOrderCount(): number { return this.orders.size; }

  resetForNewMarket(): void {
    this.orders.clear();
  }
}

// ── Resolution fetcher (lightweight, matches market_analysis.py logic) ──

const GAMMA_API = "https://gamma-api.polymarket.com";

interface MarketResolution {
  slug: string;
  winningTokenId: string;
  winnerLabel: string; // "Up" or "Down"
}

async function fetchResolution(slug: string): Promise<MarketResolution | null> {
  try {
    const resp = await fetch(`${GAMMA_API}/markets/slug/${slug}`, { signal: AbortSignal.timeout(10_000) });
    if (!resp.ok) return null;
    const data = await resp.json() as Record<string, unknown>;
    if (data.umaResolutionStatus !== "resolved") return null;

    const tokenIds = JSON.parse(data.clobTokenIds as string) as string[];
    const outcomes = JSON.parse(data.outcomes as string) as string[];
    const outcomePrices = JSON.parse(data.outcomePrices as string) as string[];

    const winningIdx = outcomePrices.findIndex((p: string) => parseFloat(p) === 1.0);
    if (winningIdx === -1) return null;

    return {
      slug,
      winningTokenId: tokenIds[winningIdx]!,
      winnerLabel: outcomes[winningIdx]!,
    };
  } catch {
    return null;
  }
}

// ── Config ──

const STRATEGY_ID = process.argv.includes("--strategy")
  ? process.argv[process.argv.indexOf("--strategy") + 1]!
  : "market-maker";
const MAX_MARKETS = process.argv.includes("--markets")
  ? parseInt(process.argv[process.argv.indexOf("--markets") + 1]!, 10)
  : 3;
const OUT_DIR = process.argv.includes("--outdir")
  ? process.argv[process.argv.indexOf("--outdir") + 1]!
  : "data/paper";

const fakeConfig = {
  polymarket: {
    host: "https://clob.polymarket.com", chainId: 137, privateKey: "",
    apiKey: "", apiSecret: "", apiPassphrase: "", proxyWallet: "",
  },
  marketMaker: {
    orderSize: 5, cancelWindowS: 6, minRemainingS: 30, maxPositions: 1, priceOffset: 0,
  },
  execution: {
    dataSource: "websocket" as const, redisUrl: "", strategyId: STRATEGY_ID,
    metricsDir: "", rpcUrl: "", resampleIntervalMs: 100,
    externalFeeds: ["binance-btcusdt"],
  },
};

// ── Helpers ──

function ts(): string { return new Date().toISOString().slice(11, 23); }

function tokenLabel(assetId: string, market: MarketInfo): string {
  if (assetId === market.upTokenId) return "Up";
  if (assetId === market.downTokenId) return "Down";
  return assetId.slice(0, 8);
}

function formatAction(action: TradeAction, market: MarketInfo): string {
  switch (action.type) {
    case "PLACE_ORDER":
      return `PLACE ${action.side} ${tokenLabel(action.assetId, market)} ${action.size}@${action.price.toFixed(2)} (${action.orderType})`;
    case "CANCEL_ORDER": return `CANCEL ${action.orderId}`;
    case "CANCEL_ALL": return "CANCEL_ALL";
    case "NONCE_INVALIDATE": return "NONCE_INVALIDATE";
    case "REDEEM": return `REDEEM ${action.conditionId.slice(0, 12)}...`;
    case "NOOP": return "NOOP";
  }
}

// ── Main ──

async function main(): Promise<void> {
  console.log(`\n=== Paper Trading Harness ===`);
  console.log(`Strategy: ${STRATEGY_ID}`);
  console.log(`Markets to trade: ${MAX_MARKETS}`);
  console.log(`Output: ${OUT_DIR}/`);
  console.log(`No real orders will be placed.\n`);

  const strategy = createStrategy(STRATEGY_ID);
  const bookManager = new OrderBookManager();
  const paperTrader = new PaperTrader();
  const rotation = new MarketRotation();
  const binanceFeed = new BinanceFeed();
  const dataSource = new WsMarketData(fakeConfig);
  const csvLogger = new PaperCsvLogger(OUT_DIR);
  const volTracker = new VolTracker(30_000);

  const externalData: Record<string, ExternalDataPoint> = {};
  let currentMarket: MarketInfo | null = null;
  let marketsTraded = 0;
  let actionCount = 0;
  let tickCount = 0;
  let btcPriceAtOpen: number | null = null;
  let marketDurationMs = 300_000;
  let lastTickLogTime = 0;
  let lastBtcTimestamp = 0;

  // Track per-market data for post-run resolution fetch and CSV
  const tradedSlugs: Array<{
    slug: string;
    conditionId: string;
    upTokenId: string;
    downTokenId: string;
    btcOpen: number | null;
    btcClose: number | null;
    startTime: string;
    endTime: string;
    ticks: number;
    orders: number;
    fills: number;
    cancels: number;
    tradingPnl: number;
    upPosAtClose: number;
    upAvgEntry: number;
    downPosAtClose: number;
    downAvgEntry: number;
  }> = [];

  // Per-market counter snapshots (reset at market open)
  let marketOrdersStart = 0;
  let marketFillsStart = 0;
  let marketCancelsStart = 0;
  let marketPnlStart = 0;

  // ── Binance feed ──
  binanceFeed.on("data", (point) => {
    externalData[point.feed] = point;
    if (point.timestamp > lastBtcTimestamp) {
      volTracker.push(point.timestamp, point.price);
      lastBtcTimestamp = point.timestamp;
    }
  });

  // ── Compute fair value (mirrors strategy logic for CSV logging) ──
  function computeFairValue(): { fairUp: number; fairDown: number; z: number; vol: number; btcReturn: number } | null {
    if (!currentMarket || !btcPriceAtOpen) return null;
    const btc = externalData["binance-btcusdt"];
    const btcNow = btc?.price ?? btcPriceAtOpen;
    const btcReturn = (btcNow - btcPriceAtOpen) / btcPriceAtOpen;
    const timeRemainingMs = Math.max(0, currentMarket.endTime.getTime() - Date.now());
    const timeElapsedMs = Math.max(1, marketDurationMs - timeRemainingMs);
    const timeElapsedFrac = Math.max(0.001, timeElapsedMs / marketDurationMs);
    const vol = volTracker.realizedVol(marketDurationMs);
    if (vol <= 0 || volTracker.tickCount < 50) return null;
    const z = (btcReturn / vol) * Math.sqrt(timeElapsedFrac);
    return { fairUp: normalCdf(z), fairDown: 1 - normalCdf(z), z, vol, btcReturn };
  }

  // ── Build MarketState ──
  function buildState(): MarketState {
    return {
      market: currentMarket!,
      books: bookManager.allSnapshots(),
      positions: paperTrader.getPositions(),
      openOrders: paperTrader.getOpenOrders(),
      timeRemainingMs: Math.max(0, currentMarket!.endTime.getTime() - Date.now()),
      timestamp: Date.now(),
      externalData: Object.keys(externalData).length > 0 ? { ...externalData } : undefined,
    };
  }

  // ── Log a tick-level snapshot (throttled to ~1/sec) ──
  function maybeLogTick(): void {
    if (!currentMarket) return;
    const now = Date.now();
    if (now - lastTickLogTime < 1000) return;
    lastTickLogTime = now;

    const btc = externalData["binance-btcusdt"];
    const fv = computeFairValue();
    const timeRemainingMs = Math.max(0, currentMarket.endTime.getTime() - now);
    const timeElapsedS = (marketDurationMs - timeRemainingMs) / 1000;

    const upBook = bookManager.allSnapshots()[currentMarket.upTokenId];
    const downBook = bookManager.allSnapshots()[currentMarket.downTokenId];

    const upPos = paperTrader.getPosition(currentMarket.upTokenId);
    const downPos = paperTrader.getPosition(currentMarket.downTokenId);

    csvLogger.logTick({
      timestamp: now,
      market_slug: currentMarket.slug,
      time_elapsed_s: timeElapsedS.toFixed(2),
      time_remaining_s: (timeRemainingMs / 1000).toFixed(2),
      btc_price: btc?.price?.toFixed(2) ?? "",
      btc_return_pct: fv ? (fv.btcReturn * 100).toFixed(6) : "",
      realized_vol_pct: fv ? (fv.vol * 100).toFixed(6) : "",
      fair_up: fv?.fairUp.toFixed(6) ?? "",
      fair_down: fv?.fairDown.toFixed(6) ?? "",
      z_score: fv?.z.toFixed(4) ?? "",
      up_best_bid: upBook?.bids?.[0]?.price?.toFixed(3) ?? "",
      up_best_ask: upBook?.asks?.[0]?.price?.toFixed(3) ?? "",
      up_mid: upBook && upBook.bids.length > 0 && upBook.asks.length > 0
        ? ((upBook.bids[0]!.price + upBook.asks[0]!.price) / 2).toFixed(4) : "",
      up_spread: upBook && upBook.bids.length > 0 && upBook.asks.length > 0
        ? (upBook.asks[0]!.price - upBook.bids[0]!.price).toFixed(3) : "",
      down_best_bid: downBook?.bids?.[0]?.price?.toFixed(3) ?? "",
      down_best_ask: downBook?.asks?.[0]?.price?.toFixed(3) ?? "",
      down_mid: downBook && downBook.bids.length > 0 && downBook.asks.length > 0
        ? ((downBook.bids[0]!.price + downBook.asks[0]!.price) / 2).toFixed(4) : "",
      down_spread: downBook && downBook.bids.length > 0 && downBook.asks.length > 0
        ? (downBook.asks[0]!.price - downBook.bids[0]!.price).toFixed(3) : "",
      up_position: upPos?.size ?? 0,
      down_position: downPos?.size ?? 0,
      net_inventory: (upPos?.size ?? 0) - (downPos?.size ?? 0),
      open_order_count: paperTrader.openOrderCount,
      cumulative_fills: paperTrader.fillCount,
      cumulative_pnl: paperTrader.totalPnl.toFixed(4),
    });
  }

  // ── Log a trade event ──
  function logTradeEvent(
    eventType: string,
    orderId: string,
    assetId: string,
    side: string,
    price: number,
    size: number,
    orderType: string = "",
    fillPrice: number | null = null,
    fillSize: number | null = null,
  ): void {
    if (!currentMarket) return;
    const btc = externalData["binance-btcusdt"];
    const fv = computeFairValue();
    const pos = paperTrader.getPosition(assetId);
    const timeRemainingMs = Math.max(0, currentMarket.endTime.getTime() - Date.now());

    csvLogger.logTrade({
      timestamp: Date.now(),
      market_slug: currentMarket.slug,
      event_type: eventType,
      order_id: orderId,
      asset_id: assetId,
      token_label: tokenLabel(assetId, currentMarket),
      side,
      price: price.toFixed(4),
      size: size.toFixed(2),
      order_type: orderType,
      fill_price: fillPrice?.toFixed(4) ?? "",
      fill_size: fillSize?.toFixed(2) ?? "",
      position_after: pos?.size ?? 0,
      avg_entry_after: pos?.avgEntryPrice?.toFixed(4) ?? "",
      realized_pnl: pos?.realizedPnl?.toFixed(4) ?? "0",
      btc_price: btc?.price?.toFixed(2) ?? "",
      fair_value: assetId === currentMarket.upTokenId
        ? (fv?.fairUp?.toFixed(4) ?? "")
        : (fv?.fairDown?.toFixed(4) ?? ""),
      time_remaining_s: (timeRemainingMs / 1000).toFixed(1),
    });
  }

  // ── Dispatch actions (paper) ──
  function dispatchActions(actions: TradeAction[]): void {
    for (const action of actions) {
      if (action.type === "NOOP") continue;
      actionCount++;

      switch (action.type) {
        case "PLACE_ORDER": {
          const orderId = paperTrader.placeOrder(action);
          const label = formatAction(action, currentMarket!);
          console.log(`  [${ts()}] ${label} → ${orderId}`);
          logTradeEvent("PLACE", orderId, action.assetId, action.side, action.price, action.size, action.orderType);
          break;
        }
        case "CANCEL_ORDER": {
          const ok = paperTrader.cancelOrder(action.orderId);
          console.log(`  [${ts()}] CANCEL ${action.orderId} → ${ok ? "ok" : "not found"}`);
          logTradeEvent("CANCEL", action.orderId, "", "", 0, 0);
          break;
        }
        case "CANCEL_ALL": {
          const n = paperTrader.cancelAll();
          console.log(`  [${ts()}] CANCEL_ALL → cancelled ${n}`);
          logTradeEvent("CANCEL_ALL", "", "", "", 0, 0);
          break;
        }
        default:
          console.log(`  [${ts()}] ${formatAction(action, currentMarket!)}`);
      }
    }
  }

  // ── Check for simulated fills ──
  function checkFills(): void {
    if (!currentMarket) return;
    const fills = paperTrader.checkFills(bookManager.allSnapshots());
    for (const fill of fills) {
      const label = tokenLabel(fill.assetId, currentMarket);
      console.log(`  [${ts()}] FILL ${fill.side} ${label} ${fill.size}@${fill.price.toFixed(2)}`);
      logTradeEvent("FILL", fill.orderId, fill.assetId, fill.side, fill.price, fill.size, "", fill.price, fill.size);
    }

    if (fills.length > 0) {
      const positions = paperTrader.getPositions();
      if (positions.length > 0) {
        console.log(
          `  [${ts()}] Positions: ${positions.map((p) => `${tokenLabel(p.assetId, currentMarket!)} ${p.size}@${p.avgEntryPrice.toFixed(3)} pnl=${p.realizedPnl.toFixed(4)}`).join(" | ")}`
        );
      }
    }
  }

  // ── Market data handler ──
  dataSource.on("data", async (event: MarketDataEvent) => {
    switch (event.type) {
      case "book": {
        const book = bookManager.get(event.book.assetId);
        book.applySnapshot(event.book);
        tickCount++;
        break;
      }
      case "price_change": {
        for (const change of event.changes) {
          const book = bookManager.get(change.assetId);
          book.applyPriceChange(change, event.timestamp);
        }
        tickCount++;
        break;
      }
    }

    if (currentMarket && strategy.dataMode === "tick") {
      checkFills();
      const state = buildState();
      const actions = await strategy.evaluate(state);
      dispatchActions(actions);
      checkFills();
      maybeLogTick();
    }
  });

  // ── Market lifecycle ──
  async function openMarket(market: MarketInfo): Promise<void> {
    currentMarket = market;
    bookManager.clear();
    paperTrader.resetForNewMarket();
    volTracker.reset();
    lastTickLogTime = 0;
    tickCount = 0;
    actionCount = 0;

    // Snapshot cumulative counters so we can compute per-market deltas at close
    marketOrdersStart = paperTrader.orderCount;
    marketFillsStart = paperTrader.fillCount;
    marketCancelsStart = paperTrader.cancelCount;
    marketPnlStart = paperTrader.totalPnl;

    const btc = externalData["binance-btcusdt"];
    btcPriceAtOpen = btc?.price ?? null;
    marketDurationMs = Math.max(1, market.endTime.getTime() - Date.now());

    console.log(`\n[${ts()}] === Market Open: ${market.slug} ===`);
    console.log(`  conditionId: ${market.conditionId.slice(0, 20)}...`);
    console.log(`  Ends at: ${market.endTime.toISOString()}`);
    if (btcPriceAtOpen) console.log(`  BTC ref: ${btcPriceAtOpen.toFixed(2)}`);

    tradedSlugs.push({
      slug: market.slug,
      conditionId: market.conditionId,
      upTokenId: market.upTokenId,
      downTokenId: market.downTokenId,
      btcOpen: btcPriceAtOpen,
      btcClose: null,
      startTime: new Date().toISOString(),
      endTime: market.endTime.toISOString(),
      ticks: 0, orders: 0, fills: 0, cancels: 0, tradingPnl: 0,
      upPosAtClose: 0, upAvgEntry: 0, downPosAtClose: 0, downAvgEntry: 0,
    });

    await dataSource.subscribe(market);
    const state = buildState();
    await strategy.onMarketOpen(state);
  }

  async function closeCurrentMarket(): Promise<void> {
    if (!currentMarket) return;

    const state = buildState();
    const actions = await strategy.onMarketClose(state);
    dispatchActions(actions);

    const btc = externalData["binance-btcusdt"];
    const btcClose = btc?.price ?? null;
    const positions = paperTrader.getPositions();
    const upPos = paperTrader.getPosition(currentMarket.upTokenId);
    const downPos = paperTrader.getPosition(currentMarket.downTokenId);

    // Compute per-market deltas
    const mktOrders = paperTrader.orderCount - marketOrdersStart;
    const mktFills = paperTrader.fillCount - marketFillsStart;
    const mktCancels = paperTrader.cancelCount - marketCancelsStart;
    const mktPnl = paperTrader.totalPnl - marketPnlStart;

    // Update traded slug with close data
    const slugEntry = tradedSlugs[tradedSlugs.length - 1];
    if (slugEntry) {
      slugEntry.btcClose = btcClose;
      // Backfill btcOpen if it was null at open (Binance tick arrived after open)
      if (slugEntry.btcOpen === null) slugEntry.btcOpen = btcPriceAtOpen;
      slugEntry.ticks = tickCount;
      slugEntry.orders = mktOrders;
      slugEntry.fills = mktFills;
      slugEntry.cancels = mktCancels;
      slugEntry.tradingPnl = mktPnl;
      slugEntry.upPosAtClose = upPos?.size ?? 0;
      slugEntry.upAvgEntry = upPos?.avgEntryPrice ?? 0;
      slugEntry.downPosAtClose = downPos?.size ?? 0;
      slugEntry.downAvgEntry = downPos?.avgEntryPrice ?? 0;
    }

    if (positions.length > 0) {
      console.log(
        `  [${ts()}] Positions at close: ${positions.map((p) => `${tokenLabel(p.assetId, currentMarket!)} ${p.size}@${p.avgEntryPrice.toFixed(3)}`).join(" | ")}`
      );
    }

    console.log(`\n[${ts()}] === Market Close: ${currentMarket.slug} ===`);
    console.log(`  Ticks: ${tickCount} | Orders: ${mktOrders} | Fills: ${mktFills} | Cancels: ${mktCancels}`);
    console.log(`  Market PnL: ${mktPnl.toFixed(4)} USDC (cumulative: ${paperTrader.totalPnl.toFixed(4)})`);

    marketsTraded++;
    currentMarket = null;
  }

  function scheduleMarketClose(market: MarketInfo): void {
    const remaining = market.endTime.getTime() - Date.now();
    const delay = Math.max(0, remaining);
    console.log(`  Closes in ${(delay / 1000).toFixed(1)}s`);

    setTimeout(async () => {
      await closeCurrentMarket();

      if (marketsTraded >= MAX_MARKETS) {
        await finalize();
        process.exit(0);
      }

      await new Promise((r) => setTimeout(r, 1_000));
      await fetchAndOpenNextMarket();
    }, delay);
  }

  async function fetchAndOpenNextMarket(): Promise<void> {
    for (let attempt = 0; attempt < 5; attempt++) {
      const market = await rotation.fetchCurrentMarket().catch(() => null);
      if (market) {
        await openMarket(market);
        scheduleMarketClose(market);
        return;
      }
      console.log(`  [${ts()}] Next market not yet available (attempt ${attempt + 1}/5)`);
      await new Promise((r) => setTimeout(r, 2_000));
    }
    console.log(`  [${ts()}] Failed to fetch next market, waiting for next boundary...`);
    const nowMs = Date.now();
    const nextBoundaryMs = Math.ceil(nowMs / 300_000) * 300_000;
    setTimeout(() => fetchAndOpenNextMarket(), nextBoundaryMs - nowMs);
  }

  async function finalize(): Promise<void> {
    console.log(`\n=== Fetching resolutions for ${tradedSlugs.length} markets... ===`);

    for (const entry of tradedSlugs) {
      // Wait a bit for resolution (markets resolve shortly after close)
      let resolution: MarketResolution | null = null;
      for (let attempt = 0; attempt < 3; attempt++) {
        resolution = await fetchResolution(entry.slug);
        if (resolution) break;
        await new Promise((r) => setTimeout(r, 5_000));
      }

      const btcReturn = entry.btcOpen && entry.btcClose
        ? ((entry.btcClose - entry.btcOpen) / entry.btcOpen * 100) : null;

      csvLogger.logMarket({
        market_slug: entry.slug,
        condition_id: entry.conditionId,
        up_token_id: entry.upTokenId,
        down_token_id: entry.downTokenId,
        start_time: entry.startTime,
        end_time: entry.endTime,
        btc_open: entry.btcOpen?.toFixed(2) ?? "",
        btc_close: entry.btcClose?.toFixed(2) ?? "",
        btc_return_pct: btcReturn?.toFixed(6) ?? "",
        total_ticks: entry.ticks,
        total_orders: entry.orders,
        total_fills: entry.fills,
        total_cancels: entry.cancels,
        realized_pnl_trading: entry.tradingPnl.toFixed(4),
        up_position_at_close: entry.upPosAtClose.toFixed(1),
        up_avg_entry: entry.upAvgEntry > 0 ? entry.upAvgEntry.toFixed(4) : "",
        down_position_at_close: entry.downPosAtClose.toFixed(1),
        down_avg_entry: entry.downAvgEntry > 0 ? entry.downAvgEntry.toFixed(4) : "",
        resolution_winner: resolution ? resolution.winnerLabel : "pending",
        resolution_pnl: "",
        total_pnl: entry.tradingPnl.toFixed(4),
      });

      if (resolution) {
        console.log(`  ${entry.slug}: winner=${resolution.winnerLabel}`);
      } else {
        console.log(`  ${entry.slug}: resolution pending`);
      }
    }

    console.log(`\n=== Paper trading complete (${marketsTraded} markets) ===`);
    console.log(`Total trading PnL: ${paperTrader.totalPnl.toFixed(4)} USDC`);
    console.log(`Orders: ${paperTrader.orderCount} | Fills: ${paperTrader.fillCount} | Cancels: ${paperTrader.cancelCount}`);
    console.log(`\nCSV files written to:`);
    const paths = csvLogger.paths;
    console.log(`  Ticks:   ${paths.ticks}`);
    console.log(`  Trades:  ${paths.trades}`);
    console.log(`  Markets: ${paths.markets}`);

    await dataSource.stop();
    await binanceFeed.stop();
  }

  // ── Graceful shutdown ──
  process.on("SIGINT", async () => {
    console.log("\n\nShutting down...");
    if (currentMarket) await closeCurrentMarket();
    await finalize();
    process.exit(0);
  });

  // ── Start ──
  console.log("Connecting to Binance BTC/USDT feed...");
  await binanceFeed.start();
  console.log("Binance feed connected.");

  console.log("Connecting to Polymarket WebSocket...");
  await dataSource.start();
  console.log("Polymarket WS connected.");

  console.log("Fetching current market...\n");
  const market = await rotation.fetchCurrentMarket();
  if (market) {
    await openMarket(market);
    scheduleMarketClose(market);
  } else {
    console.log("No active market found, waiting for next window...");
    await fetchAndOpenNextMarket();
  }
}

main().catch((err) => {
  console.error("Fatal error:", err);
  process.exit(1);
});
