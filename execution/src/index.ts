import { loadConfig } from "./config.ts";
import { createMarketDataSource, type MarketDataSource } from "./data/market-data-source.ts";
import { OrderBookManager } from "./data/orderbook.ts";
import { Resampler } from "./data/resampler.ts";
import { IndicatorEngine } from "./data/indicators.ts";
import { initProto } from "./data/proto.ts";
import { createExternalFeed, type ExternalFeed } from "./data/external-feed.ts";
import { OrderManager } from "./orders/order-manager.ts";
import { NonceCanceller } from "./orders/nonce-cancel.ts";
import { PositionStore } from "./positions/position-store.ts";
import { Redeemer } from "./positions/redemption.ts";
import { Minter } from "./positions/minter.ts";
import { UserChannel } from "./positions/user-channel.ts";
import { MarketRotation } from "./rotation/market-rotation.ts";
import { CsvLogger } from "./logging/csv-logger.ts";
import { createStrategy, type Strategy } from "./strategy/strategy.ts";
import { setupShutdown } from "./shutdown.ts";
import { DashboardServer } from "./dashboard/server.ts";
import type {
  AppConfig,
  MarketInfo,
  MarketState,
  MarketDataEvent,
  TradeAction,
  ExternalDataPoint,
  UserChannelEvent,
} from "./types.ts";

// Register built-in strategies (side-effect imports)
import "./strategy/market-maker.ts";
import "./strategy/signal-receiver.ts";
import "./strategy/smoke-test.ts";
import "./strategy/mid-drift.ts";
import "./strategy/mid-drift-mint.ts";
import "./strategy/xgb-confidence.ts";
import "./strategy/entropy-binance.ts";

async function main(): Promise<void> {
  // ── Config ──
  const config = loadConfig();
  console.log(`[Main] Asset: ${config.execution.asset}`);
  console.log(`[Main] Data source: ${config.execution.dataSource}`);
  console.log(`[Main] Strategy: ${config.execution.strategyId}`);

  // ── Init proto (needed for Redis data source) ──
  if (config.execution.dataSource === "redis") {
    await initProto();
    console.log("[Main] Protobuf initialized");
  }

  // ── Create components ──
  const dataSource = createMarketDataSource(config);
  const bookManager = new OrderBookManager();
  const resampler = new Resampler(config.execution.resampleIntervalMs);
  const indicators = new IndicatorEngine();
  const positionStore = new PositionStore();
  const orderManager = new OrderManager(config);
  const nonceCanceller = new NonceCanceller(config);
  const redeemer = new Redeemer(config);
  const minter = new Minter(config);
  const csvLogger = new CsvLogger(config.execution.metricsDir);
  const rotation = new MarketRotation(config.execution.asset, config.execution.marketDuration);
  const strategy = createStrategy(config.execution.strategyId);

  // ── User channel ──
  const userChannel = new UserChannel({
    apiKey: config.polymarket.apiKey,
    apiSecret: config.polymarket.apiSecret,
    apiPassphrase: config.polymarket.apiPassphrase,
  });

  // ── External feeds ──
  const externalFeeds: ExternalFeed[] = [];
  const externalData: Record<string, ExternalDataPoint> = {};

  for (const feedName of config.execution.externalFeeds) {
    const feed = createExternalFeed(feedName);
    feed.on("data", (point) => {
      externalData[point.feed] = point;
    });
    externalFeeds.push(feed);
  }

  // ── Dashboard ──
  let currentMarket: MarketInfo | null = null;
  const dashboard = new DashboardServer(
    config.execution.dashboardPort ?? 3456,
    {
      bookManager,
      positionStore,
      orderManager,
      minter,
      externalData,
      getCurrentMarket: () => currentMarket,
      getStrategy: () => strategy,
      config,
    }
  );
  dashboard.start();

  // ── State ──
  let resampleTimer: ReturnType<typeof setInterval> | null = null;
  let reconcileTimer: ReturnType<typeof setInterval> | null = null;
  // Track processed trade IDs to deduplicate fills.
  // Trade events arrive multiple times (MATCHED → MINED → CONFIRMED), but sometimes
  // MATCHED is skipped. Process on first sight of any status, ignore subsequent.
  const processedTradeIds = new Set<string>();

  // ── Periodic order reconciliation (remove phantom orders) ──
  reconcileTimer = setInterval(async () => {
    const localOrders = positionStore.getOpenOrders();
    if (localOrders.length === 0) return;
    try {
      const clobOrders = await orderManager.getOpenOrders();
      const clobIds = new Set(clobOrders.map((o: any) => o.id));
      const removed = positionStore.reconcileOrders(clobIds);
      if (removed > 0) {
        console.log(`[Reconcile] Removed ${removed} phantom order(s)`);
      }
    } catch (err) {
      // Non-fatal — just skip this cycle
    }
  }, 30_000);

  // ── Wire market data events ──
  dataSource.on("data", async (event: MarketDataEvent) => {
    switch (event.type) {
      case "book": {
        const book = bookManager.get(event.book.assetId);
        book.applySnapshot(event.book);
        resampler.tick(event.book.assetId, book);

        // Tick-mode strategies get called on every event
        if (strategy.dataMode === "tick" && currentMarket && !dashboard.killed) {
          const state = buildMarketState(
            currentMarket,
            bookManager,
            positionStore,
            externalData
          );
          await dispatchActions(
            await strategy.evaluate(state),
            orderManager,
            nonceCanceller,
            redeemer,
            minter,
            positionStore,
            csvLogger,
            currentMarket,
            strategy
          );
        }
        break;
      }

      case "price_change": {
        for (const change of event.changes) {
          const book = bookManager.get(change.assetId);
          book.applyPriceChange(change, event.timestamp);
          resampler.tick(change.assetId, book);
        }

        if (strategy.dataMode === "tick" && currentMarket && !dashboard.killed) {
          const state = buildMarketState(
            currentMarket,
            bookManager,
            positionStore,
            externalData
          );
          await dispatchActions(
            await strategy.evaluate(state),
            orderManager,
            nonceCanceller,
            redeemer,
            minter,
            positionStore,
            csvLogger,
            currentMarket,
            strategy
          );
        }
        break;
      }

      case "market_open": {
        console.log(
          `[Main] Market opened: ${event.market.slug} (${event.market.conditionId})`
        );
        currentMarket = event.market;

        // Reset per-market state
        bookManager.clear();
        resampler.reset();
        indicators.reset();
        positionStore.reset();

        // Subscribe user channel to this market
        userChannel.subscribeMarket(event.market.conditionId);

        // Subscribe data source (WS mode needs explicit subscription)
        await dataSource.subscribe(event.market);

        // Notify strategy
        const state = buildMarketState(
          currentMarket,
          bookManager,
          positionStore,
          externalData
        );
        await strategy.onMarketOpen(state);

        // Start resample timer for resampled-mode strategies
        if (strategy.dataMode === "resampled") {
          startResampleTimer();
        }
        break;
      }

      case "market_close": {
        console.log(`[Main] Market closed: ${event.conditionId}`);

        if (currentMarket) {
          stopResampleTimer();

          const state = buildMarketState(
            currentMarket,
            bookManager,
            positionStore,
            externalData
          );
          await dispatchActions(
            await strategy.onMarketClose(state),
            orderManager,
            nonceCanceller,
            redeemer,
            minter,
            positionStore,
            csvLogger,
            currentMarket,
            strategy
          );

          userChannel.unsubscribeMarket(event.conditionId);
          currentMarket = null;
        }
        break;
      }
    }
  });

  // ── Wire user channel events ──
  userChannel.on("data", (event: UserChannelEvent) => {
    if (event.eventType === "order") {
      const isOurs = positionStore.isOurOrder(event.id);
      if (isOurs) {
        console.log(
          `[Order] ${event.orderEventType} ${event.id.slice(0, 12)}… ${event.side} ${event.originalSize}@${event.price} (status: ${event.status}, matched: ${event.sizeMatched})`
        );
      }
      positionStore.applyOrderEvent(event);

      if (isOurs) {
        csvLogger.logOrder({
          timestamp: event.timestamp,
          marketSlug: currentMarket?.slug ?? "",
          orderId: event.id,
          assetId: event.assetId,
          side: event.side,
          price: event.price,
          size: event.originalSize,
          orderType: event.orderType,
          status: event.status,
        });
      }
    } else if (event.eventType === "trade") {
      // Check if this trade involves our orders before processing
      const isTaker = positionStore.isOurOrder(event.takerOrderId);
      const makerMatches = event.makerOrders.filter((mo) => positionStore.isOurOrder(mo.orderId));
      const isOurTrade = isTaker || makerMatches.length > 0;

      // Diagnostic: dump raw trade event details for our trades
      if (isOurTrade) {
        console.log(
          `[TradeEvent] id=${event.id.slice(0, 12)}… status=${event.status} traderSide=${event.traderSide} ` +
          `side=${event.side} size=${event.size} price=${event.price} ` +
          `takerOrderId=${event.takerOrderId.slice(0, 12)}… (ours: ${isTaker}) ` +
          `makerOrders=[${event.makerOrders.map((mo) =>
            `{id=${mo.orderId.slice(0, 12)}… ours=${positionStore.isOurOrder(mo.orderId)} side=${mo.side} amt=${mo.matchedAmount} price=${mo.price} asset=${mo.assetId.slice(0, 12)}…}`
          ).join(", ")}]`
        );
      }

      positionStore.applyFill(event);

      if (isOurTrade) {
        // Log position state after fill
        const snap = positionStore.snapshot();
        console.log(
          `[Positions] ${snap.positions.map((p) => `${p.assetId.slice(0, 12)}… ${p.side} ${p.size}@${p.avgEntryPrice.toFixed(4)} pnl=${p.realizedPnl.toFixed(4)} settled=${p.settled}`).join(" | ") || "(none)"}`
        );
        console.log(
          `[OpenOrders] ${snap.openOrders.map((o) => `${o.orderId.slice(0, 12)}… ${o.side} rem=${o.remainingSize}/${o.originalSize}@${o.price}`).join(" | ") || "(none)"}`
        );

        // Deduplicate: process each trade ID only once. Polymarket sends the same
        // trade ID with MATCHED → MINED → CONFIRMED, but sometimes MATCHED is
        // skipped (trade arrives first as MINED or CONFIRMED). Process on first
        // sight regardless of status, then ignore subsequent updates.
        const isFirstFill = !processedTradeIds.has(event.id);
        if (isFirstFill) {
          processedTradeIds.add(event.id);
        }

        // Log fill with correct details depending on our role.
        // When we're a maker, use our maker fill details (price, size, orderId),
        // not the top-level taker fields which belong to the counterparty.
        if (isTaker) {
          if (isFirstFill) {
            csvLogger.logFill({
              timestamp: event.timestamp,
              marketSlug: currentMarket?.slug ?? "",
              orderId: event.takerOrderId,
              assetId: event.assetId,
              side: event.side,
              price: event.price,
              size: event.size,
              feeRateBps: event.feeRateBps,
              fillLatencyMs: Date.now() - event.timestamp,
            });
            if (strategy.onFill) {
              strategy.onFill(event.assetId, event.side, event.size, event.price);
            }
          }
        }
        for (const mo of makerMatches) {
          if (isFirstFill) {
            csvLogger.logFill({
              timestamp: event.timestamp,
              marketSlug: currentMarket?.slug ?? "",
              orderId: mo.orderId,
              assetId: mo.assetId || event.assetId,
              side: mo.side,
              price: mo.price,
              size: mo.matchedAmount,
              feeRateBps: mo.feeRateBps,
              fillLatencyMs: Date.now() - event.timestamp,
            });
            if (strategy.onFill) {
              strategy.onFill(mo.assetId || event.assetId, mo.side, mo.matchedAmount, mo.price);
            }
          }
        }
      }
    }
  });

  // ── Resample timer helpers ──
  function startResampleTimer(): void {
    stopResampleTimer();
    resampleTimer = setInterval(() => {
      const bars = resampler.emit();
      for (const bar of bars) {
        indicators.update(bar);
      }

      if (currentMarket && !dashboard.killed) {
        const state = buildMarketState(
          currentMarket,
          bookManager,
          positionStore,
          externalData,
          bars,
          indicators.snapshot()
        );
        strategy.evaluate(state).then((actions) =>
          dispatchActions(
            actions,
            orderManager,
            nonceCanceller,
            redeemer,
            minter,
            positionStore,
            csvLogger,
            currentMarket!,
            strategy
          )
        );
      }
    }, config.execution.resampleIntervalMs);
  }

  function stopResampleTimer(): void {
    if (resampleTimer) {
      clearInterval(resampleTimer);
      resampleTimer = null;
    }
  }

  // ── Graceful shutdown ──
  setupShutdown({
    orderManager,
    nonceCanceller,
    csvLogger,
    cleanup: async () => {
      stopResampleTimer();
      if (reconcileTimer) clearInterval(reconcileTimer);
      dashboard.stop();
      redeemer.stopSweep();
      await dataSource.stop();
      await userChannel.stop();
      for (const feed of externalFeeds) {
        await feed.stop();
      }
    },
  });

  // ── Initialize order manager (derives API creds) ──
  await orderManager.init();

  // ── Pass derived creds to user channel ──
  const derivedCreds = orderManager.getDerivedCreds();
  userChannel.setCredentials({
    apiKey: derivedCreds.key,
    apiSecret: derivedCreds.secret,
    apiPassphrase: derivedCreds.passphrase,
  });

  // ── Start connections ──
  await userChannel.start();
  await dataSource.start();
  for (const feed of externalFeeds) {
    await feed.start();
  }

  // ── Start redemption sweep (with trade log backfill) ──
  redeemer.setCsvLogger(csvLogger);
  redeemer.startSweep();

  // ── In WebSocket mode, use timestamp-based market rotation ──
  if (config.execution.dataSource === "websocket") {

    async function openMarket(market: MarketInfo): Promise<void> {
      console.log(`[Main] Opening market: ${market.slug} (${market.conditionId})`);
      currentMarket = market;
      bookManager.clear();
      resampler.reset();
      indicators.reset();
      positionStore.reset();
      processedTradeIds.clear();

      const state = buildMarketState(
        market,
        bookManager,
        positionStore,
        externalData
      );
      await strategy.onMarketOpen(state);

      // Sync inventory AFTER onMarketOpen (which resets state), but BEFORE
      // subscribing to data (which triggers evaluate calls)
      if (strategy.onInventorySync) {
        try {
          const [upBalance, downBalance, clobOrders] = await Promise.all([
            minter.getTokenBalance(market.upTokenId),
            minter.getTokenBalance(market.downTokenId),
            orderManager.getOpenOrders(),
          ]);
          const upTokens = Number(upBalance) / 1_000_000;
          const downTokens = Number(downBalance) / 1_000_000;

          // Seed position store with existing CLOB orders
          for (const o of clobOrders) {
            const remaining = Number(o.original_size) - Number(o.size_matched);
            if (remaining > 0) {
              positionStore.addOrder({
                orderId: o.id,
                assetId: o.asset_id,
                side: o.side as import("./types.ts").Side,
                price: Number(o.price),
                originalSize: Number(o.original_size),
                remainingSize: remaining,
                placedAt: Date.now(),
              });
            }
          }

          strategy.onInventorySync(upTokens, downTokens, clobOrders.map((o: any) => ({
            assetId: o.asset_id,
            side: o.side,
            price: Number(o.price),
            size: Number(o.original_size) - Number(o.size_matched),
          })));
        } catch (err) {
          console.error("[Main] Inventory sync failed, starting fresh:", err);
        }
      }

      // Subscribe to data AFTER inventory sync to prevent
      // evaluate() firing before strategy state is fully initialized
      userChannel.subscribeMarket(market.conditionId);
      await dataSource.subscribe(market);

      if (strategy.dataMode === "resampled") {
        startResampleTimer();
      }
      // Schedule close at market end time
      scheduleMarketClose(market);
    }

    async function closeCurrentMarket(): Promise<void> {
      if (!currentMarket) return;
      const closedConditionId = currentMarket.conditionId;
      console.log(`[Main] Market ${currentMarket.slug} window closed`);
      stopResampleTimer();
      const state = buildMarketState(
        currentMarket,
        bookManager,
        positionStore,
        externalData
      );
      await dispatchActions(
        await strategy.onMarketClose(state),
        orderManager,
        nonceCanceller,
        redeemer,
        minter,
        positionStore,
        csvLogger,
        currentMarket,
        strategy
      );
      userChannel.unsubscribeMarket(closedConditionId);
      // Only queue for redemption if we actually held positions in this market
      const { positions } = positionStore.snapshot();
      if (positions.length > 0) {
        redeemer.queueForRedemption(closedConditionId);
      }
      currentMarket = null;
    }

    function scheduleMarketClose(market: MarketInfo): void {
      const remaining = market.endTime.getTime() - Date.now();
      const delay = Math.max(0, remaining);
      console.log(`[Main] Market closes in ${(delay / 1000).toFixed(1)}s`);
      setTimeout(async () => {
        await closeCurrentMarket();
        // Immediately fetch and open the next market
        await fetchAndOpenNextMarket();
      }, delay);
    }

    async function fetchAndOpenNextMarket(): Promise<void> {
      // Small delay to let Gamma API catch up with the new market
      await new Promise((r) => setTimeout(r, 1_000));
      for (let attempt = 0; attempt < 5; attempt++) {
        const market = await rotation.fetchCurrentMarket().catch(() => null);
        if (market) {
          await openMarket(market);
          return;
        }
        console.log(`[Main] Next market not yet available (attempt ${attempt + 1}/5)`);
        await new Promise((r) => setTimeout(r, 2_000));
      }
      console.log("[Main] Failed to fetch next market after 5 attempts, waiting for next cadence...");
      // Fall back: schedule a retry at the next 300s boundary
      const nowMs = Date.now();
      const nextBoundaryMs = Math.ceil(nowMs / rotation.cadenceMs) * rotation.cadenceMs;
      setTimeout(() => fetchAndOpenNextMarket(), nextBoundaryMs - nowMs);
    }

    // ── Initial market fetch ──
    console.log("[Main] Polling Gamma API for current market...");
    const market = await rotation.fetchCurrentMarket();
    if (market) {
      await openMarket(market);
    } else {
      console.log("[Main] No active market found, waiting for next window...");
      await fetchAndOpenNextMarket();
    }
  }

  console.log("[Main] Running. Press Ctrl+C to stop.");
}

// ── Helpers ──

function buildMarketState(
  market: MarketInfo,
  bookManager: OrderBookManager,
  positionStore: PositionStore,
  externalData: Record<string, ExternalDataPoint>,
  bars?: import("./types.ts").ResampledBar[],
  indicatorSnapshot?: import("./types.ts").IndicatorSnapshot
): MarketState {
  const { positions, openOrders } = positionStore.snapshot();
  return {
    market,
    books: bookManager.allSnapshots(),
    positions,
    openOrders,
    timeRemainingMs: Math.max(0, market.endTime.getTime() - Date.now()),
    timestamp: Date.now(),
    bars,
    indicators: indicatorSnapshot,
    externalData:
      Object.keys(externalData).length > 0 ? { ...externalData } : undefined,
  };
}

async function dispatchActions(
  actions: TradeAction[],
  orderManager: OrderManager,
  nonceCanceller: NonceCanceller,
  redeemer: Redeemer,
  minter: Minter,
  positionStore: PositionStore,
  csvLogger: CsvLogger,
  market: MarketInfo,
  strategyRef?: Strategy
): Promise<void> {
  for (const action of actions) {
    switch (action.type) {
      case "PLACE_ORDER": {
        const result = await orderManager.placeOrder(action);
        if (result.success && result.orderId) {
          positionStore.addOrder({
            orderId: result.orderId,
            assetId: action.assetId,
            side: action.side,
            price: action.price,
            originalSize: action.size,
            remainingSize: action.size,
            placedAt: Date.now(),
          });

          // Log trade with model metadata if present
          if (action.metadata) {
            const m = action.metadata;
            csvLogger.logTrade({
              timestamp: Date.now(),
              marketSlug: market.slug,
              conditionId: market.conditionId,
              assetId: action.assetId,
              tokenLabel: m.leaderLabel,
              side: action.side,
              price: action.price,
              size: action.size,
              cost: action.price * action.size,
              prediction: m.prediction,
              confidenceThreshold: m.confidenceThreshold,
              btcVol5m: m.btcVol5m,
              volThreshold: m.volThreshold,
              volPass: m.volPass,
              strategyId: m.strategyId,
            });
          }
        }
        break;
      }
      case "CANCEL_ORDER":
        await orderManager.cancelOrder(action);
        positionStore.removeOrder(action.orderId);
        break;
      case "CANCEL_ALL":
        await orderManager.cancelAll();
        break;
      case "NONCE_INVALIDATE":
        try {
          await nonceCanceller.incrementNonce();
        } catch (err) {
          console.error("[Main] Nonce invalidation failed:", err);
        }
        break;
      case "REDEEM":
        try {
          await redeemer.redeem(action.conditionId);
        } catch (err) {
          console.error("[Main] Redemption failed:", err);
        }
        break;
      case "SPLIT": {
        // Fire-and-forget: don't block evaluate loop while mint tx confirms on-chain.
        // Inventory is only credited in onMintComplete() after confirmation.
        const splitConditionId = action.conditionId;
        const splitAmount = action.amount;
        const splitPairs = Number(splitAmount / 1_000_000n);
        minter.split(splitConditionId, splitAmount).then((splitResult) => {
          if (strategyRef?.onMintComplete) {
            strategyRef.onMintComplete(splitConditionId, splitPairs, splitResult.success);
          }
        }).catch((err) => {
          console.error("[Main] Split failed:", err);
          if (strategyRef?.onMintComplete) {
            strategyRef.onMintComplete(splitConditionId, splitPairs, false);
          }
        });
        break;
      }
      case "MERGE_PAIRS":
        try {
          await minter.merge(action.conditionId, action.amount);
        } catch (err) {
          console.error("[Main] Merge failed:", err);
        }
        break;
      case "NOOP":
        break;
    }
  }
}

// ── Run ──
main().catch((err) => {
  console.error("[Main] Fatal error:", err);
  process.exit(1);
});
