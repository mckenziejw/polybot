import type { OrderManager } from "./orders/order-manager.ts";
import type { NonceCanceller } from "./orders/nonce-cancel.ts";
import type { CsvLogger } from "./logging/csv-logger.ts";

export interface ShutdownDeps {
  orderManager: OrderManager;
  nonceCanceller: NonceCanceller;
  csvLogger: CsvLogger;
  /** Additional cleanup callbacks (stop data sources, user channel, etc.) */
  cleanup: () => Promise<void>;
}

/**
 * Graceful shutdown handler.
 *
 * Sequence:
 * 1. Cancel all orders via CLOB API (soft cancel)
 * 2. If soft cancel fails, invoke on-chain nonce invalidation (nuclear option)
 * 3. Flush CSV logger
 * 4. Run additional cleanup (stop WS connections, timers, etc.)
 */
export function setupShutdown(deps: ShutdownDeps): void {
  let shutdownInProgress = false;

  const handler = async (signal: string) => {
    if (shutdownInProgress) {
      console.log(`[Shutdown] Already in progress, ignoring ${signal}`);
      return;
    }
    shutdownInProgress = true;
    console.log(`\n[Shutdown] Received ${signal}, shutting down...`);

    // Step 1: Try soft cancel
    let softCancelOk = false;
    try {
      softCancelOk = await deps.orderManager.cancelAll();
      if (softCancelOk) {
        console.log("[Shutdown] All orders cancelled via CLOB API");
      }
    } catch (err) {
      console.error("[Shutdown] Soft cancel threw:", err);
    }

    // Step 2: If soft cancel failed, use nonce invalidation
    if (!softCancelOk) {
      console.log("[Shutdown] Soft cancel failed, invoking nonce invalidation...");
      try {
        const receipt = await deps.nonceCanceller.incrementNonce();
        console.log(
          `[Shutdown] Nonce invalidated in block ${receipt.blockNumber}`
        );
      } catch (err) {
        console.error("[Shutdown] Nonce invalidation also failed:", err);
      }
    }

    // Step 3: Flush logs
    try {
      deps.csvLogger.stop();
      console.log("[Shutdown] Logs flushed");
    } catch (err) {
      console.error("[Shutdown] Log flush error:", err);
    }

    // Step 4: Cleanup (stop connections, timers)
    try {
      await deps.cleanup();
      console.log("[Shutdown] Cleanup complete");
    } catch (err) {
      console.error("[Shutdown] Cleanup error:", err);
    }

    process.exit(0);
  };

  process.on("SIGINT", () => handler("SIGINT"));
  process.on("SIGTERM", () => handler("SIGTERM"));
}
