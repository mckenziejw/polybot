import type { Position, OpenOrder, UserOrderEvent, UserTradeEvent } from "../types.ts";

export class PositionStore {
  private positions = new Map<string, Position>();
  private openOrders = new Map<string, OpenOrder>();
  /** All order IDs we've placed — survives order completion/cancellation. */
  private knownOrderIds = new Set<string>();
  /** Trade IDs we've already processed — prevents double-counting on status updates. */
  private processedTradeIds = new Set<string>();
  /** Maps trade ID → assetId(s) affected, for trades awaiting CONFIRMED status. */
  private pendingSettlement = new Map<string, string[]>();

  // ── Public write API ─────────────────────────────────────────────────────

  /** Add an order we just placed (optimistic tracking). */
  addOrder(order: OpenOrder): void {
    this.openOrders.set(order.orderId, { ...order });
    this.knownOrderIds.add(order.orderId);
  }

  /**
   * Apply a fill from the user channel trade event.
   *
   * The user channel broadcasts ALL trade events on subscribed markets,
   * not just ours. We check both paths independently:
   *   1. Is our order the taker? (takerOrderId in knownOrderIds)
   *   2. Are any of our orders in makerOrders?
   *
   * We do NOT rely on `traderSide` for branching — when someone else's
   * market order fills our resting limit, the event may arrive with
   * traderSide="TAKER" (from the seller's perspective), which would
   * skip the maker branch entirely if we branched on it.
   *
   * Each trade event is processed only once (by trade ID) to prevent
   * double-counting across MATCHED → MINED → CONFIRMED status updates.
   */
  applyFill(event: UserTradeEvent): void {
    // Handle settlement confirmations for trades we already processed
    if (this.processedTradeIds.has(event.id)) {
      if (event.status === "CONFIRMED") {
        const assetIds = this.pendingSettlement.get(event.id);
        if (assetIds) {
          for (const assetId of assetIds) {
            const pos = this.positions.get(assetId);
            if (pos) pos.settled = true;
          }
          this.pendingSettlement.delete(event.id);
        }
      }
      return;
    }

    // Accept MATCHED, MINED, or CONFIRMED — trades can skip MATCHED and arrive
    // first as MINED. RETRYING/FAILED are not fills.
    if (event.status !== "MATCHED" && event.status !== "MINED" && event.status !== "CONFIRMED") return;

    const isAlreadyConfirmed = event.status === "CONFIRMED";
    const affectedAssets: string[] = [];

    // Path 1: Check if we are the taker
    const isTaker = this.knownOrderIds.has(event.takerOrderId);
    if (isTaker) {
      const fillSize = Number(event.size);
      const fillPrice = Number(event.price);
      this.updatePosition(event.assetId, event.side, fillSize, fillPrice, isAlreadyConfirmed);
      affectedAssets.push(event.assetId);

      // Reduce our taker order's remaining size
      const order = this.openOrders.get(event.takerOrderId);
      if (order) {
        order.remainingSize = Math.max(0, order.remainingSize - fillSize);
        if (order.remainingSize < 0.01) {
          this.openOrders.delete(event.takerOrderId);
        }
      }
    }

    // Path 2: Check if any of our orders are in makerOrders
    // (independent of path 1 — a trade could theoretically involve us on both sides,
    // though in practice it won't)
    const ourMakerFills = event.makerOrders.filter(
      (mo) => this.knownOrderIds.has(mo.orderId)
    );

    for (const makerFill of ourMakerFills) {
      const fillSize = Number(makerFill.matchedAmount);
      const fillPrice = Number(makerFill.price);
      const fillAssetId = makerFill.assetId || event.assetId;
      const fillSide = makerFill.side;

      this.updatePosition(fillAssetId, fillSide, fillSize, fillPrice, isAlreadyConfirmed);
      affectedAssets.push(fillAssetId);

      // Reduce our maker order's remaining size
      const order = this.openOrders.get(makerFill.orderId);
      if (order) {
        order.remainingSize = Math.max(0, order.remainingSize - fillSize);
        if (order.remainingSize < 0.01) {
          this.openOrders.delete(makerFill.orderId);
        }
      }
    }

    // Only mark as processed if this trade actually involved us
    if (isTaker || ourMakerFills.length > 0) {
      this.processedTradeIds.add(event.id);

      // Track assets awaiting on-chain settlement
      if (!isAlreadyConfirmed && affectedAssets.length > 0) {
        this.pendingSettlement.set(event.id, affectedAssets);
      }
    }
  }

  /** Update a position with a fill. */
  private updatePosition(
    assetId: string,
    side: import("../types.ts").Side,
    fillSize: number,
    fillPrice: number,
    settled: boolean
  ): void {
    let pos = this.positions.get(assetId);
    if (!pos) {
      pos = {
        assetId,
        side,
        size: 0,
        avgEntryPrice: 0,
        realizedPnl: 0,
        settled,
      };
      this.positions.set(assetId, pos);
    }
    // If this fill is already confirmed, mark position settled
    if (settled) pos.settled = true;

    if (pos.size === 0) {
      pos.side = side;
      pos.size = fillSize;
      pos.avgEntryPrice = fillPrice;
    } else if (pos.side === side) {
      const totalSize = pos.size + fillSize;
      pos.avgEntryPrice =
        (pos.avgEntryPrice * pos.size + fillPrice * fillSize) / totalSize;
      pos.size = totalSize;
    } else {
      const pnlPerUnit =
        pos.side === "BUY"
          ? fillPrice - pos.avgEntryPrice
          : pos.avgEntryPrice - fillPrice;

      const closedSize = Math.min(pos.size, fillSize);
      pos.realizedPnl += pnlPerUnit * closedSize;
      pos.size -= closedSize;

      if (pos.size < 0) {
        const remainingFill = fillSize - closedSize;
        pos.side = side;
        pos.avgEntryPrice = fillPrice;
        pos.size = remainingFill;
      } else if (pos.size === 0) {
        pos.avgEntryPrice = 0;
      }
    }
  }

  /** Apply an order event (placement confirmation, update, cancellation). */
  applyOrderEvent(event: UserOrderEvent): void {
    const { id, orderEventType, originalSize, sizeMatched } = event;

    // Only process events for orders we placed — the user channel broadcasts
    // order events for ALL orders on subscribed markets, not just ours.
    if (!this.knownOrderIds.has(id)) return;

    const parsedOriginal = Number(originalSize);
    const parsedMatched = Number(sizeMatched);

    switch (orderEventType) {
      case "PLACEMENT": {
        // Update existing entry with server-confirmed remaining size
        const existing = this.openOrders.get(id);
        if (existing) {
          existing.remainingSize = parsedOriginal - parsedMatched;
        }
        break;
      }

      case "UPDATE": {
        const existing = this.openOrders.get(id);
        if (existing) {
          existing.remainingSize = existing.originalSize - parsedMatched;
          if (existing.remainingSize <= 0) {
            this.openOrders.delete(id);
          }
        }
        break;
      }

      case "CANCELLATION": {
        this.openOrders.delete(id);
        break;
      }
    }
  }

  // ── Public read API ──────────────────────────────────────────────────────

  /** Get current positions (non-zero size only). */
  getPositions(): Position[] {
    return Array.from(this.positions.values()).filter((p) => p.size !== 0);
  }

  /** Get open orders. */
  getOpenOrders(): OpenOrder[] {
    return Array.from(this.openOrders.values());
  }

  /** Get a snapshot of both. */
  snapshot(): { positions: Position[]; openOrders: OpenOrder[] } {
    return {
      positions: this.getPositions(),
      openOrders: this.getOpenOrders(),
    };
  }

  /** Check if an order ID belongs to us. */
  isOurOrder(orderId: string): boolean {
    return this.knownOrderIds.has(orderId);
  }

  /**
   * Reconcile local open orders against the CLOB's truth.
   * Removes any local orders not found on the CLOB (phantom orders).
   * Returns the number of phantom orders removed.
   */
  reconcileOrders(clobOrderIds: Set<string>): number {
    let removed = 0;
    for (const [orderId] of this.openOrders) {
      if (!clobOrderIds.has(orderId)) {
        console.log(`[PositionStore] Removing phantom order: ${orderId.slice(0, 16)}…`);
        this.openOrders.delete(orderId);
        removed++;
      }
    }
    return removed;
  }

  /** Optimistically remove an order (e.g. after sending cancel to CLOB). */
  removeOrder(orderId: string): void {
    this.openOrders.delete(orderId);
  }

  /** Reset all state (on market rotation or startup). */
  reset(): void {
    this.positions.clear();
    this.openOrders.clear();
    this.knownOrderIds.clear();
    this.processedTradeIds.clear();
    this.pendingSettlement.clear();
  }
}
