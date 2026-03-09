import type { PriceLevel, OrderBookState, PriceChangeEntry } from "../types.ts";

/**
 * Maintains in-memory L2 orderbook state for a single asset.
 * Supports both full snapshot replacement and incremental price_change updates.
 */
export class OrderBook {
  private _bids: PriceLevel[] = [];
  private _asks: PriceLevel[] = [];
  private _timestamp = 0;
  private _hash = "";
  private _assetId = "";

  get bids(): PriceLevel[] {
    return this._bids;
  }

  get asks(): PriceLevel[] {
    return this._asks;
  }

  get timestamp(): number {
    return this._timestamp;
  }

  get assetId(): string {
    return this._assetId;
  }

  get bestBid(): number {
    return this._bids[0]?.price ?? 0;
  }

  get bestAsk(): number {
    return this._asks[0]?.price ?? Infinity;
  }

  get mid(): number {
    if (this._bids.length === 0 || this._asks.length === 0) return 0;
    return (this.bestBid + this.bestAsk) / 2;
  }

  get spread(): number {
    if (this._bids.length === 0 || this._asks.length === 0) return 0;
    return this.bestAsk - this.bestBid;
  }

  get imbalance(): number {
    const bidSize = this._bids.slice(0, 10).reduce((s, l) => s + l.size, 0);
    const askSize = this._asks.slice(0, 10).reduce((s, l) => s + l.size, 0);
    const total = bidSize + askSize;
    if (total === 0) return 0;
    return (bidSize - askSize) / total;
  }

  /** Replace the entire book from a full snapshot. */
  applySnapshot(state: OrderBookState): void {
    this._assetId = state.assetId;
    this._timestamp = state.timestamp;
    this._hash = state.hash;
    this._bids = state.bids.map((l) => ({ ...l }));
    this._asks = state.asks.map((l) => ({ ...l }));
    // Ensure sorting: bids descending, asks ascending
    this._bids.sort((a, b) => b.price - a.price);
    this._asks.sort((a, b) => a.price - b.price);
  }

  /**
   * Apply an incremental price_change update.
   * - size === 0: remove the level
   * - size > 0: insert or update the level
   * - side BUY affects bids, SELL affects asks
   */
  applyPriceChange(change: PriceChangeEntry, timestamp: number): void {
    this._timestamp = timestamp;
    this._hash = change.hash;

    const levels = change.side === "BUY" ? this._bids : this._asks;
    const idx = levels.findIndex((l) => l.price === change.price);

    if (change.size === 0) {
      // Remove level
      if (idx !== -1) levels.splice(idx, 1);
    } else if (idx !== -1) {
      // Update existing level
      levels[idx]!.size = change.size;
    } else {
      // Insert new level in sorted position
      const newLevel: PriceLevel = { price: change.price, size: change.size };
      if (change.side === "BUY") {
        // Bids: descending — find first price smaller than ours
        const insertAt = levels.findIndex((l) => l.price < change.price);
        if (insertAt === -1) levels.push(newLevel);
        else levels.splice(insertAt, 0, newLevel);
      } else {
        // Asks: ascending — find first price larger than ours
        const insertAt = levels.findIndex((l) => l.price > change.price);
        if (insertAt === -1) levels.push(newLevel);
        else levels.splice(insertAt, 0, newLevel);
      }
    }
  }

  /** Return a snapshot of current state. */
  snapshot(): OrderBookState {
    return {
      assetId: this._assetId,
      bids: this._bids.map((l) => ({ ...l })),
      asks: this._asks.map((l) => ({ ...l })),
      timestamp: this._timestamp,
      hash: this._hash,
    };
  }

  /** Get top N levels for each side. Pads with zeros if fewer levels exist. */
  topLevels(n: number): { bidPrice: number[]; bidSize: number[]; askPrice: number[]; askSize: number[] } {
    const bidPrice: number[] = [];
    const bidSize: number[] = [];
    const askPrice: number[] = [];
    const askSize: number[] = [];

    for (let i = 0; i < n; i++) {
      bidPrice.push(this._bids[i]?.price ?? 0);
      bidSize.push(this._bids[i]?.size ?? 0);
      askPrice.push(this._asks[i]?.price ?? 0);
      askSize.push(this._asks[i]?.size ?? 0);
    }

    return { bidPrice, bidSize, askPrice, askSize };
  }

  /** Reset to empty state. */
  clear(): void {
    this._bids = [];
    this._asks = [];
    this._timestamp = 0;
    this._hash = "";
    this._assetId = "";
  }
}

/**
 * Manages orderbooks for multiple assets (e.g., Up + Down tokens).
 */
export class OrderBookManager {
  private books = new Map<string, OrderBook>();

  get(assetId: string): OrderBook {
    let book = this.books.get(assetId);
    if (!book) {
      book = new OrderBook();
      this.books.set(assetId, book);
    }
    return book;
  }

  allSnapshots(): Record<string, OrderBookState> {
    const result: Record<string, OrderBookState> = {};
    for (const [assetId, book] of this.books) {
      result[assetId] = book.snapshot();
    }
    return result;
  }

  clear(): void {
    for (const book of this.books.values()) {
      book.clear();
    }
    this.books.clear();
  }
}
