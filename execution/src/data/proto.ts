import * as path from "path";
import * as protobuf from "protobufjs";

// Type definitions for decoded protobuf messages
export interface DecodedBookSnapshot {
  market: string;
  assetId: string;
  exchangeTimestamp: number;
  hash: string;
  bids: Array<{ price: number; size: number }>;
  asks: Array<{ price: number; size: number }>;
  tickSize: number;
  lastTradePrice: number;
}

export interface DecodedPriceChange {
  assetId: string;
  price: number;
  size: number;
  side: "BUY" | "SELL";
  hash: string;
  bestBid: number;
  bestAsk: number;
}

export interface DecodedTradeEvent {
  market: string;
  exchangeTimestamp: number;
  priceChanges: DecodedPriceChange[];
}

export type DecodedMarketEvent =
  | {
      type: "book";
      receivedTimestamp: number;
      book: DecodedBookSnapshot;
    }
  | {
      type: "trade";
      receivedTimestamp: number;
      trade: DecodedTradeEvent;
    };

// Cached proto root and message types
let root: protobuf.Root | null = null;
let MarketEventType: protobuf.Type | null = null;

/**
 * Initialize the protobuf schema by loading the polymarket.proto file.
 * This function must be called before decodeMarketEvent().
 */
export async function initProto(): Promise<void> {
  const protoPath = path.resolve(
    import.meta.dir,
    "../../../polymarket.proto"
  );

  root = await protobuf.load(protoPath);
  MarketEventType = root.lookupType("polymarket.MarketEvent");
}

/**
 * Map protobuf Side enum value to string representation.
 */
function mapSide(sideValue: number): "BUY" | "SELL" {
  switch (sideValue) {
    case 1:
      return "BUY";
    case 2:
      return "SELL";
    default:
      throw new Error(`Unknown Side enum value: ${sideValue}`);
  }
}

/**
 * Decode a binary protobuf buffer into a typed DecodedMarketEvent.
 * Throws if initProto() has not been called.
 */
export function decodeMarketEvent(buffer: Buffer): DecodedMarketEvent {
  if (!root || !MarketEventType) {
    throw new Error(
      "Proto not initialized. Call initProto() before decodeMarketEvent()."
    );
  }

  const message = MarketEventType.decode(buffer);
  const obj = MarketEventType.toObject(message, {
    longs: Number,
    enums: Number,
    bytes: Buffer,
  });

  const receivedTimestamp = Number(obj.receivedTimestamp || 0);

  // Handle BookSnapshot payload
  if (obj.book) {
    const book = obj.book;
    const decodedBook: DecodedBookSnapshot = {
      market: book.market || "",
      assetId: book.assetId || "",
      exchangeTimestamp: Number(book.exchangeTimestamp || 0),
      hash: book.hash || "",
      bids: (book.bids || []).map((bid: any) => ({
        price: bid.price || 0,
        size: bid.size || 0,
      })),
      asks: (book.asks || []).map((ask: any) => ({
        price: ask.price || 0,
        size: ask.size || 0,
      })),
      tickSize: book.tickSize || 0,
      lastTradePrice: book.lastTradePrice || 0,
    };

    return {
      type: "book",
      receivedTimestamp,
      book: decodedBook,
    };
  }

  // Handle TradeEvent payload
  if (obj.trade) {
    const trade = obj.trade;
    const decodedTrade: DecodedTradeEvent = {
      market: trade.market || "",
      exchangeTimestamp: Number(trade.exchangeTimestamp || 0),
      priceChanges: (trade.priceChanges || []).map((change: any) => ({
        assetId: change.assetId || "",
        price: change.price || 0,
        size: change.size || 0,
        side: mapSide(change.side || 0),
        hash: change.hash || "",
        bestBid: change.bestBid || 0,
        bestAsk: change.bestAsk || 0,
      })),
    };

    return {
      type: "trade",
      receivedTimestamp,
      trade: decodedTrade,
    };
  }

  throw new Error("MarketEvent has no book or trade payload");
}
