// ── Primitives ──

export type Side = "BUY" | "SELL";
export type OrderType = "GTC" | "GTD" | "FOK";
export type OrderEventType = "PLACEMENT" | "UPDATE" | "CANCELLATION";
export type OrderStatus = "LIVE" | "MATCHED" | "CANCELED";
export type TradeStatus = "MATCHED" | "MINED" | "CONFIRMED" | "RETRYING" | "FAILED";
export type TraderSide = "TAKER" | "MAKER";
export type DataMode = "tick" | "resampled";

// ── Market Data ──

export interface PriceLevel {
  price: number;
  size: number;
}

export interface OrderBookState {
  assetId: string;
  bids: PriceLevel[];   // sorted descending by price
  asks: PriceLevel[];   // sorted ascending by price
  timestamp: number;     // exchange timestamp ms
  hash: string;
}

export interface MarketInfo {
  slug: string;
  conditionId: string;
  upTokenId: string;
  downTokenId: string;
  endTime: Date;
  tickSize: number;
}

// ── Positions & Orders ──

export interface Position {
  assetId: string;
  side: Side;
  size: number;
  avgEntryPrice: number;
  realizedPnl: number;
  /** Whether all fills for this position have been confirmed on-chain. */
  settled: boolean;
}

export interface OpenOrder {
  orderId: string;
  assetId: string;
  side: Side;
  price: number;
  originalSize: number;
  remainingSize: number;
  placedAt: number;      // timestamp ms
}

// ── Strategy I/O ──

export interface MarketState {
  market: MarketInfo;
  books: Record<string, OrderBookState>;
  positions: Position[];
  openOrders: OpenOrder[];
  timeRemainingMs: number;
  timestamp: number;
  bars?: ResampledBar[];
  indicators?: IndicatorSnapshot;
  externalData?: Record<string, ExternalDataPoint>;
}

/** Optional metadata attached to a PLACE_ORDER for trade logging. */
export interface TradeMetadata {
  prediction: number;
  confidenceThreshold: number;
  btcVol5m: number;
  volThreshold: number | null;
  volPass: boolean;
  shouldTrade: boolean;
  leaderLabel: string;
  buyToken: "up" | "down";
  strategyId: string;
}

export interface PlaceOrderAction {
  type: "PLACE_ORDER";
  assetId: string;
  side: Side;
  price: number;
  size: number;
  orderType: OrderType;
  /** Model/strategy metadata for trade logging. */
  metadata?: TradeMetadata;
}

export interface CancelOrderAction {
  type: "CANCEL_ORDER";
  orderId: string;
}

export interface CancelAllAction {
  type: "CANCEL_ALL";
}

export interface NonceInvalidateAction {
  type: "NONCE_INVALIDATE";
}

export interface NoopAction {
  type: "NOOP";
}

export interface RedeemAction {
  type: "REDEEM";
  conditionId: string;
}

export interface SplitAction {
  type: "SPLIT";
  conditionId: string;
  amount: bigint; // USDC.e units (6 decimals)
}

export interface MergePairsAction {
  type: "MERGE_PAIRS";
  conditionId: string;
  amount: bigint; // USDC.e units (6 decimals) — number of pairs to merge
}

export type TradeAction =
  | PlaceOrderAction
  | CancelOrderAction
  | CancelAllAction
  | NonceInvalidateAction
  | RedeemAction
  | SplitAction
  | MergePairsAction
  | NoopAction;

// ── Market Data Events (emitted by MarketDataSource) ──

export interface BookEvent {
  type: "book";
  book: OrderBookState;
}

export interface PriceChangeEntry {
  assetId: string;
  price: number;
  size: number;
  side: Side;
  hash: string;
  bestBid: number;
  bestAsk: number;
}

export interface PriceChangeEvent {
  type: "price_change";
  market: string;
  timestamp: number;
  changes: PriceChangeEntry[];
}

export interface MarketOpenEvent {
  type: "market_open";
  market: MarketInfo;
}

export interface MarketCloseEvent {
  type: "market_close";
  conditionId: string;
}

export type MarketDataEvent = BookEvent | PriceChangeEvent | MarketOpenEvent | MarketCloseEvent;

// ── User Channel Events ──

export interface UserOrderEvent {
  eventType: "order";
  id: string;
  owner: string;
  market: string;
  assetId: string;
  side: Side;
  originalSize: number;
  sizeMatched: number;
  price: number;
  associateTrades: string[] | null;
  outcome: string;
  orderEventType: OrderEventType;
  createdAt: string;
  expiration: string;
  orderType: OrderType;
  status: OrderStatus;
  makerAddress: string;
  timestamp: number;
}

export interface MakerOrderInfo {
  orderId: string;
  owner: string;
  makerAddress: string;
  matchedAmount: number;
  price: number;
  feeRateBps: number;
  assetId: string;
  outcome: string;
  side: Side;
}

export interface UserTradeEvent {
  eventType: "trade";
  id: string;
  takerOrderId: string;
  market: string;
  assetId: string;
  side: Side;
  size: number;
  price: number;
  feeRateBps: number;
  status: TradeStatus;
  matchTime: string;
  lastUpdate: string;
  outcome: string;
  owner: string;
  tradeOwner: string;
  makerAddress: string;
  transactionHash: string;
  bucketIndex: number;
  makerOrders: MakerOrderInfo[];
  traderSide: TraderSide;
  timestamp: number;
}

export type UserChannelEvent = UserOrderEvent | UserTradeEvent;

// ── Resampler ──

export interface ResampledBar {
  timestamp: number;
  assetId: string;
  midPrice: number;
  spread: number;
  bookImbalance: number;
  bidPrice: number[];    // top-5 levels
  bidSize: number[];
  askPrice: number[];
  askSize: number[];
  stalenessMs: number;
}

export interface IndicatorSnapshot {
  midPriceSma20: number;
  midPriceEma20: number;
  realizedVol: number;
  rsi14: number;
  spreadSma10: number;
  [key: string]: number;
}

// ── External Feeds ──

export interface ExternalDataPoint {
  feed: string;
  timestamp: number;
  price: number;
  bid?: number;
  ask?: number;
  volume?: number;
}

// ── Config ──

export interface PolymarketConfig {
  host: string;
  chainId: number;
  privateKey: string;
  apiKey: string;
  apiSecret: string;
  apiPassphrase: string;
  proxyWallet: string;
}

export type MarketMakerMode = "mint-and-sell" | "traditional";

export type PricingMode = "inventory-skew" | "whale-front" | "black-scholes";

export interface MarketMakerConfig {
  mode: MarketMakerMode;
  pricing: PricingMode;            // how to set ask prices
  initialPairs: number;          // pairs to mint at market open (mint-and-sell)
  maxTotalInventory: number;     // max tokens per side (total USDC capital at risk)
  maxUnhedgedExposure: number;   // max |inventoryUp - inventoryDown| before pausing lagging side
  skewK: number;                 // A-S skew parameter (price shift per unit of imbalance)
  requoteThreshold: number;      // min price change to trigger cancel-and-replace
  replenishThreshold: number;    // mint more when inventory drops below this
  whaleThreshold: number;        // min size to identify a whale resting order
  whalePullCooldownMs: number;   // how long to pull our orders after whale signal (ms)
  whaleMoveTicks: number;        // whale bid/ask move >= this many ticks triggers cooldown
  // Black-Scholes pricing mode
  bsVolWindowS: number;          // seconds of price history for realized vol estimate
  bsSpreadMultiplier: number;    // spread = multiplier * sigma * sqrt(tau)
  bsMinSpread: number;           // minimum half-spread in cents (floor)
  bsExternalFeed: string;        // which external feed provides the underlying price
  bsTDof: number;                // Student's t degrees of freedom (0 = normal, 3-5 for crypto fat tails)
  // Legacy fields kept for other strategies (smoke-test, xgb)
  orderSize: number;
  cancelWindowS: number;
  minRemainingS: number;
  maxPositions: number;
  priceOffset: number;
}

export interface ExecutionConfig {
  asset: string;
  dataSource: "redis" | "websocket";
  redisUrl: string;
  strategyId: string;
  metricsDir: string;
  rpcUrl: string;
  resampleIntervalMs: number;
  externalFeeds: string[];
  betDollars: number;
  modelServerUrl: string;
  dashboardPort?: number;
  marketDuration: string;
}

export interface AppConfig {
  polymarket: PolymarketConfig;
  marketMaker: MarketMakerConfig;
  execution: ExecutionConfig;
}
