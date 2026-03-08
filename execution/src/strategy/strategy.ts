import type { MarketState, TradeAction, DataMode } from "../types.ts";

/**
 * Strategy interface — the core abstraction for trading logic.
 *
 * A strategy is a pure function of market state → trade actions.
 * It does NOT execute orders directly; the orchestrator dispatches actions.
 *
 * Strategies declare their dataMode to control how they receive data:
 * - "tick": called on every raw market data event (book snapshot / price change)
 * - "resampled": called only when a new resampled bar is emitted (e.g. every 100ms)
 */
export interface Strategy {
  /** Unique identifier for this strategy. */
  readonly id: string;

  /** Whether this strategy consumes raw ticks or resampled bars. */
  readonly dataMode: DataMode;

  /**
   * Called once when a new market window opens.
   * Use to initialize per-market state (e.g. clear accumulators, set thresholds).
   */
  onMarketOpen(state: MarketState): Promise<void>;

  /**
   * Core decision function. Called on every tick (dataMode=tick) or bar (dataMode=resampled).
   * Receives a complete snapshot of current state and returns zero or more actions.
   * Returning an empty array or [{ type: "NOOP" }] means do nothing.
   */
  evaluate(state: MarketState): Promise<TradeAction[]>;

  /**
   * Called when a market window is about to close.
   * Return cancel actions for any open orders. The orchestrator will execute them.
   */
  onMarketClose(state: MarketState): Promise<TradeAction[]>;
}

/**
 * Registry of available strategy constructors.
 * Strategies register themselves here; the orchestrator looks them up by config.execution.strategyId.
 */
const strategyRegistry = new Map<string, () => Strategy>();

export function registerStrategy(id: string, factory: () => Strategy): void {
  strategyRegistry.set(id, factory);
}

export function createStrategy(id: string): Strategy {
  const factory = strategyRegistry.get(id);
  if (!factory) {
    const available = [...strategyRegistry.keys()].join(", ");
    throw new Error(`Unknown strategy "${id}". Available: ${available || "none"}`);
  }
  return factory();
}
