import type { MarketState, TradeAction, DataMode } from "../types.ts";
import type { Strategy } from "./strategy.ts";
import { registerStrategy } from "./strategy.ts";

/**
 * HTTP-based signal receiver strategy.
 *
 * On each evaluation tick, POSTs the full MarketState to an external model
 * server and receives TradeAction[] back. The model server owns observation
 * construction (tensor building, normalization) — this avoids train/serve skew
 * since the Python side uses the same code as training.
 *
 * The model server must expose:
 *   POST /evaluate  { ...MarketState }  →  { actions: TradeAction[] }
 *
 * Falls back to NOOP if the model server is unreachable or returns an error,
 * so the execution engine stays alive even if the model crashes.
 */
export class SignalReceiverStrategy implements Strategy {
  readonly id = "signal-receiver";
  readonly dataMode: DataMode = "resampled";

  private modelUrl: string;
  private timeoutMs: number;
  private consecutiveErrors = 0;

  constructor(modelUrl = "http://localhost:8000", timeoutMs = 50) {
    this.modelUrl = modelUrl;
    this.timeoutMs = timeoutMs;
  }

  async onMarketOpen(_state: MarketState): Promise<void> {
    this.consecutiveErrors = 0;
  }

  async evaluate(state: MarketState): Promise<TradeAction[]> {
    try {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), this.timeoutMs);

      const response = await fetch(`${this.modelUrl}/evaluate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(state),
        signal: controller.signal,
      });

      clearTimeout(timer);

      if (!response.ok) {
        this.onError(`Model server returned ${response.status}`);
        return [{ type: "NOOP" }];
      }

      const body = (await response.json()) as { actions?: TradeAction[] };
      if (!body.actions || !Array.isArray(body.actions)) {
        this.onError("Model response missing actions array");
        return [{ type: "NOOP" }];
      }

      this.consecutiveErrors = 0;
      return body.actions;
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      this.onError(`Model request failed: ${msg}`);
      return [{ type: "NOOP" }];
    }
  }

  async onMarketClose(_state: MarketState): Promise<TradeAction[]> {
    return [{ type: "CANCEL_ALL" }];
  }

  private onError(msg: string): void {
    this.consecutiveErrors++;
    if (this.consecutiveErrors <= 3 || this.consecutiveErrors % 10 === 0) {
      console.warn(
        `[SignalReceiver] ${msg} (consecutive errors: ${this.consecutiveErrors})`
      );
    }
  }
}

registerStrategy("signal-receiver", () => new SignalReceiverStrategy());
