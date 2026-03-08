import { appendFileSync, mkdirSync, statSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";


interface OrderFields {
  timestamp: number;
  marketSlug: string;
  orderId: string;
  assetId: string;
  side: string;
  price: number;
  size: number;
  orderType: string;
  status: string;
}

interface FillFields {
  timestamp: number;
  marketSlug: string;
  orderId: string;
  assetId: string;
  side: string;
  price: number;
  size: number;
  feeRateBps: number;
  fillLatencyMs: number;
}

interface PnlFields {
  timestamp: number;
  marketSlug: string;
  assetId: string;
  positionSize: number;
  avgEntry: number;
  markPrice: number;
  unrealizedPnl: number;
  realizedPnl: number;
}

export interface TradeLogFields {
  timestamp: number;
  marketSlug: string;
  conditionId: string;
  assetId: string;
  tokenLabel: string;
  side: string;
  price: number;
  size: number;
  cost: number;
  // Model parameters
  prediction: number;
  confidenceThreshold: number;
  btcVol5m: number;
  volThreshold: number | null;
  volPass: boolean;
  strategyId: string;
}

const TRADES_HEADER =
  "timestamp,session_id,market_slug,condition_id,asset_id,token_label,side,price,size,cost," +
  "prediction,confidence_threshold,btc_vol_5m,vol_threshold,vol_pass,strategy_id," +
  "outcome,pnl";

export class CsvLogger {
  private metricsDir: string;
  private sessionId: string;
  private ordersPath: string;
  private fillsPath: string;
  private pnlPath: string;
  private tradesPath: string;

  constructor(metricsDir: string) {
    this.metricsDir = metricsDir;
    this.sessionId = new Date().toISOString();

    // Create metrics directory if it doesn't exist
    mkdirSync(this.metricsDir, { recursive: true });

    // Set up file paths
    this.ordersPath = join(this.metricsDir, "orders.csv");
    this.fillsPath = join(this.metricsDir, "fills.csv");
    this.pnlPath = join(this.metricsDir, "pnl.csv");
    this.tradesPath = join(this.metricsDir, "trades.csv");

    // Initialize CSV files with headers if needed
    this.initializeFile(
      this.ordersPath,
      "timestamp,session_id,market_slug,order_id,asset_id,side,price,size,order_type,status"
    );
    this.initializeFile(
      this.fillsPath,
      "timestamp,session_id,market_slug,order_id,asset_id,side,price,size,fee_rate_bps,fill_latency_ms"
    );
    this.initializeFile(
      this.pnlPath,
      "timestamp,session_id,market_slug,asset_id,position_size,avg_entry,mark_price,unrealized_pnl,realized_pnl"
    );
    this.initializeFile(this.tradesPath, TRADES_HEADER);
  }

  private initializeFile(filePath: string, header: string): void {
    try {
      const stats = statSync(filePath);
      // File exists - only write header if it's empty
      if (stats.size === 0) {
        appendFileSync(filePath, header + "\n");
      }
    } catch {
      // File doesn't exist - write header
      appendFileSync(filePath, header + "\n");
    }
  }

  logOrder(fields: OrderFields): void {
    const row = [
      fields.timestamp,
      this.sessionId,
      fields.marketSlug,
      fields.orderId,
      fields.assetId,
      fields.side,
      fields.price,
      fields.size,
      fields.orderType,
      fields.status,
    ]
      .map((val) => this.escapeCsvField(val))
      .join(",");

    appendFileSync(this.ordersPath, row + "\n");
  }

  logFill(fields: FillFields): void {
    const row = [
      fields.timestamp,
      this.sessionId,
      fields.marketSlug,
      fields.orderId,
      fields.assetId,
      fields.side,
      fields.price,
      fields.size,
      fields.feeRateBps,
      fields.fillLatencyMs,
    ]
      .map((val) => this.escapeCsvField(val))
      .join(",");

    appendFileSync(this.fillsPath, row + "\n");
  }

  logPnl(fields: PnlFields): void {
    const row = [
      fields.timestamp,
      this.sessionId,
      fields.marketSlug,
      fields.assetId,
      fields.positionSize,
      fields.avgEntry,
      fields.markPrice,
      fields.unrealizedPnl,
      fields.realizedPnl,
    ]
      .map((val) => this.escapeCsvField(val))
      .join(",");

    appendFileSync(this.pnlPath, row + "\n");
  }

  /** Log a trade decision with model parameters. Outcome/PnL filled in later by backfillResolutions(). */
  logTrade(fields: TradeLogFields): void {
    const row = [
      fields.timestamp,
      this.sessionId,
      fields.marketSlug,
      fields.conditionId,
      fields.assetId,
      fields.tokenLabel,
      fields.side,
      fields.price,
      fields.size,
      fields.cost,
      fields.prediction.toFixed(4),
      fields.confidenceThreshold,
      fields.btcVol5m.toFixed(6),
      fields.volThreshold ?? "",
      fields.volPass,
      fields.strategyId,
      "",  // outcome — filled in by backfillResolutions
      "",  // pnl — filled in by backfillResolutions
    ]
      .map((val) => this.escapeCsvField(val))
      .join(",");

    appendFileSync(this.tradesPath, row + "\n");
    console.log(
      `[TradeLog] Logged trade: ${fields.marketSlug} ${fields.tokenLabel} ` +
      `${fields.side} ${fields.size}@${fields.price.toFixed(3)} ` +
      `pred=${fields.prediction.toFixed(3)} cost=$${fields.cost.toFixed(2)}`
    );
  }

  /**
   * Backfill resolution data for trades with empty outcome column.
   * Checks Gamma API for each unresolved conditionId + assetId pair.
   * Called periodically from the redeemer sweep loop.
   */
  async backfillResolutions(): Promise<number> {
    let content: string;
    try {
      content = readFileSync(this.tradesPath, "utf-8");
    } catch {
      return 0;
    }

    const lines = content.split("\n");
    if (lines.length <= 1) return 0; // only header or empty

    const header = lines[0]!;
    let updated = 0;
    const newLines = [header];

    for (let i = 1; i < lines.length; i++) {
      const line = lines[i]!;
      if (!line.trim()) {
        newLines.push(line);
        continue;
      }

      const fields = this.parseCsvLine(line);
      // trades.csv columns: 0=timestamp, 1=session_id, 2=market_slug, 3=condition_id,
      // 4=asset_id, 5=token_label, 6=side, 7=price, 8=size, 9=cost,
      // 10=prediction, 11=confidence_threshold, 12=btc_vol_5m, 13=vol_threshold,
      // 14=vol_pass, 15=strategy_id, 16=outcome, 17=pnl
      const outcome = fields[16] ?? "";
      if (outcome !== "") {
        // Already resolved
        newLines.push(line);
        continue;
      }

      const assetId = fields[4] ?? "";
      const size = parseFloat(fields[8] ?? "0");
      const cost = parseFloat(fields[9] ?? "0");

      const resolution = await this.resolveViaGamma(assetId);
      if (!resolution || !resolution.resolved) {
        newLines.push(line);
        continue;
      }

      const won = assetId === resolution.winnerTokenId;
      const pnl = won ? size - cost : -cost;
      const outcomeStr = won ? "WIN" : "LOSS";

      // Replace the last two fields
      fields[16] = outcomeStr;
      fields[17] = pnl.toFixed(4);
      newLines.push(fields.map((f) => this.escapeCsvField(f)).join(","));
      updated++;

      console.log(
        `[TradeLog] Resolved: ${fields[2]} → ${outcomeStr} (pnl=$${pnl >= 0 ? "+" : ""}${pnl.toFixed(2)})`
      );
    }

    if (updated > 0) {
      writeFileSync(this.tradesPath, newLines.join("\n"));
    }

    return updated;
  }

  /** Query Gamma API for market resolution by asset (token) ID. */
  private async resolveViaGamma(
    assetId: string
  ): Promise<{ resolved: boolean; winnerTokenId: string } | null> {
    try {
      const resp = await fetch(
        `https://gamma-api.polymarket.com/markets?clob_token_ids=${assetId}`
      );
      const data = await resp.json();
      if (!Array.isArray(data) || data.length === 0) return null;

      const market = data[0];
      const prices: string[] = JSON.parse(market.outcomePrices || "[]");
      const tokens: string[] = JSON.parse(market.clobTokenIds || "[]");

      let winnerTokenId = "";
      for (let i = 0; i < prices.length; i++) {
        if (prices[i] === "1") {
          winnerTokenId = tokens[i]?.trim() || "";
        }
      }
      return { resolved: winnerTokenId !== "", winnerTokenId };
    } catch {
      return null;
    }
  }

  /** Parse a CSV line handling quoted fields. */
  private parseCsvLine(line: string): string[] {
    const fields: string[] = [];
    let current = "";
    let inQuotes = false;

    for (let i = 0; i < line.length; i++) {
      const ch = line[i]!;
      if (inQuotes) {
        if (ch === '"') {
          if (i + 1 < line.length && line[i + 1] === '"') {
            current += '"';
            i++; // skip escaped quote
          } else {
            inQuotes = false;
          }
        } else {
          current += ch;
        }
      } else {
        if (ch === '"') {
          inQuotes = true;
        } else if (ch === ",") {
          fields.push(current);
          current = "";
        } else {
          current += ch;
        }
      }
    }
    fields.push(current);
    return fields;
  }

  flush(): void {
    // No-op for synchronous writes
  }

  stop(): void {
    this.flush();
  }

  private escapeCsvField(value: unknown): string {
    const str = String(value);
    // Escape quotes and wrap in quotes if contains comma, newline, or quote
    if (str.includes(",") || str.includes("\n") || str.includes('"')) {
      return '"' + str.replace(/"/g, '""') + '"';
    }
    return str;
  }
}
