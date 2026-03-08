/**
 * Integration test: Schema Validation
 *
 * Validates that the captured data files match expected schemas, documents
 * the differences between data sources, and verifies field compatibility
 * with the execution engine's types.
 *
 * Covers:
 * 1. Live book_snapshots schema (10-level depth, 1-indexed, full_bids/full_asks)
 * 2. Live trade_events schema
 * 3. Telonex book_snapshots schema (5-level depth, 1-indexed, token_label)
 * 4. Telonex 100ms resampled schema (staleness_ms, resolved_value)
 * 5. Raw Telonex source schema (0-indexed, string prices)
 * 6. Cross-source field mapping validation
 */
import { describe, test, expect } from "bun:test";
import { spawnSync } from "bun";
import { readdirSync, existsSync } from "node:fs";
import { join } from "node:path";

const DATA_DIR = join(import.meta.dir, "../../../data");
const DATASETS_DIR = join(import.meta.dir, "../../../datasets");

function getParquetSchema(filePath: string): Record<string, string> {
  const script = `
import pyarrow.parquet as pq
import json, sys
schema = pq.read_schema(sys.argv[1])
result = {}
for field in schema:
    result[field.name] = str(field.type)
print(json.dumps(result))
`;
  const result = spawnSync(["python3", "-c", script, filePath]);
  if (result.exitCode !== 0) {
    throw new Error(
      `Python schema read failed: ${new TextDecoder().decode(result.stderr)}`
    );
  }
  return JSON.parse(new TextDecoder().decode(result.stdout));
}

function getSampleRow(filePath: string): Record<string, any> {
  const script = `
import pyarrow.parquet as pq
import json, sys
t = pq.read_table(sys.argv[1])
df = t.to_pandas()
row = df.iloc[0]
result = {}
for col in df.columns:
    val = row[col]
    if hasattr(val, 'item'):
        result[col] = val.item()
    else:
        result[col] = val
print(json.dumps(result, default=str))
`;
  const result = spawnSync(["python3", "-c", script, filePath]);
  if (result.exitCode !== 0) {
    throw new Error(
      `Python read failed: ${new TextDecoder().decode(result.stderr)}`
    );
  }
  return JSON.parse(new TextDecoder().decode(result.stdout));
}

function findFirstFile(dir: string): string | null {
  try {
    if (!existsSync(dir)) return null;
    const files = readdirSync(dir)
      .filter((f) => f.endsWith(".parquet"))
      .sort();
    return files.length > 0 ? join(dir, files[0]) : null;
  } catch {
    return null;
  }
}

function findFirstFileInSubdirs(baseDir: string): string | null {
  try {
    if (!existsSync(baseDir)) return null;
    const subdirs = readdirSync(baseDir);
    for (const sub of subdirs) {
      const subPath = join(baseDir, sub);
      const file = findFirstFile(subPath);
      if (file) return file;
    }
    // Also check baseDir directly
    return findFirstFile(baseDir);
  } catch {
    return null;
  }
}

// ── Expected column sets ──

const LIVE_BOOK_SNAPSHOT_COLS = [
  "exchange_timestamp",
  "received_timestamp",
  "market",
  "asset_id",
  "mid_price",
  "spread",
  "book_imbalance",
  "hash",
  "full_bids",
  "full_asks",
];
// Plus bid_price_1..10, bid_size_1..10, ask_price_1..10, ask_size_1..10
for (let i = 1; i <= 10; i++) {
  LIVE_BOOK_SNAPSHOT_COLS.push(
    `bid_price_${i}`,
    `bid_size_${i}`,
    `ask_price_${i}`,
    `ask_size_${i}`
  );
}

const LIVE_TRADE_EVENT_COLS = [
  "exchange_timestamp",
  "received_timestamp",
  "market",
  "asset_id",
  "side",
  "trade_price",
  "size",
  "best_bid",
  "best_ask",
  "mid_price",
  "spread",
  "hash",
];

const TELONEX_BOOK_SNAPSHOT_COLS = [
  "exchange_timestamp",
  "received_timestamp",
  "slug",
  "market",
  "asset_id",
  "token_label",
  "mid_price",
  "spread",
  "book_imbalance",
];
for (let i = 1; i <= 5; i++) {
  TELONEX_BOOK_SNAPSHOT_COLS.push(
    `bid_price_${i}`,
    `bid_size_${i}`,
    `ask_price_${i}`,
    `ask_size_${i}`
  );
}

const TELONEX_100MS_COLS = [
  "exchange_timestamp",
  "token_label",
  "asset_id",
  "mid_price",
  "spread",
  "book_imbalance",
  "staleness_ms",
  "resolved_value",
];
for (let i = 1; i <= 5; i++) {
  TELONEX_100MS_COLS.push(
    `bid_price_${i}`,
    `bid_size_${i}`,
    `ask_price_${i}`,
    `ask_size_${i}`
  );
}

// ── Tests ──

describe("integration: schema validation", () => {
  // ── Live book_snapshots ──
  describe("live book_snapshots", () => {
    const file = findFirstFile(join(DATA_DIR, "book_snapshots"));

    test("file exists", () => {
      expect(file).not.toBeNull();
    });

    if (!file) return;

    test("has all expected columns", () => {
      const schema = getParquetSchema(file);
      for (const col of LIVE_BOOK_SNAPSHOT_COLS) {
        expect(schema).toHaveProperty(col);
      }
    });

    test("has 10-level depth (1-indexed)", () => {
      const schema = getParquetSchema(file);
      // Level 10 should exist
      expect(schema).toHaveProperty("bid_price_10");
      expect(schema).toHaveProperty("ask_size_10");
    });

    test("numeric columns are float64 or int64", () => {
      const schema = getParquetSchema(file);
      expect(schema["exchange_timestamp"]).toBe("int64");
      expect(schema["mid_price"]).toBe("double");
      expect(schema["bid_price_1"]).toBe("double");
    });

    test("full_bids/full_asks are valid JSON arrays", () => {
      const row = getSampleRow(file);
      const bids = JSON.parse(row.full_bids);
      const asks = JSON.parse(row.full_asks);
      expect(Array.isArray(bids)).toBe(true);
      expect(Array.isArray(asks)).toBe(true);
      if (bids.length > 0) {
        expect(bids[0]).toHaveLength(2); // [price, size]
        expect(typeof bids[0][0]).toBe("number");
      }
    });

    test("asset_id is a long numeric string (token ID)", () => {
      const row = getSampleRow(file);
      expect(row.asset_id.length).toBeGreaterThan(20);
      expect(/^\d+$/.test(row.asset_id)).toBe(true);
    });

    test("market is a hex condition ID", () => {
      const row = getSampleRow(file);
      expect(row.market.startsWith("0x")).toBe(true);
    });
  });

  // ── Live trade_events ──
  describe("live trade_events", () => {
    const file = findFirstFile(join(DATA_DIR, "trade_events"));

    test("file exists", () => {
      expect(file).not.toBeNull();
    });

    if (!file) return;

    test("has all expected columns", () => {
      const schema = getParquetSchema(file);
      for (const col of LIVE_TRADE_EVENT_COLS) {
        expect(schema).toHaveProperty(col);
      }
    });

    test("side is BUY or SELL", () => {
      const row = getSampleRow(file);
      expect(["BUY", "SELL"]).toContain(row.side);
    });

    test("trade_price is between 0 and 1", () => {
      const row = getSampleRow(file);
      expect(row.trade_price).toBeGreaterThanOrEqual(0);
      expect(row.trade_price).toBeLessThanOrEqual(1);
    });
  });

  // ── Telonex book_snapshots ──
  describe("telonex book_snapshots", () => {
    const file = findFirstFile(join(DATA_DIR, "telonex_book_snapshots"));

    test("file exists", () => {
      expect(file).not.toBeNull();
    });

    if (!file) return;

    test("has all expected columns", () => {
      const schema = getParquetSchema(file);
      for (const col of TELONEX_BOOK_SNAPSHOT_COLS) {
        expect(schema).toHaveProperty(col);
      }
    });

    test("has only 5-level depth (no level 6+)", () => {
      const schema = getParquetSchema(file);
      expect(schema).toHaveProperty("bid_price_5");
      expect(schema).not.toHaveProperty("bid_price_6");
    });

    test("token_label is Up or Down", () => {
      const row = getSampleRow(file);
      expect(["Up", "Down"]).toContain(row.token_label);
    });

    test("slug matches expected format", () => {
      const row = getSampleRow(file);
      expect(row.slug).toMatch(/^btc-updown-\d+m-\d+$/);
    });

    test("received_timestamp is always 0 (Telonex does not populate it)", () => {
      const row = getSampleRow(file);
      expect(row.received_timestamp).toBe(0);
    });
  });

  // ── Telonex 100ms resampled ──
  describe("telonex 100ms resampled", () => {
    const file = findFirstFile(join(DATA_DIR, "telonex_100ms"));

    test("file exists", () => {
      expect(file).not.toBeNull();
    });

    if (!file) return;

    test("has all expected columns", () => {
      const schema = getParquetSchema(file);
      for (const col of TELONEX_100MS_COLS) {
        expect(schema).toHaveProperty(col);
      }
    });

    test("has staleness_ms column", () => {
      const schema = getParquetSchema(file);
      expect(schema).toHaveProperty("staleness_ms");
    });

    test("has resolved_value column", () => {
      const schema = getParquetSchema(file);
      expect(schema).toHaveProperty("resolved_value");
    });

    test("exchange_timestamp is aligned to 100ms grid", () => {
      const row = getSampleRow(file);
      expect(row.exchange_timestamp % 100).toBe(0);
    });

    test("does NOT have market, received_timestamp, or slug columns", () => {
      const schema = getParquetSchema(file);
      expect(schema).not.toHaveProperty("market");
      expect(schema).not.toHaveProperty("received_timestamp");
      expect(schema).not.toHaveProperty("slug");
    });
  });

  // ── Raw Telonex source (0-indexed, string types) ──
  describe("raw telonex source", () => {
    // These are in datasets/telonex_raw/ or datasets/telonex_15m_raw/
    const rawDir15m = join(DATASETS_DIR, "telonex_15m_raw");
    const rawDir5m = join(DATASETS_DIR, "telonex_raw");
    const file = findFirstFile(rawDir15m) ?? findFirstFile(rawDir5m);

    test("file exists", () => {
      expect(file).not.toBeNull();
    });

    if (!file) return;

    test("uses 0-based indexing (bid_price_0, not bid_price_1)", () => {
      const schema = getParquetSchema(file);
      expect(schema).toHaveProperty("bid_price_0");
      expect(schema).toHaveProperty("ask_price_0");
      expect(schema).not.toHaveProperty("bid_price_5"); // only 0-4
    });

    test("price/size columns are STRING type (not double)", () => {
      const schema = getParquetSchema(file);
      // pyarrow reports string as "string" or "large_string"
      expect(schema["bid_price_0"]).toMatch(/string/);
      expect(schema["ask_price_0"]).toMatch(/string/);
    });

    test("timestamps are microseconds (not milliseconds)", () => {
      const schema = getParquetSchema(file);
      expect(schema).toHaveProperty("timestamp_us");
      expect(schema).not.toHaveProperty("exchange_timestamp");

      const row = getSampleRow(file);
      // Microsecond timestamps are ~16 digits (vs 13 for ms)
      expect(String(row.timestamp_us).length).toBeGreaterThanOrEqual(15);
    });

    test("uses outcome instead of token_label", () => {
      const schema = getParquetSchema(file);
      expect(schema).toHaveProperty("outcome");
      expect(schema).not.toHaveProperty("token_label");

      const row = getSampleRow(file);
      expect(["Up", "Down"]).toContain(row.outcome);
    });
  });

  // ── Cross-source compatibility ──
  describe("cross-source field mapping", () => {
    const liveFile = findFirstFile(join(DATA_DIR, "book_snapshots"));
    const telonexFile = findFirstFile(
      join(DATA_DIR, "telonex_book_snapshots")
    );

    if (!liveFile || !telonexFile) {
      test("both data sources available", () => {
        expect(liveFile).not.toBeNull();
        expect(telonexFile).not.toBeNull();
      });
      return;
    }

    test("common columns have compatible types", () => {
      const liveSchema = getParquetSchema(liveFile);
      const telonexSchema = getParquetSchema(telonexFile);

      // These columns exist in both and should have the same type
      const commonCols = [
        "exchange_timestamp",
        "market",
        "asset_id",
        "mid_price",
        "spread",
        "book_imbalance",
      ];

      for (const col of commonCols) {
        expect(liveSchema).toHaveProperty(col);
        expect(telonexSchema).toHaveProperty(col);
        expect(liveSchema[col]).toBe(telonexSchema[col]);
      }
    });

    test("live has 10 levels, telonex has 5", () => {
      const liveSchema = getParquetSchema(liveFile);
      const telonexSchema = getParquetSchema(telonexFile);

      // First 5 levels should match in both
      for (let i = 1; i <= 5; i++) {
        expect(liveSchema).toHaveProperty(`bid_price_${i}`);
        expect(telonexSchema).toHaveProperty(`bid_price_${i}`);
      }

      // Levels 6-10 only in live
      expect(liveSchema).toHaveProperty("bid_price_10");
      expect(telonexSchema).not.toHaveProperty("bid_price_6");
    });

    test("telonex has token_label, live does not", () => {
      const liveSchema = getParquetSchema(liveFile);
      const telonexSchema = getParquetSchema(telonexFile);

      expect(telonexSchema).toHaveProperty("token_label");
      expect(liveSchema).not.toHaveProperty("token_label");
    });

    test("live has full_bids/full_asks, telonex does not", () => {
      const liveSchema = getParquetSchema(liveFile);
      const telonexSchema = getParquetSchema(telonexFile);

      expect(liveSchema).toHaveProperty("full_bids");
      expect(liveSchema).toHaveProperty("full_asks");
      expect(telonexSchema).not.toHaveProperty("full_bids");
      expect(telonexSchema).not.toHaveProperty("full_asks");
    });
  });
});
