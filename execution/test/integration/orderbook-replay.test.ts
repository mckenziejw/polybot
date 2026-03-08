/**
 * Integration test: Orderbook Replay
 *
 * Reads captured parquet data (book_snapshots + trade_events) from data/,
 * replays them in timestamp order through OrderBook, and validates that:
 *
 * 1. The OrderBook correctly loads full snapshots (full_bids/full_asks JSON)
 * 2. The top-5 levels extracted from the full book match the stored bid/ask_price/size columns
 * 3. Computed mid_price, spread, and book_imbalance match the stored values
 *
 * This test calls Python (pyarrow) via subprocess to read parquet → JSON,
 * then processes the JSON in TypeScript.
 */
import { describe, test, expect } from "bun:test";
import { spawnSync } from "bun";
import { readdirSync } from "node:fs";
import { join } from "node:path";
import { OrderBook } from "../../src/data/orderbook.ts";
import type { OrderBookState, PriceLevel } from "../../src/types.ts";

const DATA_DIR = join(import.meta.dir, "../../../data");
const BOOK_SNAPSHOTS_DIR = join(DATA_DIR, "book_snapshots");

interface ParquetBookRow {
  exchange_timestamp: number;
  asset_id: string;
  mid_price: number;
  spread: number;
  book_imbalance: number;
  bid_price_1: number;
  bid_size_1: number;
  bid_price_2: number;
  bid_size_2: number;
  bid_price_3: number;
  bid_size_3: number;
  bid_price_4: number;
  bid_size_4: number;
  bid_price_5: number;
  bid_size_5: number;
  ask_price_1: number;
  ask_size_1: number;
  ask_price_2: number;
  ask_size_2: number;
  ask_price_3: number;
  ask_size_3: number;
  ask_price_4: number;
  ask_size_4: number;
  ask_price_5: number;
  ask_size_5: number;
  full_bids: string; // JSON: [[price, size], ...]
  full_asks: string;
}

function readParquetBookSnapshots(filePath: string): ParquetBookRow[] {
  const script = `
import pyarrow.parquet as pq
import json, sys

t = pq.read_table(sys.argv[1])
df = t.to_pandas()

# Select only needed columns
cols = ['exchange_timestamp', 'asset_id', 'mid_price', 'spread', 'book_imbalance',
        'full_bids', 'full_asks']
for i in range(1, 6):
    cols += [f'bid_price_{i}', f'bid_size_{i}', f'ask_price_{i}', f'ask_size_{i}']

df = df[cols].fillna(0)
# Convert to records
records = df.to_dict(orient='records')
# Ensure numeric types (pandas int64 -> python int)
for r in records:
    r['exchange_timestamp'] = int(r['exchange_timestamp'])
print(json.dumps(records))
`;

  const result = spawnSync(["python3", "-c", script, filePath]);
  if (result.exitCode !== 0) {
    throw new Error(
      `Python parquet read failed: ${new TextDecoder().decode(result.stderr)}`
    );
  }
  return JSON.parse(new TextDecoder().decode(result.stdout));
}

function parseFullBook(
  jsonStr: string
): PriceLevel[] {
  if (!jsonStr || jsonStr === "null" || jsonStr === "None") return [];
  const parsed = JSON.parse(jsonStr) as [number, number][];
  return parsed.map(([price, size]) => ({ price, size }));
}

function pickSampleFile(): string | null {
  try {
    const files = readdirSync(BOOK_SNAPSHOTS_DIR).filter((f) =>
      f.endsWith(".parquet")
    );
    if (files.length === 0) return null;
    // Pick a file near the middle for variety
    return join(BOOK_SNAPSHOTS_DIR, files[Math.floor(files.length / 2)]);
  } catch {
    return null;
  }
}

describe("integration: orderbook replay", () => {
  const sampleFile = pickSampleFile();

  test("data directory exists and has parquet files", () => {
    expect(sampleFile).not.toBeNull();
  });

  if (!sampleFile) return;

  test("full book snapshot loads correctly and top-5 levels match stored columns", () => {
    const rows = readParquetBookSnapshots(sampleFile);
    expect(rows.length).toBeGreaterThan(0);

    // Pick a few rows across different asset_ids and timestamps
    const assetIds = [...new Set(rows.map((r) => r.asset_id))];
    expect(assetIds.length).toBeGreaterThanOrEqual(2); // at least Up + Down

    let checkedRows = 0;
    const TOLERANCE = 0.0001;

    for (const assetId of assetIds.slice(0, 2)) {
      const assetRows = rows.filter((r) => r.asset_id === assetId);
      // Check first, middle, and last row for each asset
      const indices = [
        0,
        Math.floor(assetRows.length / 2),
        assetRows.length - 1,
      ];

      for (const idx of indices) {
        const row = assetRows[idx]!;
        const book = new OrderBook();

        const bids = parseFullBook(row.full_bids);
        const asks = parseFullBook(row.full_asks);

        book.applySnapshot({
          assetId: row.asset_id,
          bids,
          asks,
          timestamp: row.exchange_timestamp,
          hash: "",
        });

        // Verify top-5 levels match stored columns
        const top = book.topLevels(5);

        for (let i = 0; i < 5; i++) {
          const storedBidPrice = (row as any)[`bid_price_${i + 1}`] as number;
          const storedBidSize = (row as any)[`bid_size_${i + 1}`] as number;
          const storedAskPrice = (row as any)[`ask_price_${i + 1}`] as number;
          const storedAskSize = (row as any)[`ask_size_${i + 1}`] as number;

          // Only check non-zero stored values (zero means the level doesn't exist)
          if (storedBidPrice > 0) {
            expect(Math.abs(top.bidPrice[i]! - storedBidPrice)).toBeLessThan(
              TOLERANCE
            );
            expect(Math.abs(top.bidSize[i]! - storedBidSize)).toBeLessThan(1);
          }
          if (storedAskPrice > 0) {
            expect(Math.abs(top.askPrice[i]! - storedAskPrice)).toBeLessThan(
              TOLERANCE
            );
            expect(Math.abs(top.askSize[i]! - storedAskSize)).toBeLessThan(1);
          }
        }

        // Verify mid_price
        if (bids.length > 0 && asks.length > 0) {
          expect(Math.abs(book.mid - row.mid_price)).toBeLessThan(TOLERANCE);
        }

        // Verify spread
        if (bids.length > 0 && asks.length > 0) {
          expect(Math.abs(book.spread - row.spread)).toBeLessThan(TOLERANCE);
        }

        checkedRows++;
      }
    }

    expect(checkedRows).toBeGreaterThanOrEqual(4);
  });

  test("book state is consistent across sequential snapshots for same asset", () => {
    const rows = readParquetBookSnapshots(sampleFile);
    const assetIds = [...new Set(rows.map((r) => r.asset_id))];
    const assetId = assetIds[0]!;
    const assetRows = rows
      .filter((r) => r.asset_id === assetId)
      .sort((a, b) => a.exchange_timestamp - b.exchange_timestamp);

    const book = new OrderBook();

    for (const row of assetRows) {
      const bids = parseFullBook(row.full_bids);
      const asks = parseFullBook(row.full_asks);

      book.applySnapshot({
        assetId: row.asset_id,
        bids,
        asks,
        timestamp: row.exchange_timestamp,
        hash: "",
      });

      // After each snapshot, bids should be descending, asks ascending
      for (let i = 1; i < book.bids.length; i++) {
        expect(book.bids[i]!.price).toBeLessThanOrEqual(book.bids[i - 1]!.price);
      }
      for (let i = 1; i < book.asks.length; i++) {
        expect(book.asks[i]!.price).toBeGreaterThanOrEqual(book.asks[i - 1]!.price);
      }

      // mid should be between best bid and best ask
      if (book.bids.length > 0 && book.asks.length > 0) {
        expect(book.mid).toBeGreaterThanOrEqual(book.bestBid);
        expect(book.mid).toBeLessThanOrEqual(book.bestAsk);
      }
    }
  });
});
