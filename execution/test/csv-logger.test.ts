import { describe, test, expect, beforeEach, afterEach } from "bun:test";
import { readFileSync, rmSync, existsSync } from "node:fs";
import { join } from "node:path";
import { CsvLogger } from "../src/logging/csv-logger.ts";

// Generate a unique temp directory per test run to avoid cross-test contamination.
function makeTempDir(): string {
  const rand = Math.random().toString(36).slice(2, 10);
  return `/tmp/test-csv-logger-${rand}`;
}

function readLines(filePath: string): string[] {
  return readFileSync(filePath, "utf-8").split("\n").filter((l) => l.length > 0);
}

describe("CsvLogger", () => {
  let tmpDir: string;
  let logger: CsvLogger;

  beforeEach(() => {
    tmpDir = makeTempDir();
    logger = new CsvLogger(tmpDir);
  });

  afterEach(() => {
    logger.stop();
    if (existsSync(tmpDir)) {
      rmSync(tmpDir, { recursive: true, force: true });
    }
  });

  // ── Constructor ────────────────────────────────────────────────────────────

  describe("constructor", () => {
    test("creates the metrics directory", () => {
      expect(existsSync(tmpDir)).toBe(true);
    });

    test("creates orders.csv with correct header", () => {
      const ordersPath = join(tmpDir, "orders.csv");
      expect(existsSync(ordersPath)).toBe(true);
      const lines = readLines(ordersPath);
      expect(lines[0]).toBe(
        "timestamp,session_id,market_slug,order_id,asset_id,side,price,size,order_type,status"
      );
    });

    test("creates fills.csv with correct header", () => {
      const fillsPath = join(tmpDir, "fills.csv");
      expect(existsSync(fillsPath)).toBe(true);
      const lines = readLines(fillsPath);
      expect(lines[0]).toBe(
        "timestamp,session_id,market_slug,order_id,asset_id,side,price,size,fee_rate_bps,fill_latency_ms"
      );
    });

    test("creates pnl.csv with correct header", () => {
      const pnlPath = join(tmpDir, "pnl.csv");
      expect(existsSync(pnlPath)).toBe(true);
      const lines = readLines(pnlPath);
      expect(lines[0]).toBe(
        "timestamp,session_id,market_slug,asset_id,position_size,avg_entry,mark_price,unrealized_pnl,realized_pnl"
      );
    });

    test("does not overwrite existing non-empty CSV files", () => {
      // Write one order so the file is non-empty
      logger.logOrder({
        timestamp: 1000,
        marketSlug: "slug",
        orderId: "o1",
        assetId: "a1",
        side: "BUY",
        price: 0.5,
        size: 10,
        orderType: "GTC",
        status: "LIVE",
      });

      // Create a second logger pointing at the same directory
      const logger2 = new CsvLogger(tmpDir);
      const lines = readLines(join(tmpDir, "orders.csv"));
      // header + 1 data row — no duplicate header
      expect(lines[0]).toMatch(/^timestamp,/);
      expect(lines.filter((l) => l.startsWith("timestamp")).length).toBe(1);
      logger2.stop();
    });
  });

  // ── logOrder ───────────────────────────────────────────────────────────────

  describe("logOrder", () => {
    test("appends a row with the correct number of columns", () => {
      logger.logOrder({
        timestamp: 1700000000000,
        marketSlug: "btc-price",
        orderId: "ord-123",
        assetId: "asset-abc",
        side: "BUY",
        price: 0.45,
        size: 50,
        orderType: "GTC",
        status: "LIVE",
      });

      const lines = readLines(join(tmpDir, "orders.csv"));
      // lines[0] = header, lines[1] = data row
      expect(lines).toHaveLength(2);
      const cols = lines[1]!.split(",");
      // 10 columns: timestamp,session_id,market_slug,order_id,asset_id,side,price,size,order_type,status
      expect(cols).toHaveLength(10);
    });

    test("writes correct field values in the data row", () => {
      logger.logOrder({
        timestamp: 1234567890,
        marketSlug: "eth-price",
        orderId: "ord-999",
        assetId: "asset-xyz",
        side: "SELL",
        price: 0.72,
        size: 25,
        orderType: "FOK",
        status: "MATCHED",
      });

      const lines = readLines(join(tmpDir, "orders.csv"));
      const row = lines[1]!;
      expect(row).toContain("1234567890");
      expect(row).toContain("eth-price");
      expect(row).toContain("ord-999");
      expect(row).toContain("asset-xyz");
      expect(row).toContain("SELL");
      expect(row).toContain("0.72");
      expect(row).toContain("25");
      expect(row).toContain("FOK");
      expect(row).toContain("MATCHED");
    });

    test("appends multiple rows", () => {
      for (let i = 0; i < 3; i++) {
        logger.logOrder({
          timestamp: i,
          marketSlug: "market",
          orderId: `ord-${i}`,
          assetId: "a",
          side: "BUY",
          price: 0.5,
          size: 10,
          orderType: "GTC",
          status: "LIVE",
        });
      }
      const lines = readLines(join(tmpDir, "orders.csv"));
      expect(lines).toHaveLength(4); // 1 header + 3 rows
    });
  });

  // ── logFill ────────────────────────────────────────────────────────────────

  describe("logFill", () => {
    test("appends a row with the correct number of columns", () => {
      logger.logFill({
        timestamp: 1700000001000,
        marketSlug: "btc-price",
        orderId: "fill-ord-1",
        assetId: "asset-abc",
        side: "BUY",
        price: 0.5,
        size: 30,
        feeRateBps: 50,
        fillLatencyMs: 12,
      });

      const lines = readLines(join(tmpDir, "fills.csv"));
      expect(lines).toHaveLength(2);
      const cols = lines[1]!.split(",");
      // 10 columns: timestamp,session_id,market_slug,order_id,asset_id,side,price,size,fee_rate_bps,fill_latency_ms
      expect(cols).toHaveLength(10);
    });

    test("writes correct field values", () => {
      logger.logFill({
        timestamp: 9876543210,
        marketSlug: "sol-price",
        orderId: "fill-999",
        assetId: "asset-sol",
        side: "SELL",
        price: 0.8,
        size: 5,
        feeRateBps: 100,
        fillLatencyMs: 42,
      });

      const lines = readLines(join(tmpDir, "fills.csv"));
      const row = lines[1]!;
      expect(row).toContain("9876543210");
      expect(row).toContain("sol-price");
      expect(row).toContain("fill-999");
      expect(row).toContain("asset-sol");
      expect(row).toContain("SELL");
      expect(row).toContain("0.8");
      expect(row).toContain("5");
      expect(row).toContain("100");
      expect(row).toContain("42");
    });
  });

  // ── logPnl ─────────────────────────────────────────────────────────────────

  describe("logPnl", () => {
    test("appends a row with the correct number of columns", () => {
      logger.logPnl({
        timestamp: 1700000002000,
        marketSlug: "btc-price",
        assetId: "asset-abc",
        positionSize: 100,
        avgEntry: 0.45,
        markPrice: 0.5,
        unrealizedPnl: 5.0,
        realizedPnl: 2.5,
      });

      const lines = readLines(join(tmpDir, "pnl.csv"));
      expect(lines).toHaveLength(2);
      const cols = lines[1]!.split(",");
      // 9 columns: timestamp,session_id,market_slug,asset_id,position_size,avg_entry,mark_price,unrealized_pnl,realized_pnl
      expect(cols).toHaveLength(9);
    });

    test("writes correct field values", () => {
      logger.logPnl({
        timestamp: 111222333,
        marketSlug: "eth-price",
        assetId: "asset-eth",
        positionSize: 200,
        avgEntry: 0.3,
        markPrice: 0.35,
        unrealizedPnl: 10,
        realizedPnl: 3.75,
      });

      const lines = readLines(join(tmpDir, "pnl.csv"));
      const row = lines[1]!;
      expect(row).toContain("111222333");
      expect(row).toContain("eth-price");
      expect(row).toContain("asset-eth");
      expect(row).toContain("200");
      expect(row).toContain("0.3");
      expect(row).toContain("0.35");
      expect(row).toContain("10");
      expect(row).toContain("3.75");
    });
  });

  // ── CSV escaping ───────────────────────────────────────────────────────────

  describe("CSV escaping", () => {
    test("wraps field in quotes when it contains a comma", () => {
      // Market slug with comma-like content is unusual, but we can test via
      // a marketSlug that contains a comma to exercise the escaping path.
      logger.logOrder({
        timestamp: 1,
        marketSlug: "slug,with,commas",
        orderId: "o1",
        assetId: "a1",
        side: "BUY",
        price: 0.5,
        size: 1,
        orderType: "GTC",
        status: "LIVE",
      });

      const raw = readFileSync(join(tmpDir, "orders.csv"), "utf-8");
      // The slug field must be wrapped in double quotes
      expect(raw).toContain('"slug,with,commas"');
    });

    test("escapes embedded double quotes by doubling them", () => {
      logger.logOrder({
        timestamp: 2,
        marketSlug: 'slug"quoted"',
        orderId: "o2",
        assetId: "a2",
        side: "SELL",
        price: 0.6,
        size: 2,
        orderType: "GTC",
        status: "LIVE",
      });

      const raw = readFileSync(join(tmpDir, "orders.csv"), "utf-8");
      // Should appear as "slug""quoted""" in the CSV
      expect(raw).toContain('"slug""quoted"""');
    });

    test("does not quote plain numeric or string fields", () => {
      logger.logOrder({
        timestamp: 3,
        marketSlug: "plain",
        orderId: "o3",
        assetId: "a3",
        side: "BUY",
        price: 0.5,
        size: 5,
        orderType: "GTC",
        status: "LIVE",
      });

      const lines = readLines(join(tmpDir, "orders.csv"));
      const row = lines[1]!;
      // None of the plain fields should be wrapped in extra quotes
      expect(row).toContain(",plain,");
      expect(row).toContain(",o3,");
      expect(row).not.toMatch(/,"plain",/); // no surrounding quotes
    });

    test("wraps field containing newline in quotes", () => {
      logger.logOrder({
        timestamp: 4,
        marketSlug: "slug\nnewline",
        orderId: "o4",
        assetId: "a4",
        side: "BUY",
        price: 0.5,
        size: 1,
        orderType: "GTC",
        status: "LIVE",
      });

      const raw = readFileSync(join(tmpDir, "orders.csv"), "utf-8");
      expect(raw).toContain('"slug\nnewline"');
    });
  });
});
