/**
 * Fetch and analyze live trading performance from Polymarket CLOB API.
 *
 * Usage:
 *   bun run fetch_activity.ts              # all trades
 *   bun run fetch_activity.ts --since 03-07 # trades from March 7 onward
 *   bun run fetch_activity.ts --xgb        # filter to likely XGB trades (single-side, ~$5)
 */
import { ClobClient } from "@polymarket/clob-client";
import { SignatureType } from "@polymarket/order-utils";
import { Wallet } from "@ethersproject/wallet";
import config from "../config.json";

const pm = config.polymarket;
const PROXY_WALLET = pm.proxy_wallet.toLowerCase();

// Parse CLI args
const args = process.argv.slice(2);
const sinceIdx = args.indexOf("--since");
const sinceDate = sinceIdx >= 0 ? args[sinceIdx + 1] : null;
const xgbOnly = args.includes("--xgb");

interface RawTrade {
  id: string;
  taker_order_id: string;
  market: string;
  asset_id: string;
  side: string;
  size: string;
  price: string;
  fee_rate_bps: string;
  status: string;
  match_time: string;
  outcome: string;
  maker_address: string;
  trader_side: string; // "TAKER" or "MAKER"
  maker_orders: {
    order_id: string;
    maker_address: string;
    matched_amount: string;
    price: string;
    side: string;
    asset_id: string;
    outcome: string;
  }[];
}

interface OurFill {
  tradeId: string;
  conditionId: string;
  assetId: string;
  outcome: string;
  side: string;
  price: number;
  size: number;
  cost: number;
  matchTime: number;
  traderSide: string;
}

function extractOurFills(trade: RawTrade): OurFill[] {
  const fills: OurFill[] = [];

  if (trade.trader_side === "TAKER") {
    // We are the taker — top-level fields are our perspective
    fills.push({
      tradeId: trade.id,
      conditionId: trade.market,
      assetId: trade.asset_id,
      outcome: trade.outcome,
      side: trade.side,
      price: parseFloat(trade.price),
      size: parseFloat(trade.size),
      cost: parseFloat(trade.price) * parseFloat(trade.size),
      matchTime: parseInt(trade.match_time),
      traderSide: "TAKER",
    });
  } else {
    // We are a maker — our fills are in maker_orders where maker_address matches
    for (const mo of trade.maker_orders) {
      if (mo.maker_address.toLowerCase() === PROXY_WALLET) {
        const price = parseFloat(mo.price);
        const size = parseFloat(mo.matched_amount);
        fills.push({
          tradeId: trade.id,
          conditionId: trade.market,
          assetId: mo.asset_id,
          outcome: mo.outcome,
          side: mo.side,
          price,
          size,
          cost: price * size,
          matchTime: parseInt(trade.match_time),
          traderSide: "MAKER",
        });
      }
    }
  }

  return fills;
}

interface MarketPosition {
  conditionId: string;
  assetId: string;
  outcome: string;
  entryPrice: number;
  totalSize: number;
  totalCost: number;
  nFills: number;
  matchTime: number;
}

async function resolveMarket(
  assetId: string
): Promise<{ winner: string; resolved: boolean } | null> {
  try {
    const resp = await fetch(
      `https://gamma-api.polymarket.com/markets?clob_token_ids=${assetId}`
    );
    const data = await resp.json();
    if (!Array.isArray(data) || data.length === 0) return null;

    const market = data[0];
    const prices: string[] = JSON.parse(market.outcomePrices || "[]");
    // clobTokenIds is a JSON-encoded array, NOT comma-separated
    const tokens: string[] = JSON.parse(market.clobTokenIds || "[]");

    let winnerTokenId = "";
    for (let i = 0; i < prices.length; i++) {
      if (prices[i] === "1") {
        winnerTokenId = tokens[i]?.trim() || "";
      }
    }
    return { winner: winnerTokenId, resolved: winnerTokenId !== "" };
  } catch {
    return null;
  }
}

async function main() {
  const signer = new Wallet(pm.private_key);
  const hasProxy = pm.proxy_wallet && pm.proxy_wallet.length > 0;
  const sigType = hasProxy
    ? SignatureType.POLY_GNOSIS_SAFE
    : SignatureType.EOA;

  let client = new ClobClient(
    pm.host,
    pm.chain_id,
    signer,
    { key: pm.api_key, secret: pm.api_secret, passphrase: pm.api_passphrase },
    sigType,
    hasProxy ? pm.proxy_wallet : undefined
  );

  // Derive L2 creds (logged error is expected — derivation still succeeds)
  const derived = await client.createOrDeriveApiKey();
  client = new ClobClient(
    pm.host,
    pm.chain_id,
    signer,
    derived,
    sigType,
    hasProxy ? pm.proxy_wallet : undefined
  );

  // Fetch all trades from CLOB API
  const rawTrades: RawTrade[] = await client.getTrades();
  console.log(`Fetched ${rawTrades.length} raw trades from CLOB API`);

  // Extract our fills (handling taker vs maker perspective)
  let fills = rawTrades.flatMap(extractOurFills);
  console.log(`Extracted ${fills.length} fills (our perspective)`);

  // Date filter
  if (sinceDate) {
    const [month, day] = sinceDate.split("-").map(Number);
    const sinceTs = new Date(2026, month - 1, day).getTime() / 1000;
    fills = fills.filter((f) => f.matchTime >= sinceTs);
    console.log(`After date filter (>= ${sinceDate}): ${fills.length} fills`);
  }

  // Group fills by market+asset to get per-position summary
  const posKey = (f: OurFill) => `${f.conditionId}:${f.assetId}`;
  const positions = new Map<string, MarketPosition>();

  for (const f of fills) {
    const key = posKey(f);
    if (!positions.has(key)) {
      positions.set(key, {
        conditionId: f.conditionId,
        assetId: f.assetId,
        outcome: f.outcome,
        entryPrice: 0,
        totalSize: 0,
        totalCost: 0,
        nFills: 0,
        matchTime: f.matchTime,
      });
    }
    const pos = positions.get(key)!;
    pos.totalSize += f.size;
    pos.totalCost += f.cost;
    pos.nFills++;
    pos.entryPrice = pos.totalCost / pos.totalSize;
    if (f.matchTime < pos.matchTime) pos.matchTime = f.matchTime;
  }

  let marketPositions = [...positions.values()];

  // XGB filter: single-side per market, cost ~$3-$8 (allowing some variance)
  if (xgbOnly) {
    // Find markets where we bought both sides (smoke-test pattern)
    const marketAssets = new Map<string, Set<string>>();
    for (const pos of marketPositions) {
      if (!marketAssets.has(pos.conditionId))
        marketAssets.set(pos.conditionId, new Set());
      marketAssets.get(pos.conditionId)!.add(pos.assetId);
    }
    const dualSideMarkets = new Set(
      [...marketAssets.entries()]
        .filter(([, assets]) => assets.size > 1)
        .map(([cid]) => cid)
    );

    marketPositions = marketPositions.filter(
      (pos) =>
        !dualSideMarkets.has(pos.conditionId) && // single-side only
        pos.totalCost >= 2 &&
        pos.totalCost <= 25 // ~$5 bet range
    );
    console.log(`After XGB filter: ${marketPositions.length} positions`);
  }

  // Sort by time
  marketPositions.sort((a, b) => a.matchTime - b.matchTime);

  // Resolve outcomes via Gamma API
  const resolutions = new Map<string, { winner: string; resolved: boolean }>();
  const uniqueAssets = [...new Set(marketPositions.map((p) => p.assetId))];

  // Batch in groups of 5 to avoid rate limiting
  for (let i = 0; i < uniqueAssets.length; i += 5) {
    const batch = uniqueAssets.slice(i, i + 5);
    const results = await Promise.all(batch.map(resolveMarket));
    for (let j = 0; j < batch.length; j++) {
      if (results[j]) {
        // Find the conditionId for this asset
        const pos = marketPositions.find((p) => p.assetId === batch[j]);
        if (pos) resolutions.set(pos.conditionId, results[j]!);
      }
    }
  }

  // Print results
  console.log("\n=== TRADE-BY-TRADE PERFORMANCE ===");
  let totalPnl = 0;
  let wins = 0;
  let losses = 0;
  let pending = 0;
  const tradeResults: {
    time: string;
    outcome: string;
    entry: number;
    cost: number;
    pnl: number;
    result: string;
  }[] = [];

  for (const pos of marketPositions) {
    const dt = new Date(pos.matchTime * 1000);
    const timeStr = dt.toISOString().slice(5, 16).replace("T", " ");
    const res = resolutions.get(pos.conditionId);

    let result = "PENDING";
    let pnl = 0;

    if (res?.resolved) {
      const won = pos.assetId === res.winner;
      if (won) {
        pnl = pos.totalSize - pos.totalCost;
        wins++;
        result = "WIN";
      } else {
        pnl = -pos.totalCost;
        losses++;
        result = "LOSS";
      }
      totalPnl += pnl;
    } else {
      pending++;
    }

    tradeResults.push({
      time: timeStr,
      outcome: pos.outcome,
      entry: pos.entryPrice,
      cost: pos.totalCost,
      pnl,
      result,
    });

    const pnlStr =
      result === "PENDING"
        ? "---"
        : `$${pnl >= 0 ? "+" : ""}${pnl.toFixed(2)}`;
    console.log(
      `${timeStr} | ${pos.outcome.padEnd(4)} @ ${pos.entryPrice.toFixed(3)} | ` +
        `cost=$${pos.totalCost.toFixed(2)} | ${result.padEnd(7)} | ${pnlStr}`
    );
  }

  // Summary
  console.log("\n=== SUMMARY ===");
  console.log(`Total positions: ${marketPositions.length}`);
  console.log(`Resolved: ${wins + losses} (${wins}W / ${losses}L)`);
  console.log(`Pending: ${pending}`);
  if (wins + losses > 0) {
    console.log(
      `Win rate: ${((wins / (wins + losses)) * 100).toFixed(1)}%`
    );
    console.log(`Total PnL: $${totalPnl.toFixed(2)}`);
    console.log(
      `Avg PnL/trade: $${(totalPnl / (wins + losses)).toFixed(2)}`
    );
    console.log(
      `Avg entry price: $${(tradeResults.filter((r) => r.result !== "PENDING").reduce((s, r) => s + r.entry, 0) / (wins + losses)).toFixed(3)}`
    );
  }

  // Entry price distribution
  console.log("\n=== WIN RATE BY ENTRY PRICE ===");
  const priceBuckets = [0, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.0];
  for (let i = 0; i < priceBuckets.length - 1; i++) {
    const lo = priceBuckets[i];
    const hi = priceBuckets[i + 1];
    const resolved = tradeResults.filter(
      (r) => r.entry >= lo && r.entry < hi && r.result !== "PENDING"
    );
    if (resolved.length > 0) {
      const w = resolved.filter((r) => r.result === "WIN").length;
      const pnl = resolved.reduce((s, r) => s + r.pnl, 0);
      console.log(
        `  [${lo.toFixed(2)}, ${hi.toFixed(2)}): ${w}/${resolved.length} = ${((w / resolved.length) * 100).toFixed(0)}% WR, PnL $${pnl.toFixed(2)}`
      );
    }
  }

  // Daily breakdown
  console.log("\n=== DAILY BREAKDOWN ===");
  const byDay = new Map<string, typeof tradeResults>();
  for (const r of tradeResults) {
    const day = r.time.slice(0, 5);
    if (!byDay.has(day)) byDay.set(day, []);
    byDay.get(day)!.push(r);
  }
  for (const [day, dayResults] of byDay) {
    const resolved = dayResults.filter((r) => r.result !== "PENDING");
    const w = resolved.filter((r) => r.result === "WIN").length;
    const l = resolved.length - w;
    const pend = dayResults.length - resolved.length;
    const pnl = resolved.reduce((s, r) => s + r.pnl, 0);
    const wr =
      resolved.length > 0
        ? ((w / resolved.length) * 100).toFixed(0)
        : "n/a";
    console.log(
      `  ${day} | ${dayResults.length} trades | ${w}W/${l}L (${pend} pending) | WR=${wr}% | PnL=$${pnl.toFixed(2)}`
    );
  }

  // Cumulative PnL
  console.log("\n=== CUMULATIVE PnL ===");
  let cum = 0;
  for (const r of tradeResults) {
    if (r.result !== "PENDING") {
      cum += r.pnl;
      console.log(
        `  ${r.time} | ${r.result.padEnd(4)} ${r.outcome.padEnd(4)} @ ${r.entry.toFixed(3)} | ` +
          `trade=$${r.pnl >= 0 ? "+" : ""}${r.pnl.toFixed(2)} | cum=$${cum.toFixed(2)}`
      );
    }
  }
}

main().catch((e) => console.error("Error:", e.message));
