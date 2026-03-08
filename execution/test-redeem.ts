#!/usr/bin/env bun
/**
 * Manual test: check redemption readiness and attempt redemption
 * for a known resolved market.
 *
 * Usage: bun run test-redeem.ts [conditionId]
 *
 * If no conditionId provided, it will try to find one from recent markets.
 */

import { loadConfig } from "./src/config.ts";
import { Redeemer } from "./src/positions/redemption.ts";
import {
  createPublicClient,
  http,
  parseAbi,
  type Hex,
} from "viem";
import { polygon } from "viem/chains";
import { privateKeyToAccount } from "viem/accounts";

const config = loadConfig();
const redeemer = new Redeemer(config);

const account = privateKeyToAccount(config.polymarket.privateKey as Hex);
const holder = config.polymarket.proxyWallet || account.address;

console.log(`EOA: ${account.address}`);
console.log(`Holder (proxy): ${holder}`);
console.log(`RPC: ${config.execution.rpcUrl}`);

// Try to get a conditionId from CLI arg or use a known one
const conditionId = process.argv[2];

if (!conditionId) {
  console.log("\nUsage: bun run test-redeem.ts <conditionId>");
  console.log("Pass a conditionId from a resolved market to test redemption.");
  process.exit(1);
}

console.log(`\nTesting conditionId: ${conditionId}`);

// Step 1: Check resolution
console.log("\n--- Step 1: Check resolution ---");
try {
  const resolved = await redeemer.isResolved(conditionId);
  console.log(`isResolved: ${resolved}`);
  if (!resolved) {
    console.log("Market not resolved yet. Cannot redeem.");
    process.exit(0);
  }
} catch (err) {
  console.error("Error checking resolution:", err);
  process.exit(1);
}

// Step 2: Check balances
console.log("\n--- Step 2: Check balances ---");
try {
  const balances = await redeemer.getBalances(conditionId);
  console.log(`YES balance: ${balances.yes}`);
  console.log(`NO balance: ${balances.no}`);
  if (balances.yes === 0n && balances.no === 0n) {
    console.log("No tokens to redeem (already redeemed or never held).");
    process.exit(0);
  }
} catch (err) {
  console.error("Error checking balances:", err);
  process.exit(1);
}

// Step 3: Attempt redemption
console.log("\n--- Step 3: Attempt redemption ---");
try {
  const result = await redeemer.redeem(conditionId);
  console.log("Result:", JSON.stringify(result, null, 2));
} catch (err) {
  console.error("Redemption error:", err);
}
