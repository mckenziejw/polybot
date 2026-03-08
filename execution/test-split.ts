#!/usr/bin/env bun
/**
 * Manual test: split a small amount of USDC.e into YES+NO tokens.
 *
 * Usage: bun run test-split.ts <conditionId> [amount_usdc]
 *
 * amount_usdc defaults to 1 (= 1_000_000 USDC.e units, i.e. $1)
 */

import { loadConfig } from "./src/config.ts";
import { Minter } from "./src/positions/minter.ts";
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
const minter = new Minter(config);
const redeemer = new Redeemer(config);

const account = privateKeyToAccount(config.polymarket.privateKey as Hex);
const holder = config.polymarket.proxyWallet || account.address;

console.log(`EOA: ${account.address}`);
console.log(`Holder (proxy): ${holder}`);

const conditionId = process.argv[2];
const amountUsdc = Number(process.argv[3] || "1");

if (!conditionId) {
  console.log("\nUsage: bun run test-split.ts <conditionId> [amount_usdc]");
  console.log("  conditionId: from an active (unresolved) market");
  console.log("  amount_usdc: dollars to split (default: 1)");
  process.exit(1);
}

const amountUnits = BigInt(Math.round(amountUsdc * 1_000_000));
console.log(`\nSplitting $${amountUsdc} (${amountUnits} units) for condition ${conditionId.slice(0, 16)}…`);

// Step 1: Check USDC.e balance
console.log("\n--- Step 1: Check USDC.e balance ---");
const publicClient = createPublicClient({
  chain: polygon,
  transport: http(config.execution.rpcUrl),
});

const usdcBalance = await publicClient.readContract({
  address: "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174" as Hex,
  abi: parseAbi(["function balanceOf(address) view returns (uint256)"]),
  functionName: "balanceOf",
  args: [holder as Hex],
});
console.log(`USDC.e balance: ${Number(usdcBalance) / 1_000_000} USDC`);

if (usdcBalance < amountUnits) {
  console.log("Insufficient USDC.e balance!");
  process.exit(1);
}

// Step 2: Check token balances before split
console.log("\n--- Step 2: Token balances before split ---");
const balancesBefore = await redeemer.getBalances(conditionId);
console.log(`YES: ${balancesBefore.yes}  NO: ${balancesBefore.no}`);

// Step 3: Split
console.log("\n--- Step 3: Execute split ---");
const result = await minter.split(conditionId, amountUnits);
console.log(`Result: success=${result.success} txHash=${result.txHash ?? "none"} error=${result.error ?? "none"}`);

if (!result.success) {
  process.exit(1);
}

// Step 4: Check token balances after split
console.log("\n--- Step 4: Token balances after split ---");
const balancesAfter = await redeemer.getBalances(conditionId);
console.log(`YES: ${balancesAfter.yes}  NO: ${balancesAfter.no}`);
console.log(`YES delta: +${balancesAfter.yes - balancesBefore.yes}`);
console.log(`NO delta:  +${balancesAfter.no - balancesBefore.no}`);

// Step 5: Check USDC.e balance after
const usdcAfter = await publicClient.readContract({
  address: "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174" as Hex,
  abi: parseAbi(["function balanceOf(address) view returns (uint256)"]),
  functionName: "balanceOf",
  args: [holder as Hex],
});
console.log(`\nUSDC.e spent: ${Number(usdcBalance - usdcAfter) / 1_000_000} USDC`);
