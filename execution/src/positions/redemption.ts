import {
  createWalletClient,
  createPublicClient,
  http,
  parseAbi,
  encodeFunctionData,
  type Hex,
  type TransactionReceipt,
} from "viem";
import { polygon } from "viem/chains";
import { privateKeyToAccount } from "viem/accounts";
import type { AppConfig } from "../types.ts";
import type { CsvLogger } from "../logging/csv-logger.ts";

// ── Contract addresses (Polygon) ──────────────────────────────────────────────

/** Gnosis ConditionalTokens (CTF Core) — holds positions, handles redemption. */
const CTF_CORE_ADDRESS = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045" as const;

/** USDC.e on Polygon — collateral token for Polymarket. */
const USDC_E_ADDRESS = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174" as const;

/** Zero bytes32 — parentCollectionId for non-branching (top-level) markets. */
const PARENT_COLLECTION_ID =
  "0x0000000000000000000000000000000000000000000000000000000000000000" as Hex;

/** Index sets for binary YES/NO markets: YES=1, NO=2. */
const BINARY_INDEX_SETS = [1n, 2n] as const;

// ── ABIs ──────────────────────────────────────────────────────────────────────

const CTF_CORE_ABI = parseAbi([
  "function redeemPositions(address collateralToken, bytes32 parentCollectionId, bytes32 conditionId, uint256[] indexSets) external",
  "function payoutDenominator(bytes32 conditionId) external view returns (uint256)",
  "function balanceOf(address owner, uint256 id) external view returns (uint256)",
  "function getCollectionId(bytes32 parentCollectionId, bytes32 conditionId, uint256 indexSet) external view returns (bytes32)",
  "function getPositionId(address collateralToken, bytes32 collectionId) external pure returns (uint256)",
]);

const GNOSIS_SAFE_ABI = parseAbi([
  "function execTransaction(address to, uint256 value, bytes data, uint8 operation, uint256 safeTxGas, uint256 baseGas, uint256 gasPrice, address gasToken, address refundReceiver, bytes signatures) external payable returns (bool success)",
  "function nonce() external view returns (uint256)",
  "function getTransactionHash(address to, uint256 value, bytes data, uint8 operation, uint256 safeTxGas, uint256 baseGas, uint256 gasPrice, address gasToken, address refundReceiver, uint256 _nonce) external view returns (bytes32)",
]);

// ── Types ─────────────────────────────────────────────────────────────────────

export interface RedemptionResult {
  conditionId: string;
  success: boolean;
  txHash?: string;
  error?: string;
}

const DEFAULT_SWEEP_INTERVAL_MS = 60_000;

/** Minimum time after market close before attempting redemption. */
const MIN_REDEMPTION_DELAY_MS = 600_000; // 10 minutes — BTC Up/Down markets take ~5min+ to resolve

// ── Redeemer class ────────────────────────────────────────────────────────────

interface PendingRedemption {
  conditionId: string;
  queuedAt: number; // Date.now() when queued
}

/**
 * Redeems winning conditional tokens for USDC.e collateral after market resolution.
 *
 * Routes through Gnosis Safe `execTransaction` when a proxy wallet is configured,
 * otherwise calls `redeemPositions` directly from the EOA.
 *
 * Markets are not checked for resolution until MIN_REDEMPTION_DELAY_MS after
 * being queued, since markets cannot resolve instantly after close.
 *
 * Usage:
 *   const redeemer = new Redeemer(config);
 *   redeemer.queueForRedemption(conditionId);
 *   redeemer.startSweep();  // checks every 60s
 */
export class Redeemer {
  private config: AppConfig;
  private pendingQueue = new Map<string, PendingRedemption>();
  private sweepTimer: ReturnType<typeof setInterval> | null = null;
  private csvLogger: CsvLogger | null = null;

  constructor(config: AppConfig) {
    this.config = config;
  }

  /** Attach a CsvLogger for trade resolution backfill during sweeps. */
  setCsvLogger(logger: CsvLogger): void {
    this.csvLogger = logger;
  }

  // ── On-chain reads ────────────────────────────────────────────────────────

  /** Check if a condition has been resolved (payout denominator > 0). */
  async isResolved(conditionId: string): Promise<boolean> {
    const publicClient = this.createPublicClient();
    const denom = await publicClient.readContract({
      address: CTF_CORE_ADDRESS,
      abi: CTF_CORE_ABI,
      functionName: "payoutDenominator",
      args: [conditionId as Hex],
    });
    return denom > 0n;
  }

  /** Get token balances for both YES (indexSet=1) and NO (indexSet=2) outcomes. */
  async getBalances(
    conditionId: string
  ): Promise<{ yes: bigint; no: bigint }> {
    const publicClient = this.createPublicClient();
    const holder = this.getHolder();

    const [yesPositionId, noPositionId] = await Promise.all([
      this.getPositionId(publicClient, conditionId, 1n),
      this.getPositionId(publicClient, conditionId, 2n),
    ]);

    const [yes, no] = await Promise.all([
      publicClient.readContract({
        address: CTF_CORE_ADDRESS,
        abi: CTF_CORE_ABI,
        functionName: "balanceOf",
        args: [holder, yesPositionId],
      }),
      publicClient.readContract({
        address: CTF_CORE_ADDRESS,
        abi: CTF_CORE_ABI,
        functionName: "balanceOf",
        args: [holder, noPositionId],
      }),
    ]);

    return { yes, no };
  }

  // ── Redemption ────────────────────────────────────────────────────────────

  /** Redeem winning tokens for a single resolved market. */
  async redeem(conditionId: string): Promise<RedemptionResult> {
    try {
      const resolved = await this.isResolved(conditionId);
      if (!resolved) {
        return { conditionId, success: false, error: "Market not yet resolved" };
      }

      const balances = await this.getBalances(conditionId);
      if (balances.yes === 0n && balances.no === 0n) {
        return { conditionId, success: false, error: "No token balance to redeem" };
      }

      console.log(
        `[Redeemer] Redeeming ${conditionId.slice(0, 10)}… (YES: ${balances.yes}, NO: ${balances.no})`
      );

      const hasProxy =
        this.config.polymarket.proxyWallet &&
        this.config.polymarket.proxyWallet.length > 0;

      const receipt = hasProxy
        ? await this.redeemViaSafe(conditionId)
        : await this.redeemDirect(conditionId);

      if (receipt.status === "reverted") {
        return {
          conditionId,
          success: false,
          txHash: receipt.transactionHash,
          error: "Transaction reverted",
        };
      }

      console.log(
        `[Redeemer] Redeemed ${conditionId.slice(0, 10)}… in block ${receipt.blockNumber} (tx: ${receipt.transactionHash})`
      );

      return {
        conditionId,
        success: true,
        txHash: receipt.transactionHash,
      };
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      console.error(`[Redeemer] Failed to redeem ${conditionId.slice(0, 10)}…: ${msg}`);
      return { conditionId, success: false, error: msg };
    }
  }

  // ── Queue management ──────────────────────────────────────────────────────

  /** Add a conditionId to the pending redemption queue. */
  queueForRedemption(conditionId: string): void {
    if (this.pendingQueue.has(conditionId)) return;
    this.pendingQueue.set(conditionId, {
      conditionId,
      queuedAt: Date.now(),
    });
    console.log(
      `[Redeemer] Queued ${conditionId.slice(0, 10)}… for redemption (${this.pendingQueue.size} pending)`
    );
  }

  /** Process the pending queue: skip too-recent entries, check resolution, redeem if possible. */
  async sweepPending(): Promise<RedemptionResult[]> {
    if (this.pendingQueue.size === 0) return [];

    const now = Date.now();
    const results: RedemptionResult[] = [];
    const pending = Array.from(this.pendingQueue.values());
    let skipped = 0;

    for (const entry of pending) {
      // Don't attempt redemption until enough time has passed for market to resolve
      const elapsed = now - entry.queuedAt;
      if (elapsed < MIN_REDEMPTION_DELAY_MS) {
        const remainingSec = ((MIN_REDEMPTION_DELAY_MS - elapsed) / 1000).toFixed(0);
        skipped++;
        continue;
      }

      console.log(
        `[Redeemer] Attempting redemption for ${entry.conditionId.slice(0, 10)}… (queued ${(elapsed / 1000).toFixed(0)}s ago)`
      );
      const result = await this.redeem(entry.conditionId);
      results.push(result);

      if (result.success || result.error === "No token balance to redeem") {
        this.pendingQueue.delete(entry.conditionId);
      }
      // Keep in queue if not resolved yet or if there was a transient error
    }

    if (skipped > 0) {
      console.log(
        `[Redeemer] Sweep: ${skipped}/${pending.length} entries still waiting for delay`
      );
    }

    return results;
  }

  /** Get the current pending queue size (for logging/monitoring). */
  get pendingCount(): number {
    return this.pendingQueue.size;
  }

  // ── Lifecycle ─────────────────────────────────────────────────────────────

  /** Start the periodic redemption sweep. */
  startSweep(intervalMs: number = DEFAULT_SWEEP_INTERVAL_MS): void {
    if (this.sweepTimer) return;
    this.sweepTimer = setInterval(async () => {
      try {
        const results = await this.sweepPending();
        if (results.length > 0) {
          const ok = results.filter((r) => r.success).length;
          const fail = results.length - ok;
          console.log(`[Redeemer] Sweep complete: ${ok} redeemed, ${fail} failed`);
        }

        // Backfill trade resolutions via Gamma API
        if (this.csvLogger) {
          try {
            const updated = await this.csvLogger.backfillResolutions();
            if (updated > 0) {
              console.log(`[Redeemer] Backfilled ${updated} trade resolution(s)`);
            }
          } catch (err) {
            console.error("[Redeemer] Resolution backfill error:", err);
          }
        }
      } catch (err) {
        console.error("[Redeemer] Sweep error:", err);
      }
    }, intervalMs);
    console.log(`[Redeemer] Sweep started (every ${intervalMs / 1000}s)`);
  }

  /** Stop the periodic sweep. */
  stopSweep(): void {
    if (this.sweepTimer) {
      clearInterval(this.sweepTimer);
      this.sweepTimer = null;
    }
  }

  // ── Private: redemption paths ─────────────────────────────────────────────

  /** Direct EOA call — for non-proxy setups. */
  private async redeemDirect(conditionId: string): Promise<TransactionReceipt> {
    const walletClient = this.createWalletClient();
    const publicClient = this.createPublicClient();

    const hash = await walletClient.writeContract({
      address: CTF_CORE_ADDRESS,
      abi: CTF_CORE_ABI,
      functionName: "redeemPositions",
      args: [
        USDC_E_ADDRESS,
        PARENT_COLLECTION_ID,
        conditionId as Hex,
        [...BINARY_INDEX_SETS],
      ],
    });

    return publicClient.waitForTransactionReceipt({ hash });
  }

  /** Safe execTransaction — for proxy wallet setups. */
  private async redeemViaSafe(conditionId: string): Promise<TransactionReceipt> {
    const walletClient = this.createWalletClient();
    const publicClient = this.createPublicClient();
    const safeAddress = this.config.polymarket.proxyWallet as Hex;

    // 1. Encode the redeemPositions calldata
    const innerData = encodeFunctionData({
      abi: CTF_CORE_ABI,
      functionName: "redeemPositions",
      args: [
        USDC_E_ADDRESS,
        PARENT_COLLECTION_ID,
        conditionId as Hex,
        [...BINARY_INDEX_SETS],
      ],
    });

    // 2. Get the Safe's current nonce
    const safeNonce = await publicClient.readContract({
      address: safeAddress,
      abi: GNOSIS_SAFE_ABI,
      functionName: "nonce",
    });

    // 3. Compute the Safe transaction hash
    const zeroAddr = "0x0000000000000000000000000000000000000000" as Hex;
    const safeTxHash = await publicClient.readContract({
      address: safeAddress,
      abi: GNOSIS_SAFE_ABI,
      functionName: "getTransactionHash",
      args: [
        CTF_CORE_ADDRESS, // to
        0n,               // value
        innerData,        // data
        0,                // operation (CALL)
        0n,               // safeTxGas
        0n,               // baseGas
        0n,               // gasPrice
        zeroAddr,         // gasToken
        zeroAddr,         // refundReceiver
        safeNonce,        // _nonce
      ],
    });

    // 4. Sign the Safe tx hash with the EOA (raw ECDSA, no EIP-191 prefix)
    const account = privateKeyToAccount(
      this.config.polymarket.privateKey as Hex
    );
    const signature = await account.sign({
      hash: safeTxHash as Hex,
    });

    // 5. Submit execTransaction
    const hash = await walletClient.writeContract({
      address: safeAddress,
      abi: GNOSIS_SAFE_ABI,
      functionName: "execTransaction",
      args: [
        CTF_CORE_ADDRESS, // to
        0n,               // value
        innerData,        // data
        0,                // operation (CALL)
        0n,               // safeTxGas
        0n,               // baseGas
        0n,               // gasPrice
        zeroAddr,         // gasToken
        zeroAddr,         // refundReceiver
        signature,        // signatures
      ],
    });

    return publicClient.waitForTransactionReceipt({ hash });
  }

  // ── Private: helpers ──────────────────────────────────────────────────────

  /** Derive the ERC-1155 position ID for a given condition + indexSet. */
  private async getPositionId(
    publicClient: ReturnType<typeof createPublicClient>,
    conditionId: string,
    indexSet: bigint
  ): Promise<bigint> {
    const collectionId = await publicClient.readContract({
      address: CTF_CORE_ADDRESS,
      abi: CTF_CORE_ABI,
      functionName: "getCollectionId",
      args: [PARENT_COLLECTION_ID, conditionId as Hex, indexSet],
    });

    return publicClient.readContract({
      address: CTF_CORE_ADDRESS,
      abi: CTF_CORE_ABI,
      functionName: "getPositionId",
      args: [USDC_E_ADDRESS, collectionId],
    });
  }

  /** Get the address that holds the tokens (Safe if proxy, otherwise EOA). */
  private getHolder(): Hex {
    const proxy = this.config.polymarket.proxyWallet;
    if (proxy && proxy.length > 0) {
      return proxy as Hex;
    }
    const account = privateKeyToAccount(
      this.config.polymarket.privateKey as Hex
    );
    return account.address;
  }

  private createPublicClient() {
    return createPublicClient({
      chain: polygon,
      transport: http(this.config.execution.rpcUrl),
    });
  }

  private createWalletClient() {
    const account = privateKeyToAccount(
      this.config.polymarket.privateKey as Hex
    );
    return createWalletClient({
      account,
      chain: polygon,
      transport: http(this.config.execution.rpcUrl),
    });
  }
}
