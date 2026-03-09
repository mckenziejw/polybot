import {
  createWalletClient,
  createPublicClient,
  http,
  parseAbi,
  encodeFunctionData,
  type Hex,
  type TransactionReceipt,
  maxUint256,
} from "viem";
import { polygon } from "viem/chains";
import { privateKeyToAccount } from "viem/accounts";
import type { AppConfig } from "../types.ts";

/** Gnosis ConditionalTokens (CTF Core) — holds positions, handles split/merge/redeem. */
const CTF_CORE_ADDRESS = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045" as const;

/** USDC.e on Polygon — collateral token for Polymarket. */
const USDC_E_ADDRESS = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174" as const;

/** Zero bytes32 — parentCollectionId for non-branching (top-level) markets. */
const PARENT_COLLECTION_ID =
  "0x0000000000000000000000000000000000000000000000000000000000000000" as Hex;

/** Index sets for binary YES/NO markets: YES=1, NO=2. */
const BINARY_PARTITION = [1n, 2n] as const;

const CTF_CORE_ABI = parseAbi([
  "function splitPosition(address collateralToken, bytes32 parentCollectionId, bytes32 conditionId, uint256[] partition, uint256 amount) external",
  "function mergePositions(address collateralToken, bytes32 parentCollectionId, bytes32 conditionId, uint256[] partition, uint256 amount) external",
]);

const ERC20_ABI = parseAbi([
  "function allowance(address owner, address spender) external view returns (uint256)",
  "function approve(address spender, uint256 amount) external returns (bool)",
]);

const GNOSIS_SAFE_ABI = parseAbi([
  "function execTransaction(address to, uint256 value, bytes data, uint8 operation, uint256 safeTxGas, uint256 baseGas, uint256 gasPrice, address gasToken, address refundReceiver, bytes signatures) external payable returns (bool success)",
  "function nonce() external view returns (uint256)",
  "function getTransactionHash(address to, uint256 value, bytes data, uint8 operation, uint256 safeTxGas, uint256 baseGas, uint256 gasPrice, address gasToken, address refundReceiver, uint256 _nonce) external view returns (bytes32)",
]);

export interface SplitResult {
  success: boolean;
  txHash?: string;
  amount: bigint;
  error?: string;
}

export interface MergeResult {
  success: boolean;
  txHash?: string;
  amount: bigint;
  error?: string;
}

/**
 * Handles splitPosition and mergePositions on the CTF Core contract.
 *
 * splitPosition takes USDC.e and mints equal amounts of YES + NO tokens.
 * mergePositions burns equal YES + NO tokens and returns USDC.e.
 *
 * Routes through Gnosis Safe when proxyWallet is configured.
 */
export class Minter {
  private config: AppConfig;
  private approvalChecked = false;

  constructor(config: AppConfig) {
    this.config = config;
  }

  /**
   * Split USDC.e into YES + NO conditional tokens.
   *
   * @param conditionId - The market's condition ID
   * @param amount - Amount in USDC.e smallest units (6 decimals, so 1 USDC = 1_000_000)
   */
  async split(conditionId: string, amount: bigint): Promise<SplitResult> {
    try {
      // Ensure CTF Core has approval to spend USDC.e from the holder
      await this.ensureApproval();

      const innerData = encodeFunctionData({
        abi: CTF_CORE_ABI,
        functionName: "splitPosition",
        args: [
          USDC_E_ADDRESS,
          PARENT_COLLECTION_ID,
          conditionId as Hex,
          [...BINARY_PARTITION],
          amount,
        ],
      });

      console.log(
        `[Minter] Splitting ${amount} USDC.e units into YES+NO for ${conditionId.slice(0, 10)}…`
      );

      const receipt = await this.execViaSafe(CTF_CORE_ADDRESS, innerData);

      if (receipt.status === "reverted") {
        return { success: false, amount, txHash: receipt.transactionHash, error: "Transaction reverted" };
      }

      console.log(
        `[Minter] Split confirmed in block ${receipt.blockNumber} (tx: ${receipt.transactionHash})`
      );

      return { success: true, amount, txHash: receipt.transactionHash };
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      console.error(`[Minter] Split failed: ${msg}`);
      return { success: false, amount, error: msg };
    }
  }

  /**
   * Merge YES + NO conditional tokens back into USDC.e.
   *
   * @param conditionId - The market's condition ID
   * @param amount - Amount in USDC.e smallest units (6 decimals, so 1 pair = 1_000_000)
   */
  async merge(conditionId: string, amount: bigint): Promise<MergeResult> {
    try {
      const innerData = encodeFunctionData({
        abi: CTF_CORE_ABI,
        functionName: "mergePositions",
        args: [
          USDC_E_ADDRESS,
          PARENT_COLLECTION_ID,
          conditionId as Hex,
          [...BINARY_PARTITION],
          amount,
        ],
      });

      console.log(
        `[Minter] Merging ${amount} units of YES+NO back to USDC.e for ${conditionId.slice(0, 10)}…`
      );

      const receipt = await this.execViaSafe(CTF_CORE_ADDRESS, innerData);

      if (receipt.status === "reverted") {
        return { success: false, amount, txHash: receipt.transactionHash, error: "Transaction reverted" };
      }

      console.log(
        `[Minter] Merge confirmed in block ${receipt.blockNumber} (tx: ${receipt.transactionHash})`
      );

      return { success: true, amount, txHash: receipt.transactionHash };
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      console.error(`[Minter] Merge failed: ${msg}`);
      return { success: false, amount, error: msg };
    }
  }

  /**
   * Check and set USDC.e approval for CTF Core if needed.
   * Only needs to run once per session (max approval).
   */
  private async ensureApproval(): Promise<void> {
    if (this.approvalChecked) return;

    const publicClient = this.createPublicClient();
    const holder = this.getHolder();

    const allowance = await publicClient.readContract({
      address: USDC_E_ADDRESS,
      abi: ERC20_ABI,
      functionName: "allowance",
      args: [holder, CTF_CORE_ADDRESS],
    });

    // If allowance is already large enough, skip
    if (allowance > 1_000_000_000_000n) { // > 1M USDC.e
      this.approvalChecked = true;
      return;
    }

    console.log(`[Minter] Setting USDC.e approval for CTF Core...`);

    const approveData = encodeFunctionData({
      abi: ERC20_ABI,
      functionName: "approve",
      args: [CTF_CORE_ADDRESS, maxUint256],
    });

    const receipt = await this.execViaSafe(USDC_E_ADDRESS, approveData);

    if (receipt.status === "reverted") {
      throw new Error("USDC.e approval transaction reverted");
    }

    console.log(`[Minter] Approval set (tx: ${receipt.transactionHash})`);
    this.approvalChecked = true;
  }

  /** Execute a transaction through the Gnosis Safe. */
  private async execViaSafe(to: Hex, data: Hex): Promise<TransactionReceipt> {
    const walletClient = this.createWalletClient();
    const publicClient = this.createPublicClient();
    const safeAddress = this.config.polymarket.proxyWallet as Hex;

    const safeNonce = await publicClient.readContract({
      address: safeAddress,
      abi: GNOSIS_SAFE_ABI,
      functionName: "nonce",
    });

    const zeroAddr = "0x0000000000000000000000000000000000000000" as Hex;
    const safeTxHash = await publicClient.readContract({
      address: safeAddress,
      abi: GNOSIS_SAFE_ABI,
      functionName: "getTransactionHash",
      args: [to, 0n, data, 0, 0n, 0n, 0n, zeroAddr, zeroAddr, safeNonce],
    });

    const account = privateKeyToAccount(this.config.polymarket.privateKey as Hex);
    const signature = await account.sign({ hash: safeTxHash as Hex });

    const hash = await walletClient.writeContract({
      address: safeAddress,
      abi: GNOSIS_SAFE_ABI,
      functionName: "execTransaction",
      args: [to, 0n, data, 0, 0n, 0n, 0n, zeroAddr, zeroAddr, signature],
    });

    return publicClient.waitForTransactionReceipt({ hash });
  }

  private getHolder(): Hex {
    const proxy = this.config.polymarket.proxyWallet;
    if (proxy && proxy.length > 0) return proxy as Hex;
    const account = privateKeyToAccount(this.config.polymarket.privateKey as Hex);
    return account.address;
  }

  private createPublicClient() {
    return createPublicClient({
      chain: polygon,
      transport: http(this.config.execution.rpcUrl),
    });
  }

  private createWalletClient() {
    const account = privateKeyToAccount(this.config.polymarket.privateKey as Hex);
    return createWalletClient({
      account,
      chain: polygon,
      transport: http(this.config.execution.rpcUrl),
    });
  }
}
