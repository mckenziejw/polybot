import {
  createWalletClient,
  createPublicClient,
  http,
  parseAbi,
  type Hex,
  type TransactionReceipt,
} from "viem";
import { polygon } from "viem/chains";
import { privateKeyToAccount } from "viem/accounts";
import type { AppConfig } from "../types.ts";

/**
 * CTFExchange contract on Polygon.
 * Calling incrementNonce() invalidates ALL open orders for the caller.
 * This is the nuclear cancellation option — costs gas, but guarantees
 * no orders survive even if the CLOB API is unreachable.
 */
const CTF_EXCHANGE_ADDRESS = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E" as const;

const CTF_EXCHANGE_ABI = parseAbi([
  "function incrementNonce() external",
  "function nonces(address) external view returns (uint256)",
]);

export class NonceCanceller {
  private config: AppConfig;

  constructor(config: AppConfig) {
    this.config = config;
  }

  /**
   * Call incrementNonce() on the CTFExchange contract.
   * This invalidates all open orders for the signing wallet.
   *
   * Returns the transaction receipt on success, or throws on failure.
   */
  async incrementNonce(): Promise<TransactionReceipt> {
    const rpcUrl = this.config.execution.rpcUrl;
    const privateKey = this.config.polymarket.privateKey as Hex;

    const account = privateKeyToAccount(privateKey);

    const walletClient = createWalletClient({
      account,
      chain: polygon,
      transport: http(rpcUrl),
    });

    const publicClient = createPublicClient({
      chain: polygon,
      transport: http(rpcUrl),
    });

    console.log(
      `[NonceCanceller] Sending incrementNonce() from ${account.address}...`
    );

    const hash = await walletClient.writeContract({
      address: CTF_EXCHANGE_ADDRESS,
      abi: CTF_EXCHANGE_ABI,
      functionName: "incrementNonce",
    });

    console.log(`[NonceCanceller] Tx submitted: ${hash}`);

    const receipt = await publicClient.waitForTransactionReceipt({ hash });

    if (receipt.status === "reverted") {
      throw new Error(`incrementNonce() reverted. Tx: ${hash}`);
    }

    console.log(
      `[NonceCanceller] incrementNonce() confirmed in block ${receipt.blockNumber}`
    );

    return receipt;
  }

  /** Read the current nonce for the signing wallet (useful for debugging). */
  async getCurrentNonce(): Promise<bigint> {
    const rpcUrl = this.config.execution.rpcUrl;
    const privateKey = this.config.polymarket.privateKey as Hex;
    const account = privateKeyToAccount(privateKey);

    const publicClient = createPublicClient({
      chain: polygon,
      transport: http(rpcUrl),
    });

    return publicClient.readContract({
      address: CTF_EXCHANGE_ADDRESS,
      abi: CTF_EXCHANGE_ABI,
      functionName: "nonces",
      args: [account.address],
    });
  }
}
