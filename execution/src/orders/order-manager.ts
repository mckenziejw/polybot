import { ClobClient } from "@polymarket/clob-client";
import { SignatureType } from "@polymarket/order-utils";
import { Wallet } from "@ethersproject/wallet";
import type {
  AppConfig,
  Side,
  OrderType,
  PlaceOrderAction,
  CancelOrderAction,
} from "../types.ts";

// Map our string OrderType to the clob-client enum
import { OrderType as ClobOrderType, Side as ClobSide } from "@polymarket/clob-client";

export interface OrderResult {
  success: boolean;
  orderId: string;
  errorMsg?: string;
}

/**
 * Wraps @polymarket/clob-client to place and cancel orders on Polymarket's CLOB.
 *
 * Initialization:
 * 1. Create ethers Wallet from private key
 * 2. Instantiate ClobClient with host, chainId, signer, creds
 * 3. If no API creds in config, call createOrDeriveApiKey() to derive L2 credentials
 *
 * SignatureType.POLY_GNOSIS_SAFE (2) is used when a proxy wallet (funder address) is configured.
 * SignatureType.EOA (0) is used otherwise.
 */
export class OrderManager {
  private client!: ClobClient;
  private config: AppConfig;
  private initialized = false;
  private derivedCreds: { key: string; secret: string; passphrase: string } | null = null;

  constructor(config: AppConfig) {
    this.config = config;
  }

  async init(): Promise<void> {
    const pm = this.config.polymarket;

    const signer = new Wallet(pm.privateKey);

    // Determine signature type based on proxy wallet configuration
    const hasProxy = pm.proxyWallet && pm.proxyWallet.length > 0;
    const sigType = hasProxy
      ? SignatureType.POLY_GNOSIS_SAFE
      : SignatureType.EOA;

    // Build creds object from config (used to construct initial client)
    let creds: { key: string; secret: string; passphrase: string } | undefined;
    if (pm.apiKey && pm.apiSecret && pm.apiPassphrase) {
      creds = {
        key: pm.apiKey,
        secret: pm.apiSecret,
        passphrase: pm.apiPassphrase,
      };
    }

    this.client = new ClobClient(
      pm.host,
      pm.chainId,
      signer,
      creds,
      sigType,
      hasProxy ? pm.proxyWallet : undefined
    );

    // Always derive fresh L2 session credentials (config creds alone can't auth)
    console.log("[OrderManager] Deriving API credentials...");
    const derived = await this.client.createOrDeriveApiKey();

    // Reconstruct client with derived creds (creds is readonly on ClobClient)
    this.client = new ClobClient(
      pm.host,
      pm.chainId,
      signer,
      derived,
      sigType,
      hasProxy ? pm.proxyWallet : undefined
    );
    this.derivedCreds = derived;
    console.log("[OrderManager] API credentials derived");

    this.initialized = true;
    console.log("[OrderManager] Initialized");
  }

  private ensureInit(): void {
    if (!this.initialized) {
      throw new Error("OrderManager not initialized. Call init() first.");
    }
  }

  /**
   * Place a limit order on the CLOB.
   * Maps our PlaceOrderAction to the clob-client's UserOrder format.
   */
  async placeOrder(action: PlaceOrderAction): Promise<OrderResult> {
    this.ensureInit();

    const clobSide = action.side === "BUY" ? ClobSide.BUY : ClobSide.SELL;
    const clobOrderType = mapOrderType(action.orderType);

    try {
      const result = await this.client.createAndPostOrder(
        {
          tokenID: action.assetId,
          price: action.price,
          size: action.size,
          side: clobSide,
        },
        undefined,
        clobOrderType
      );

      const orderId = result?.orderID ?? result?.id ?? "";
      const errorMsg = result?.errorMsg;
      // Treat as failure if: explicit success=false, error message present, or no orderId returned
      const success = result?.success !== false && !errorMsg && orderId.length > 0;

      if (!success) {
        console.error(`[OrderManager] Order rejected: ${errorMsg || "no orderId returned"}`);
      } else {
        console.log(
          `[OrderManager] Order placed: ${orderId} ${action.side} ${action.size}@${action.price}`
        );
      }

      return { success, orderId, errorMsg };
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      console.error("[OrderManager] Place order error:", msg);
      return { success: false, orderId: "", errorMsg: msg };
    }
  }

  /** Cancel a single order by ID. */
  async cancelOrder(action: CancelOrderAction): Promise<boolean> {
    this.ensureInit();

    try {
      await this.client.cancelOrder({ orderID: action.orderId });
      console.log(`[OrderManager] Cancelled order: ${action.orderId}`);
      return true;
    } catch (err) {
      console.error(
        `[OrderManager] Cancel failed for ${action.orderId}:`,
        err instanceof Error ? err.message : err
      );
      return false;
    }
  }

  /** Cancel multiple orders by ID. */
  async cancelOrders(orderIds: string[]): Promise<boolean> {
    this.ensureInit();

    if (orderIds.length === 0) return true;

    try {
      await this.client.cancelOrders(orderIds);
      console.log(`[OrderManager] Cancelled ${orderIds.length} orders`);
      return true;
    } catch (err) {
      console.error(
        "[OrderManager] Bulk cancel failed:",
        err instanceof Error ? err.message : err
      );
      return false;
    }
  }

  /** Cancel all open orders. */
  async cancelAll(): Promise<boolean> {
    this.ensureInit();

    try {
      await this.client.cancelAll();
      console.log("[OrderManager] Cancelled all orders");
      return true;
    } catch (err) {
      console.error(
        "[OrderManager] Cancel all failed:",
        err instanceof Error ? err.message : err
      );
      return false;
    }
  }

  /** Get the derived L2 API credentials (available after init). */
  getDerivedCreds(): { key: string; secret: string; passphrase: string } {
    if (!this.derivedCreds) {
      throw new Error("No derived creds available. Call init() first.");
    }
    return this.derivedCreds;
  }

  /** Get the underlying ClobClient (for advanced use / testing). */
  getClient(): ClobClient {
    this.ensureInit();
    return this.client;
  }
}

function mapOrderType(ot: OrderType): ClobOrderType.GTC | ClobOrderType.GTD {
  switch (ot) {
    case "GTC":
      return ClobOrderType.GTC;
    case "GTD":
      return ClobOrderType.GTD;
    case "FOK":
      // FOK is only for market orders; for limit orders, default to GTC
      return ClobOrderType.GTC;
    default:
      return ClobOrderType.GTC;
  }
}
