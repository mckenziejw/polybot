# Polymarket Technical Architecture Guide

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Core Concepts](#2-core-concepts)
3. [Token Operations](#3-token-operations)
4. [Market Settlement](#4-market-settlement)
5. [Technical Architecture](#5-technical-architecture)
6. [Trading Flow](#6-trading-flow)
7. [API Interfaces](#7-api-interfaces)
8. [Smart Contracts](#8-smart-contracts)
9. [Development Guide](#9-development-guide)
10. [Advanced Topics](#10-advanced-topics)
11. [References](#11-references)

---

## 1. System Overview

Polymarket is the world's largest decentralized prediction market platform, built on the **Polygon network** with a **hybrid decentralized architecture**.

### 1.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Polymarket Ecosystem                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    │
│  │  Frontend   │    │  CLOB System│    │  Data API   │    │
│  │  (Web/App)  │◄──►│(Order Matching)│◄──►│ (Market Data)│   │
│  └─────────────┘    └──────┬──────┘    └─────────────┘    │
│                            │                               │
│                            ▼                               │
│                     ┌─────────────┐                       │
│                     │Smart Contract│                       │
│                     │  (Polygon)  │                       │
│                     └──────┬──────┘                       │
│                            │                               │
│                            ▼                               │
│                     ┌─────────────┐                       │
│                     │  CTF Protocol│                       │
│                     │ (ERC-1155)  │                       │
│                     └─────────────┘                       │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Core Components

| Component | Function | Tech Stack |
|-----------|----------|------------|
| **CLOB** | Central Limit Order Book for order matching | Off-chain matching |
| **CTF** | Conditional Token Framework for prediction market tokens | ERC-1155 |
| **Data API** | Provides market data and trading history | REST API |
| **Smart Contracts** | Custody and final settlement | Polygon (PoS) |

---

## 2. Core Concepts

### 2.1 Conditional Token Framework (CTF)

Polymarket uses the **Gnosis CTF protocol** to tokenize prediction market outcomes.

#### ERC-1155 Multi-Token Standard

```typescript
// ERC-1155 allows managing multiple tokens in a single contract
interface IERC1155 {
    function safeBatchTransferFrom(
        address from,
        address to,
        uint256[] calldata ids,
        uint256[] calldata values,
        bytes calldata data
    ) external;
}
```

### 2.2 YES/NO Tokens

Each binary prediction market has two outcome tokens:

```
Market Example: "Will Trump be elected US President in 2024?"

┌─────────────────────────────────────┐
│     Conditional Token (Condition ID) │
│   0x1234...abcd (Unique Identifier)  │
├─────────────────────────────────────┤
│                                     │
│  ┌──────────┐      ┌──────────┐    │
│  │ YES Token│      │ NO Token │    │
│  │ Token ID:│      │ Token ID:│    │
│  │ 1234...1 │      │ 1234...0 │    │
│  └──────────┘      └──────────┘    │
│                                     │
│  Price Range: $0.00 - $1.00        │
│  YES + NO = $1.00 (Constant)        │
└─────────────────────────────────────┘
```

### 2.3 Condition ID

Condition ID is the unique identifier for a prediction market:

```typescript
import { ethers } from 'ethers';

function calculateConditionId(
    oracleAddress: string,
    questionHash: string,
    outcomeSlotCount: number  // Binary market=2, Multiple choice=3+
): string {
    const encoded = ethers.utils.defaultAbiCoder.encode(
        ['address', 'bytes32', 'uint256'],
        [oracleAddress, questionHash, outcomeSlotCount]
    );
    return ethers.utils.keccak256(encoded);
}
```

### 2.4 Fund Flow

```
USDC (Collateral) → CTF Main Token → Conditional Tokens (YES/NO) → Trade/Redeem
```

---

## 3. Token Operations

### 3.1 Two Ways to Open Positions

```
┌─────────────────────────────────────────────────────────────┐
│      Method 1: Direct Purchase (Regular Traders) ⭐Recommended│
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  USDC ──► CLOB Order Book ──► YES or NO                    │
│                                                             │
│  For: Regular users, most traders                           │
│  Features: Direct trading on order book, tokens from other users │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│           Method 2: Split Position (Market Makers/LP)       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  USDC ──► split() ──► YES + NO (1:1 mint)                  │
│                                                             │
│  For: Market makers, liquidity providers, arbitrageurs      │
│  Features: Interact with CTF contract, mint token pairs yourself │
└─────────────────────────────────────────────────────────────┘
```

> **Note**: Most users use Method 1 for direct trading without splitting themselves.

### 3.2 Split Operation

Split USDC into YES + NO token pairs:

```typescript
async function splitPosition(conditionId: string, amount: bigint) {
    const ctf = new ethers.Contract(CTF_CORE, CTF_ABI, wallet);

    // Approve USDC first
    await usdc.approve(CTF_CORE, amount);

    // Split: 1 USDC → 1 YES + 1 NO
    const tx = await ctf.splitPosition(
        USDC_ADDRESS,
        ethers.constants.HashZero,  // parentCollectionId
        conditionId,
        [0b01, 0b10],               // partition (YES=01, NO=10)
        amount
    );
    await tx.wait();
}
```

### 3.3 Merge Operation

Merge YES + NO token pairs back to USDC:

```typescript
async function mergePosition(conditionId: string, amount: bigint) {
    const ctf = new ethers.Contract(CTF_CORE, CTF_ABI, wallet);

    // Merge: 1 YES + 1 NO → 1 USDC
    const tx = await ctf.mergePositions(
        USDC_ADDRESS,
        ethers.constants.HashZero,
        conditionId,
        [0b01, 0b10],
        amount
    );
    await tx.wait();
}
```

### 3.4 Operation Timing

```
Market Lifecycle:

Created ─────────────── Trading ─────────────── Settled
  │                      │                      │
  │  ✅ Can split       │  ✅ Can split       │  ❌ Cannot split
  │  ✅ Can merge       │  ✅ Can merge       │  ✅ Only redeem
```

### 3.5 Arbitrage Mechanism

Split/Merge is the core mechanism maintaining YES + NO = $1.00:

```
┌─────────────────────────────────────────────────────────────┐
│                   Price Arbitrage Principle                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Case 1: YES($0.60) + NO($0.35) = $0.95 < $1.00            │
│  ─────────────────────────────────────────────────         │
│  Arbitrage:                                                 │
│    1. Buy YES @ $0.60 + Buy NO @ $0.35                     │
│    2. Call merge() to convert to USDC                      │
│    3. Get $1.00, net profit $0.05                          │
│                                                             │
│  Case 2: YES($0.70) + NO($0.35) = $1.05 > $1.00            │
│  ─────────────────────────────────────────────────         │
│  Arbitrage:                                                 │
│    1. Call split() to convert $1 USDC to YES + NO          │
│    2. Sell YES @ $0.70 + Sell NO @ $0.35                   │
│    3. Get $1.05, net profit $0.05                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

| Operation | Use Case | Description |
|-----------|----------|-------------|
| **Split** | Market makers add liquidity | Mint new token pairs |
| **Split** | Arbitrage (YES+NO > $1) | Split and sell both sides |
| **Merge** | Arbitrage (YES+NO < $1) | Buy both sides and merge |
| **Merge** | Exit position early | When holding both YES+NO |

---

## 4. Market Settlement

### 4.1 Market Lifecycle

```
┌─────────────────────────────────────────────────────────────┐
│                    Market Lifecycle                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Creation   2. Trading     3. Settlement                │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐                   │
│  │ Oracle  │   │  CLOB   │   │ Oracle  │                   │
│  │Create   │───►│Buy/Sell │───►│Report   │                   │
│  │Condition│   │YES/NO   │   │Result   │                   │
│  └─────────┘   └─────────┘   └────┬────┘                   │
│                                   │                         │
│                                   ▼                         │
│                             ┌─────────┐                    │
│                             │  Users  │                    │
│                             │Redeem   │                    │
│                             │Tokens   │                    │
│                             └─────────┘                    │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Oracle Reports Results

Polymarket uses **UMA Optimistic Oracle** to determine market outcomes:

```typescript
// Oracle calls reportPayouts function
function reportPayouts(
    bytes32 questionId,
    uint256[] calldata payouts  // [NO payout, YES payout]
) external;

// Examples
reportPayouts(questionId, [0, 1]);  // YES wins
reportPayouts(questionId, [1, 0]);  // NO wins
reportPayouts(questionId, [1, 1]);  // Tie/Refund
```

### 4.3 Token Value Changes

```
Before Settlement:
┌──────────────────────────────────────┐
│  YES Token: $0.65 (Market Price)     │
│  NO Token:  $0.35 (Market Price)     │
│  YES + NO = $1.00                    │
└──────────────────────────────────────┘

After Settlement (Assuming YES Wins):
┌──────────────────────────────────────┐
│  YES Token: $1.00 ✅ (Redeemable)    │
│  NO Token:  $0.00 ❌ (Worthless)     │
└──────────────────────────────────────┘
```

### 4.4 Settlement Data Storage

CTF Core contract uses two mappings to store settlement data:

```solidity
contract ConditionalTokens {
    // conditionId => denominator (0=unsettled, >0=settled)
    mapping(bytes32 => uint256) public payoutDenominator;

    // conditionId => outcomeIndex => numerator
    // outcomeIndex: 0=NO, 1=YES
    mapping(bytes32 => uint256[]) public payoutNumerators;
}
```

**Redemption Amount = Holdings × (Numerator / Denominator)**

### 4.5 Settlement Scenarios

| Scenario | payouts Parameter | Denominator | Result |
|----------|-------------------|-------------|--------|
| YES Wins | [0, 1] | 1 | NO=0%, YES=100% |
| NO Wins | [1, 0] | 1 | NO=100%, YES=0% |
| 50-50 Tie | [1, 1] | 2 | NO=50%, YES=50% |
| 70-30 Partial | [3, 7] | 10 | NO=30%, YES=70% |

### 4.6 Redeem Tokens

```typescript
// Check if market is settled
async function isMarketResolved(conditionId: string): Promise<boolean> {
    const ctf = new ethers.Contract(CTF_CORE, CTF_ABI, provider);
    const payoutDenominator = await ctf.payoutDenominator(conditionId);
    return payoutDenominator.gt(0);
}

// Redeem tokens
async function redeemTokens(conditionId: string) {
    const ctf = new ethers.Contract(CTF_CORE, CTF_ABI, wallet);

    const tx = await ctf.redeemPositions(
        USDC_ADDRESS,
        ethers.constants.HashZero,
        conditionId,
        [0b01, 0b10]  // Redeem YES and NO
    );
    await tx.wait();
}
```

### 4.7 UMA Dispute Resolution

```
┌─────────────────────────────────────────────────────────────┐
│                   UMA Dispute Resolution Flow               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Oracle Submits Result                                   │
│       │                                                     │
│       ▼                                                     │
│  2. Dispute Period (Usually 2-4 hours)                      │
│       │                                                     │
│       ├─► No dispute ──► Result生效                         │
│       │                                                     │
│       └─► Dispute raised ──► Post bond                     │
│               │                                             │
│               ▼                                             │
│          3. UMA DVM Voting                                 │
│               │                                             │
│               ├─► Original result correct ──► Disputer loses bond │
│               │                                             │
│               └─► Original result wrong ──► Correct result │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 4.8 Complete Fund Flow Example

```
┌─────────────────────────────────────────────────────────────┐
│                    Complete Fund Flow                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  [Position Opening - Market Maker Creates Liquidity]        │
│  MM: 100 USDC ──► split ──► 100 YES + 100 NO               │
│                                                             │
│  [Trading Phase - On CLOB Order Book]                      │
│  MM: Place sell order 100 NO @ $0.35                        │
│  UserB: Buy 100 NO @ $0.35 ──► Pay 35 USDC                 │
│                                                             │
│  [Settlement Phase - YES Wins]                              │
│  MM: 100 YES ──► redeem ──► 100 USDC ✅                     │
│  UserB: 100 NO ──► redeem ──► 0 USDC ❌                     │
│                                                             │
│  [Final P&L]                                                │
│  MM: Invested 100, Sold NO for 35, Redeemed YES for 100 = Net +35 USDC │
│  UserB: Bought NO for 35, Redeemed NO for 0 = Net -35 USDC │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. Technical Architecture

### 5.1 Three-Layer Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Application Layer                        │
│  • Web Frontend (React)                                     │
│  • Mobile Apps                                              │
│  • Third-party Applications                                 │
├─────────────────────────────────────────────────────────────┤
│                      Service Layer                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │  CLOB API   │  │  Data API   │  │ Gamma API   │        │
│  │ Order Mgmt  │  │ Market Data │  │ Market Info │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
├─────────────────────────────────────────────────────────────┤
│                      Protocol Layer                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │ CTF Exchange│  │  CTF Core   │  │  USDC Token │        │
│  │ Order Match │  │ Cond. Token │  │ Payment     │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 Network Topology

```
User Wallet (EOA/Gnosis Safe)
        │
        ├─► Polygon RPC (Infura/Alchemy)
        │       │
        │       └─► CLOB HTTP API (https://clob.polymarket.com/)
        │               │
        │               ├─► POST /order (create order)
        │               ├─► DELETE /order (cancel order)
        │               ├─► GET /orderbook (query order book)
        │               └─► WebSocket (real-time subscription)
        │
        └─► Data API (https://data-api.polymarket.com/)
                │
                ├─► GET /markets (market list)
                ├─► GET /positions (user positions)
                └─► GET /prices (price history)
```

---

## 6. Trading Flow

### 6.1 Trading Flow Overview

```typescript
async function executeTrade() {
    // 1. Initialize CLOB client
    const clobClient = await createClobClient();

    // 2. Get order book
    const orderBook = await clobClient.getOrderBook(tokenId);

    // 3. Find best price
    const bestAsk = orderBook.asks[0];

    // 4. Create market order
    const signedOrder = await clobClient.createMarketOrder({
        side: Side.BUY,
        tokenID: tokenId,
        amount: usdAmount,
        price: parseFloat(bestAsk.price),
    });

    // 5. Submit order
    const response = await clobClient.postOrder(signedOrder, OrderType.FOK);
}
```

### 6.2 Detailed Flowchart

```
┌──────────────────────────────────────────────────────────────┐
│                    1. Order Creation Phase                   │
└──────────────────────────────────────────────────────────────┘

User initiates trade
    │
    ├─ Wallet signature (choose by account type)
    │  ├─ SignatureType.EOA (0) - Direct EOA trading
    │  ├─ SignatureType.POLY_PROXY (1) - Magic/Email login
    │  └─ SignatureType.POLY_GNOSIS_SAFE (2) - Browser wallet
    │
    ├─ Generate API Key
    │  ├─ createApiKey()  // First time use
    │  └─ deriveApiKey()  // Derive from private key
    │
    └─ Create signed order
       ├─ createMarketOrder()  // Market order
       └─ createLimitOrder()   // Limit order

┌──────────────────────────────────────────────────────────────┐
│                    2. Order Submission Phase                 │
└──────────────────────────────────────────────────────────────┘

Signed order
    │
    ├─ Select order type
    │  ├─ OrderType.FOK (Fill-Or-Kill) - Fill all or cancel
    │  ├─ OrderType.IOC (Immediate-Or-Cancel) - Partial fill
    │  └─ OrderType.Limit - Limit order
    │
    └─ POST to CLOB API
       └─ Endpoint: https://clob.polymarket.com/order

┌──────────────────────────────────────────────────────────────┐
│                    3. Order Matching Phase                   │
└──────────────────────────────────────────────────────────────┘

CLOB Server (Off-chain Matching Engine)
    │
    ├─ Order book matching
    │  ├─ Asks (Sell orders): Sorted low to high
    │  └─ Bids (Buy orders): Sorted high to low
    │
    └─ Return result
       ├─ success: true  → Trade successful
       └─ success: false → Trade failed

┌──────────────────────────────────────────────────────────────┐
│                    4. On-chain Settlement Phase              │
└──────────────────────────────────────────────────────────────┘

After successful match
    │
    ├─ Call CTF Exchange contract
    │
    └─ Fund transfer
       ├─ Buyer: USDC → CTF Token (YES/NO)
       └─ Seller: CTF Token (YES/NO) → USDC
```

### 6.3 Order Request Format

```typescript
POST /order

{
  "order": {
    "maker": "0x...",           // Seller address
    "taker": "0x0000...0000",   // 0x0 means anyone
    "token_id": "54708...",     // CTF token ID
    "maker_amount": "5000000",  // 5 USDC (6 decimals)
    "taker_amount": "10000000", // 10 YES tokens
    "expiration": "1735689600", // Unix timestamp
    "salt": "12345",            // Random number
    "signature": "0x..."        // EIP-712 signature
  }
}
```

---

## 7. API Interfaces

### 7.1 CLOB API

**Base Endpoints**

```typescript
const CLOB_HTTP_URL = 'https://clob.polymarket.com/';
const CLOB_WS_URL = 'wss://ws-subscriptions-clob.polymarket.com/ws';
```

**Create Order**

```typescript
POST /order
Headers: { Authorization: <api-key> }

Response: { "success": true, "order_hash": "0x..." }
```

**Query Order Book**

```typescript
GET /orderbook?token_id=<token_id>

Response:
{
  "token_id": "...",
  "asks": [{ "price": "0.65", "size": "1000" }],
  "bids": [{ "price": "0.63", "size": "2000" }]
}
```

**Cancel Order**

```typescript
DELETE /order
Body: { "order_id": "0x..." }
```

**WebSocket Subscription**

```typescript
const ws = new WebSocket(CLOB_WS_URL);

ws.send(JSON.stringify({
  type: 'subscriptions',
  channel: 'order_book_updates',
  payload: { token_ids: ['<token_id>'] }
}));
```

### 7.2 Data API

**Base Endpoint**

```typescript
const DATA_API_URL = 'https://data-api.polymarket.com/';
```

**Get Market List**

```typescript
GET /markets?active=true&limit=100

Response:
[{
  "condition_id": "0x...",
  "question": "Will Bitcoin reach $100k?",
  "outcome_prices": [0.65, 0.35],
  "volume": "500000",
  "active": true
}]
```

**Get User Positions**

```typescript
GET /positions?user=<wallet_address>

Response:
[{
  "condition_id": "0x...",
  "side": "YES",
  "size": 1000,
  "avg_price": 0.65,
  "current_price": 0.70,
  "unrealized_pnl": 50
}]
```

**Get Trade History**

```typescript
GET /markets/<condition_id>/trades?limit=50

Response:
[{
  "transaction_hash": "0x...",
  "side": "BUY",
  "size": 100,
  "price": 0.65
}]
```

### 7.3 Gamma API

```typescript
const GAMMA_API_URL = 'https://gamma-api.polymarket.com/';

GET /markets   // Market information
GET /prices    // Price history
```

---

## 8. Smart Contracts

### 8.1 Contract Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   CTF Smart Contract Ecosystem              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌────────────────────────────────────────────────────┐    │
│  │      CTF Exchange (Binary Market Trading Contract)  │    │
│  │   • Process CTF token <-> USDC atomic swaps        │    │
│  │   • Used for YES/NO binary markets                 │    │
│  └────────────────────────────────────────────────────┘    │
│                           │                                │
│  ┌────────────────────────────────────────────────────┐    │
│  │  Neg Risk CTF Exchange (Multi-choice Market Contract)│   │
│  │   • Handle multiple outcome markets (3+ results)   │    │
│  │   • Works with Neg Risk Adapter                   │    │
│  └────────────────────────────────────────────────────┘    │
│                           │                                │
│  ┌────────────────────────────────────────────────────┐    │
│  │           CTF Core (Conditional Token Core)         │    │
│  │   • Deploy conditional tokens (ERC-1155)           │    │
│  │   • split/merge position                           │    │
│  └────────────────────────────────────────────────────┘    │
│                           │                                │
│  ┌────────────────────────────────────────────────────┐    │
│  │           Oracle Manager (Oracle Management)        │    │
│  │   • Manage UMA oracle                              │    │
│  │   • Submit and verify market results               │    │
│  └────────────────────────────────────────────────────┘    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 8.2 Contract Addresses (Polygon)

```typescript
// USDC.e (Polygon bridged USDC, used by Polymarket)
const USDC_ADDRESS = '0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174';

// CTF Exchange (Binary markets)
const CTF_EXCHANGE = '0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E';

// CTF Core (Conditional token core)
const CTF_CORE = '0x4D97DCd97eC945f40cF65F87097ACe5EA0476045';

// Neg Risk CTF Exchange (Multi-choice markets)
const NEG_RISK_CTF_EXCHANGE = '0xC5d563A36AE78145C45a50134d48A1215220f80a';

// Neg Risk Adapter (Multi-choice market adapter)
const NEG_RISK_ADAPTER = '0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296';
```

### 8.3 Contract Interaction Example

```typescript
import { ethers } from 'ethers';

const provider = new ethers.providers.JsonRpcProvider(RPC_URL);
const wallet = new ethers.Wallet(PRIVATE_KEY, provider);

const ctfContract = new ethers.Contract(CTF_CORE, CTF_ABI, wallet);

// Query token balance
async function getTokenBalance(tokenId: string): Promise<bigint> {
  return await ctfContract.balanceOf(wallet.address, tokenId);
}
```

### 8.4 Exchange vs Core Relationship

```
┌─────────────────────────────────────────────────────────────────┐
│                     CTF Exchange (Trading Settlement Layer)     │
│  • Order matching settlement                                    │
│  • Only Operator can call trading functions                     │
│  • Process USDC ↔ CTF Token atomic swaps                        │
├─────────────────────────────────────────────────────────────────┤
│                          │ Internal Call                        │
│                          ▼                                     │
├─────────────────────────────────────────────────────────────────┤
│                     CTF Core (Underlying Protocol Layer)        │
│  • ERC-1155 token management                                    │
│  • split() / merge() / redeemPositions()                        │
│  • Anyone can call directly                                     │
└─────────────────────────────────────────────────────────────────┘
```

**Simply put**: Core is the underlying token protocol (permissionless), Exchange is the trading matching layer built on top (requires Operator permission).

### 8.5 Exchange Contract Functions

#### Trading Functions (Operator Only)

| Function | Parameters | Purpose |
|----------|------------|---------|
| `fillOrder` | `(Order order, uint256 fillAmount)` | Fill single order |
| `fillOrders` | `(Order[] orders, uint256[] fillAmounts)` | Batch fill multiple orders |
| `matchOrders` | `(Order takerOrder, Order[] makerOrders, uint256 takerFillAmount, uint256[] makerFillAmounts)` | Match taker with maker orders |

#### Admin Functions (Admin Only)

| Function | Parameters | Purpose |
|----------|------------|---------|
| `registerToken` | `(uint256 token, uint256 complement, bytes32 conditionId)` | **Register new market token pairs** |
| `pauseTrading` | `()` | Pause trading |
| `unpauseTrading` | `()` | Resume trading |
| `setProxyFactory` | `(address newProxyFactory)` | Set proxy wallet factory |
| `setSafeFactory` | `(address newSafeFactory)` | Set Safe wallet factory |

### 8.6 Order Matching Scenarios

Exchange contract supports three matching modes:

```
Scenario 1: NORMAL (Regular Trading)
═══════════════════════════════════════════════════════════════
UserA Buy 100 YES @ $0.50  ←→  UserB Sell 50 YES @ $0.50

Flow:
  1. UserB's 50 YES tokens → Exchange
  2. UserA's 25 USDC → Exchange
  3. Exchange transfers 50 YES → UserA
  4. Exchange transfers 25 USDC → UserB

Scenario 2: MINT (Mint Matching)
═══════════════════════════════════════════════════════════════
UserA Buy YES @ $0.50  ←→  UserB Buy NO @ $0.50

Flow:
  1. UserA's 25 USDC → Exchange
  2. UserB's 25 USDC → Exchange
  3. Exchange calls CTF.split() to mint 50 YES + 50 NO
  4. 50 YES → UserA
  5. 50 NO → UserB

Scenario 3: MERGE (Merge Matching)
═══════════════════════════════════════════════════════════════
UserA Sell 50 YES @ $0.50  ←→  UserB Sell 50 NO @ $0.50

Flow:
  1. UserA's 50 YES → Exchange
  2. UserB's 50 NO → Exchange
  3. Exchange calls CTF.merge() to merge into 50 USDC
  4. 25 USDC → UserA
  5. 25 USDC → UserB
```

### 8.7 Complete Flow for Adding New Market

```
1. CTF Core: prepareCondition(oracle, questionId, outcomeSlotCount)
   │  └─► Generate conditionId
   │
   ▼
2. CTF Exchange: registerToken(tokenId, complementId, conditionId)
   │  └─► Admin calls, register YES/NO token pair to Exchange
   │
   ▼
3. CLOB Backend: Associate market metadata
   │  └─► Title, description, end time, etc.
   │
   ▼
4. Market launches, trading begins
```

### 8.8 Directly Call CTF Core (Bypass Exchange/CLOB)

CTF Core is the underlying protocol, **anyone can call directly**, without going through Exchange or CLOB.

#### Trading Method Comparison

```
┌─────────────────────────────────────────────────────────────────┐
│                    Trading Method Comparison                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Method 1: Standard Polymarket Flow                             │
│  ════════════════════════════                                   │
│  User → CLOB API → Operator → Exchange → CTF Core               │
│  ✅ Has liquidity  ✅ Price discovery  ❌ Requires API Key       │
│                                                                  │
│  Method 2: Directly Call CTF Core                               │
│  ════════════════════════════                                   │
│  User → CTF Core (split/merge)                                  │
│  ✅ Permissionless  ❌ Only mint/merge  ❌ Cannot trade with others│
│                                                                  │
│  Method 3: P2P Transfer + Off-chain Negotiation                 │
│  ════════════════════════════                                   │
│  UserA ──(CTF.safeTransferFrom)──► UserB                        │
│  UserB ──(USDC.transfer)──► UserA                               │
│  ✅ Permissionless  ❌ Non-atomic  ❌ Trust required             │
│                                                                  │
│  Method 4: Self-built Swap Contract                             │
│  ════════════════════════════                                   │
│  User → Your Contract → CTF Core                                │
│  ✅ Atomic swap  ✅ Permissionless  ✅ Custom logic              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### CTF Core Directly Callable Functions

```solidity
// 1. Mint token pairs: 1 USDC → 1 YES + 1 NO
function splitPosition(
    IERC20 collateralToken,      // USDC address
    bytes32 parentCollectionId,  // Usually 0x0
    bytes32 conditionId,         // Market condition ID
    uint256[] partition,         // [0b01, 0b10] = YES, NO
    uint256 amount               // USDC amount
) external;

// 2. Merge token pairs: 1 YES + 1 NO → 1 USDC
function mergePositions(
    IERC20 collateralToken,
    bytes32 parentCollectionId,
    bytes32 conditionId,
    uint256[] partition,
    uint256 amount
) external;

// 3. Redeem after settlement
function redeemPositions(
    IERC20 collateralToken,
    bytes32 parentCollectionId,
    bytes32 conditionId,
    uint256[] indexSets
) external;

// 4. ERC-1155 standard transfer
function safeTransferFrom(
    address from,
    address to,
    uint256 id,        // tokenId
    uint256 amount,
    bytes data
) external;
```

#### Self-built Atomic Swap Contract Example

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/token/ERC1155/IERC1155.sol";

/// @title SimpleCtfSwap
/// @notice Simple CTF token atomic swap contract
contract SimpleCtfSwap {
    IERC20 public immutable usdc;
    IERC1155 public immutable ctf;

    constructor(address _usdc, address _ctf) {
        usdc = IERC20(_usdc);
        ctf = IERC1155(_ctf);
    }

    /// @notice Atomic swap: USDC ↔ CTF Token
    /// @dev Both parties need to approve this contract first
    /// @param buyer Person buying CTF (paying USDC)
    /// @param seller Person selling CTF (paying CTF Token)
    /// @param tokenId YES or NO tokenId
    /// @param tokenAmount CTF token amount
    /// @param usdcAmount USDC amount (6 decimals)
    function swap(
        address buyer,
        address seller,
        uint256 tokenId,
        uint256 tokenAmount,
        uint256 usdcAmount
    ) external {
        // Atomicity: all succeed or all revert
        usdc.transferFrom(buyer, seller, usdcAmount);
        ctf.safeTransferFrom(seller, buyer, tokenId, tokenAmount, "");
    }
}
```

#### Advanced Use Cases

| Use Case | Implementation | Advantage |
|----------|---------------|-----------|
| **Arbitrage Bot** | Directly call CTF Core's split/merge | Arbitrage when YES+NO deviates from $1.00 |
| **OTC Block Trading** | Self-built contract atomic settlement | Avoid slippage on CLOB |
| **Cross-platform Arbitrage** | ERC-1155 trading on any DEX | CTF tokens can be traded on OpenSea etc |
| **Self-built Market Making** | No dependency on Polymarket CLOB | Provide liquidity directly on-chain |

#### Limitations

| Bypass Method | Feasibility | Description |
|---------------|-------------|-------------|
| Bypass CLOB API | ✅ Possible | Lose order book liquidity and price discovery |
| Bypass Exchange | ✅ Possible | Need to implement atomic swap logic yourself |
| Bypass CTF Core | ❌ Not possible | Tokens stored in CTF Core, cannot bypass |

---

## 9. Development Guide

### 9.1 Project Initialization

```bash
npm install @polymarket/clob-client ethers
cp .env.example .env
```

### 9.2 Create CLOB Client

```typescript
import { ClobClient } from '@polymarket/clob-client';
import { SignatureType } from '@polymarket/order-utils';

async function createClient() {
  const chainId = 137;  // Polygon
  const host = 'https://clob.polymarket.com/';
  const wallet = new ethers.Wallet(PRIVATE_KEY);

  // Detect wallet type
  const isGnosisSafe = await checkIfGnosisSafe(wallet.address);
  const signatureType = isGnosisSafe
    ? SignatureType.POLY_GNOSIS_SAFE
    : SignatureType.EOA;

  // Create client
  const clobClient = new ClobClient(
    host, chainId, wallet, undefined,
    signatureType,
    isGnosisSafe ? wallet.address : undefined
  );

  // Set API key
  const creds = await clobClient.createApiKey();

  // Recreate client (with credentials)
  return new ClobClient(
    host, chainId, wallet, creds,
    signatureType,
    isGnosisSafe ? wallet.address : undefined
  );
}
```

### 9.3 Execute Trade

```typescript
import { Side, OrderType } from '@polymarket/clob-client';

async function placeBuyOrder(clobClient, tokenId, usdAmount) {
  const orderBook = await clobClient.getOrderBook(tokenId);
  const bestAsk = orderBook.asks[0];

  const signedOrder = await clobClient.createMarketOrder({
    side: Side.BUY,
    tokenID: tokenId,
    amount: usdAmount,
    price: parseFloat(bestAsk.price),
  });

  return await clobClient.postOrder(signedOrder, OrderType.FOK);
}
```

### 9.4 Error Handling

```typescript
const errorTypes = {
  INSUFFICIENT_BALANCE: 'not enough balance',
  INSUFFICIENT_ALLOWANCE: 'allowance',
  PRICE_SLIPPAGE: 'price',
  INVALID_ORDER_SIZE: 'minimum',
  ORDER_BOOK_EMPTY: 'no bids',
};

function extractOrderError(response) {
  if (!response || typeof response !== 'object') return undefined;

  if (typeof response.error === 'string') return response.error;
  if (typeof response.message === 'string') return response.message;

  return undefined;
}
```

---

## 10. Advanced Topics

### 10.1 Gnosis Safe Integration

Polymarket uses three signature types:

| SignatureType | Value | Use Case |
|---------------|-------|----------|
| `EOA` | 0 | Direct EOA trading |
| `POLY_PROXY` | 1 | Magic/Email login |
| `POLY_GNOSIS_SAFE` | 2 | Browser wallet |

#### Proxy Wallet Architecture

When users log in to Polymarket with MetaMask, the system creates a **Gnosis Safe proxy wallet**:

```
┌─────────────────────────────────────────────────────────────┐
│                    Polymarket Wallet Architecture            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Your MetaMask Wallet (EOA)                                 │
│  Address: 0xABC...123                                       │
│       │                                                     │
│       │ Control (Owner)                                     │
│       ▼                                                     │
│  Polymarket Proxy Wallet (Gnosis Safe)                      │
│  Address: 0xDEF...456  ← This is your displayed address     │
│       │                                                     │
│       └─► All trades, positions, deposits go through this   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Proxy wallet belongs entirely to you** - Uses 1-of-1 single-signature configuration, Polymarket has no control.

#### Why Use Gnosis Safe?

| Feature | EOA | Gnosis Safe |
|---------|-----|-------------|
| Gas Optimization | ❌ | ✅ Batch transactions save gas |
| Transaction Aggregation | ❌ | ✅ Multiple operations in one execution |
| Upgradeability | ❌ | ✅ Support module extensions |

---

## 11. References

### Official Documentation
- [Polymarket Official Docs](https://docs.polymarket.com/)
- [CLOB Introduction](https://docs.polymarket.com/developers/CLOB/introduction)
- [CTF Protocol Overview](https://docs.polymarket.com/developers/CTF/overview)

### Technical Analysis
- [How Polymarket Works](https://rocknblock.io/blog/how-polymarket-works-the-tech-behind-prediction-markets)
- [Building a Prediction Market Arbitrage Bot](https://navnoorbawa.substack.com/p/building-a-prediction-market-arbitrage)

### Open Source Projects
- [@polymarket/clob-client](https://github.com/Polymarket/clob-client) - Official CLOB client
- [Polymarket/ctf-exchange](https://github.com/Polymarket/ctf-exchange) - CTF exchange contracts
- [Polymarket Agents](https://github.com/Polymarket/agents) - AI trading agent framework

### Community Resources
- [Bitquery Polymarket Data API](https://docs.bitquery.io/docs/examples/polymarket-api/)
- [ChainSecurity Audit Report](https://old.chainsecurity.com/wp-content/uploads/2024/04/ChainSecurity_Polymarket_Conditional_Tokens_audit.pdf)

---

**Language:** [English](./) | [中文](../zh-CN/)