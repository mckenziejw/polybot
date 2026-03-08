# Cross-Platform Arbitrage: Polymarket vs Kalshi

> **Research date:** March 2026
> **Updated:** With verified web research as of March 2026.
> **Caveat:** This is not legal or financial advice. Verify all legal and regulatory claims before acting.

---

## 1. Jurisdictional Limitations

### Kalshi — CFTC-Regulated

- **Regulatory status:** Kalshi is a CFTC-regulated Designated Contract Market (DCM), the first standalone prediction market exchange to receive this designation (approved 2020).
- **Who can trade:** US residents only (as of early 2025). Kalshi requires US residency and citizenship/permanent residency.
- **State restrictions:** As of early 2025, Kalshi was available in most but not all US states. Historically, some states with stricter gambling/derivatives laws have been excluded. The specific list changes — check Kalshi's current terms of service.
- **Age requirement:** 18+ (some states may require 21+).
- **Event contracts:** After a landmark court victory in late 2024 (Kalshi v. CFTC regarding election contracts), Kalshi was cleared to offer political event contracts. This ruling broadly expanded what CFTC-regulated exchanges can list.

### Polymarket — NOW CFTC-REGULATED (as of Dec 2025)

- **Regulatory status (UPDATED March 2026):** Polymarket acquired QCEX in July 2025 for $112M, gaining its CFTC exchange and clearinghouse licenses. The CFTC issued a "no-action" letter in September 2025, and Polymarket officially relaunched as a US-compliant Designated Contract Market (DCM) in December 2025.
- **US users:** US users CAN now legally trade on Polymarket. However, US access requires KYC and trading through approved Futures Commission Merchants (FCMs) or regulated brokerages — no more direct crypto wallet access for US users.
- **Two versions:** There are now effectively two Polymarket platforms:
  1. **Polymarket US (DCM):** Regulated, KYC required, trades via FCMs, US fee structure (1 basis point / 0.01% taker fee)
  2. **Polymarket International:** Original crypto-native version, wallet-based, different fee structure (crypto markets: up to 1.56% taker fee at p=0.50)
- **State restrictions:** Despite federal approval, some states (Tennessee, Massachusetts, Nevada) are pushing back. Check state-level legality.
- **Historical:** Settled with CFTC in 2022 for $1.4M. Blocked US users 2022-2025.

### Can the Same Person Trade on Both?

- **YES — this is now legally feasible for US persons.** Both Kalshi and Polymarket US are CFTC-regulated DCMs. A US person can legally have accounts on both.
- **The jurisdictional blocker is RESOLVED.** Cross-platform arb between Polymarket US and Kalshi is now straightforward and legal for US residents (subject to state-level restrictions).
- **Non-US persons:** May be able to use Polymarket International but still cannot trade on Kalshi.
- **Key consideration:** Polymarket US has different fees and access methods than Polymarket International. The arb may need to factor in the US fee structure (0.01% taker) vs Kalshi fees.

### KYC/AML Implications

- **Kalshi:** Full KYC — name, address, SSN, government ID. All transactions are reported.
- **Polymarket:** Minimal KYC for most users (wallet-based). Some larger withdrawals may trigger KYC.
- **Cross-platform concern:** If you're identifiable on both platforms, regulators could flag pattern-of-life trading across regulated and unregulated venues. This is analogous to trading on a licensed exchange and an offshore unregulated exchange simultaneously — a red flag for compliance.
- **Wash trading / manipulation:** Holding positions on the same event across platforms is not inherently wash trading, but regulators could scrutinize it, especially if resolution criteria differ.

### Tax Implications

- **Kalshi:** Issues 1099-B forms. Gains/losses are treated as Section 1256 contracts (60% long-term / 40% short-term capital gains), which is actually favorable tax treatment. Mark-to-market at year-end.
- **Polymarket:** No tax forms issued. Gains are still taxable (likely as ordinary income or short-term capital gains from crypto/derivatives). You are responsible for self-reporting.
- **Cross-platform arb:** If you're long on one platform and short on the other (the arb), you may trigger constructive sale or straddle rules under the IRS code, which could defer or disallow losses. Consult a tax professional familiar with derivatives.
- **DYOR:** The IRS treatment of prediction market contracts (especially offshore/crypto ones) is still evolving. This area is genuinely unsettled.

---

## 2. Kalshi Account Requirements

### Opening an Account

- **Required information:**
  - Full legal name
  - Date of birth
  - Social Security Number (SSN) or ITIN
  - US residential address (no PO boxes)
  - Government-issued photo ID (driver's license, passport, state ID)
  - Phone number and email
- **Verification:** Automated identity verification (similar to brokerage account opening). Usually approved within minutes; manual review can take 1-2 business days.
- **Entity accounts:** Available for LLCs and other entities (additional documentation required).

### Deposits and Withdrawals

- **Minimum deposit:** No formal minimum, but you need enough to place trades (contracts typically range from $0.01 to $0.99 per share, minimum order sizes apply).
- **Deposit methods:** Bank transfer (ACH), wire transfer, debit card. ACH is free but takes 1-3 business days to settle. Wire is faster but may have fees.
- **Withdrawal methods:** ACH (free, 1-3 business days), wire transfer.
- **Withdrawal timing:** ACH withdrawals typically take 2-3 business days. Settled funds only.
- **Daily limits:** There may be position limits on certain contracts (CFTC-mandated). Check specific contract specs.

### API Access

- **Availability:** Kalshi offers a REST API and WebSocket feeds.
- **Requirements:** As of early 2025, API access was available to all verified account holders. No separate approval process needed (unlike some brokerages).
- **Documentation:** Public API docs at https://trading-api.readme.io/ (verify current URL).
- **Rate limits:** API has rate limits (historically around 10 requests/second for REST). WebSocket feeds for market data.
- **Order types:** Limit orders, market orders. API supports programmatic trading.
- **VERIFY:** Check if Kalshi has added any additional API tiers, fees, or approval requirements since mid-2025.

---

## 3. Polymarket Legal Status for US Users (UPDATED March 2026)

### Current Status — RESOLVED

- **Polymarket is now CFTC-regulated** as a Designated Contract Market (DCM) since December 2025.
- **Timeline:**
  - Jan 2022: CFTC settlement ($1.4M fine), US users blocked
  - Jul 2025: Polymarket acquires QCEX for $112M (existing CFTC licenses)
  - Sep 2025: CFTC issues "no-action" letter to QCX
  - Dec 2025: Polymarket US officially launches as regulated DCM
  - Jan 2026: US users can trade via approved FCMs with full KYC
- **US access:** Legal and regulated. Requires KYC, approved broker, no more direct crypto wallet trading for US users.
- **State restrictions:** Some states still pushing back (Tennessee, Massachusetts, Nevada). Check current state-level availability.

### Implications for Cross-Platform Arb

The regulatory landscape has fundamentally changed. The primary blocker (Polymarket unavailable to US users) is removed. Cross-platform arb between Polymarket US and Kalshi is now legally feasible, making this strategy significantly more viable than it was even 6 months ago.

**Key question:** Does Polymarket US share the same orderbook/liquidity as Polymarket International? If they're separate pools, the arb opportunity may exist between all three venues (Kalshi, Polymarket US, Polymarket International).

---

## 4. Practical Arb Execution Concerns

### Settlement Timing

- **Kalshi:** Settles in USD. Contracts settle shortly after the event resolves (typically within hours). Funds available for withdrawal after settlement.
- **Polymarket:** Settles on Polygon (USDC). Resolution depends on UMA's Optimistic Oracle (or the specific oracle used). There is a dispute window (typically 2-4 hours for UMA). After resolution, USDC is claimable immediately on-chain.
- **Mismatch risk:** If one platform settles hours before the other, you have a window where capital is locked. For BTC Up/Down contracts with 5-minute or 15-minute durations, this matters less for the event risk but matters for capital turnover.
- **Key risk:** If the two platforms resolve the same event differently (rare but possible), you could lose on both sides.

### Resolution Criteria Differences

This is the most critical arb risk:

- **Kalshi BTC contracts:** Typically reference a specific price source (e.g., CME CF Bitcoin Reference Rate, or CoinDesk BPI) at a specific time.
- **Polymarket BTC contracts:** Reference price may differ (e.g., Binance BTC/USDT spot, or a different aggregated feed). Resolution is via oracle vote.
- **CRITICAL:** If Kalshi says "BTC closed above $X at 4:00 PM ET per CME CF BRR" and Polymarket says "BTC was above $X at 4:00 PM ET per Binance," these can give **different results** when BTC is near the boundary. This is not arb — it's basis risk.
- **Due diligence:** Before arbing, you MUST verify both platforms reference the exact same price source and timestamp. If they don't, the "arb" has unhedged risk.

### Capital Requirements

- **Both platforms require prefunded positions.** There is no cross-margining.
- **Worst case for a pure arb:** You buy YES on Platform A at $0.45 and NO on Platform B at $0.45. Total cost: $0.90 per share. Guaranteed payout: $1.00. Profit: $0.10 per share minus fees.
- **Capital efficiency:** You need capital locked on BOTH platforms simultaneously. If you're doing this at scale, you need significant capital on each platform.
- **Fees:**
  - Kalshi: Maker/taker fees (historically 1-2 cents per contract or percentage-based). Check current fee schedule.
  - Polymarket: No explicit trading fees on the CLOB, but there are gas costs (minimal on Polygon), and the bid-ask spread is the implicit cost.
- **Minimum profitable spread:** After fees on both sides, you likely need a combined price (YES_A + NO_B or NO_A + YES_B) below $0.96-0.97 to make the arb worthwhile. Tighter than that and fees eat the profit.

### Withdrawal Risk

- **Kalshi:** Regulated US entity. FDIC-like protections do NOT apply (this is not a bank), but as a DCM, customer funds are segregated. Withdrawal is via standard banking rails (1-3 days).
- **Polymarket:** Funds are on-chain (Polygon USDC). You can withdraw anytime by withdrawing from the Polymarket proxy wallet to your own wallet. However, bridging from Polygon to Ethereum mainnet or to an exchange for fiat off-ramp adds time and cost.
- **Platform risk:** Polymarket has higher platform risk (offshore, unregulated). If the platform has issues, your recourse is limited. Kalshi, being CFTC-regulated, has more regulatory oversight and customer protections.

---

## 5. Competitive Landscape

### How Many Bots Are Already Doing This?

- **Significant bot activity on both platforms.** Polymarket in particular has very active algorithmic trading — the API is well-documented and the on-chain nature makes it accessible.
- **Cross-platform arb specifically:** Harder to quantify. The legal barriers (Section 1 above) reduce the pool of participants, but sophisticated actors (potentially non-US entities with US partners, or US actors accepting legal risk) likely operate.
- **Kalshi API adoption:** Growing but smaller than Polymarket. Kalshi's API was less mature historically but has improved.

### Has the Spread Compressed?

- **Yes, significantly.** As both platforms matured through 2024-2025:
  - More markets on both platforms cover the same events
  - Liquidity improved on both sides
  - Algorithmic market makers tightened spreads
- **BTC-specific contracts:** These are among the most liquid on both platforms. Spreads are likely tight and arb opportunities are fleeting.
- **Event-specific vs. continuous:** For one-off events (elections, specific date outcomes), arb windows can persist longer because there's less automated flow. For high-frequency BTC contracts, any arb is likely picked off in seconds.

### Barriers to Entry

1. **Legal barrier (primary):** The jurisdictional mismatch is the biggest barrier. This protects existing players who have found a legal structure (or accepted the risk).
2. **Capital lockup:** Need capital on both platforms simultaneously. No cross-margining.
3. **Speed:** For BTC contracts specifically, you're competing against fast bots on both platforms. Latency matters.
4. **Resolution risk knowledge:** Understanding the exact resolution criteria on both platforms requires deep domain expertise.
5. **Operational complexity:** Managing two different platforms (one fiat/REST, one crypto/on-chain), two different order management systems, different settlement mechanics.

### Is It Worth It?

**Honest assessment for BTC Up/Down contracts specifically:**

- The most liquid, high-frequency contracts (like your 5-min/15-min BTC Up/Down) are likely **not good arb candidates** because:
  - Spreads are tight on both platforms individually
  - Cross-platform spread (after fees) is likely minimal
  - Resolution criteria may differ (basis risk)
  - Legal risk is non-trivial
  - You're competing against well-capitalized, fast actors

- **Better arb candidates** (if the legal issues are resolved):
  - Longer-dated, less liquid prediction markets
  - Events where one platform has much more liquidity/flow than the other
  - New market listings where pricing hasn't converged

---

## 6. Recommendations (UPDATED — Polymarket IS now US-approved)

### The Jurisdictional Blocker Is Removed

Polymarket received CFTC DCM approval in late 2025. Cross-platform arb is now:
- **Legally straightforward** for US persons (both platforms are CFTC-regulated)
- **Operationally feasible** (you already have Polymarket execution infra, just need Kalshi)
- **Worth investigating seriously**, especially for BTC contracts that exist on both platforms

### Recommended Next Steps

1. **Open a Kalshi account** — straightforward KYC, API access available to all verified users
2. **Verify resolution criteria** — do Kalshi and Polymarket BTC contracts use the same price oracle? This is the critical risk factor.
3. **Compare fee structures:**
   - Polymarket US: 0.01% taker fee (very low)
   - Polymarket International crypto: up to 1.56% taker fee
   - Kalshi: max $0.02/contract
   - Minimum profitable spread after fees on both sides: ~2-3%
4. **Build a price monitoring tool** — fetch real-time prices from both platforms, flag divergences
5. **Determine if Polymarket US and International share the same orderbook** — if separate, there may be three-way arb opportunities
6. **Start small** — validate the full lifecycle (entry, settlement, withdrawal) before scaling
7. **Consult a tax professional** — straddle rules, Section 1256 vs ordinary income treatment

---

## 7. Remaining Open Questions

- [ ] Do Kalshi and Polymarket BTC contracts reference the same price oracle?
- [ ] What is the exact resolution timestamp for BTC contracts on each platform?
- [ ] Does Polymarket US share the same orderbook as Polymarket International?
- [ ] What are Kalshi's current API rate limits for WebSocket data?
- [ ] Are there other regulated platforms (Robinhood prediction markets, Webull+Kalshi, Nasdaq/Cboe event contracts) worth including?
- [ ] What is the typical spread divergence frequency and magnitude for BTC contracts?

---

## Sources

- [Polymarket CFTC Approval](https://www.thebulldog.law/polymarket-receives-cftc-approval-to-resume-us-operations-after-years-offshore)
- [Polymarket US Legal Status 2026](https://www.gamblinginsider.com/in-depth/106291/is-polymarket-legal-in-the-us)
- [Polymarket Returns to US](https://reason.com/2026/01/04/the-return-of-polymarket/)
- [Kalshi API Docs](https://docs.kalshi.com/welcome)
- [Kalshi Fee Schedule](https://kalshi.com/fee-schedule)
- [Polymarket Fee Structure](https://docs.polymarket.com/polymarket-learn/trading/fees)
- [Cross-Platform Arb Guide](https://www.trevorlasn.com/blog/how-prediction-market-polymarket-kalshi-arbitrage-works)
- [EventArb Calculator](https://www.eventarb.com/)
