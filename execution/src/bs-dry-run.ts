/**
 * Dry-run Black-Scholes pricing model.
 *
 * Connects to Binance feed, captures a strike price, builds vol estimate,
 * and logs fair values + ask prices every few seconds. No orders placed.
 *
 * Usage: bun run src/bs-dry-run.ts [--duration 4h|5m] [--vol-window 300] [--t-dof 4]
 */

import { BinanceFeed } from "./data/binance-feed.ts";

// ── Config ──
const args = process.argv.slice(2);
const durationArg = args.includes("--duration") ? args[args.indexOf("--duration") + 1] : "4h";
const volWindowS = args.includes("--vol-window")
  ? Number(args[args.indexOf("--vol-window") + 1])
  : 300;
const spreadMultiplier = args.includes("--spread-mult")
  ? Number(args[args.indexOf("--spread-mult") + 1])
  : 1.0;
const minSpread = args.includes("--min-spread")
  ? Number(args[args.indexOf("--min-spread") + 1])
  : 0.02;
const skewK = args.includes("--skew-k")
  ? Number(args[args.indexOf("--skew-k") + 1])
  : 0.005;
const tDof = args.includes("--t-dof")
  ? Number(args[args.indexOf("--t-dof") + 1])
  : 4;

const DURATION_S: Record<string, number> = { "5m": 300, "4h": 14400 };
const marketDurationS = DURATION_S[durationArg!] ?? 14400;

console.log(`[BS Dry Run] duration=${durationArg} (${marketDurationS}s), volWindow=${volWindowS}s`);
console.log(`[BS Dry Run] spreadMult=${spreadMultiplier}, minSpread=${minSpread}, skewK=${skewK}`);
console.log(`[BS Dry Run] distribution=${tDof > 0 ? `Student's t (ν=${tDof})` : "Normal"}`);

// ── State ──
let strike = 0;
let marketStartMs = 0;
const priceHistory: { ts: number; price: number }[] = [];

// ── Math helpers ──

function normalCdf(x: number): number {
  if (x > 6) return 1;
  if (x < -6) return 0;
  const a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741;
  const a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;
  const sign = x < 0 ? -1 : 1;
  // A&S formula 7.1.26 — input must be |x|/√2
  const z = Math.abs(x) / Math.SQRT2;
  const t = 1.0 / (1.0 + p * z);
  const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-z * z);
  return 0.5 * (1.0 + sign * y);
}

function logGamma(x: number): number {
  const g = 7;
  const coefs = [
    0.99999999999980993, 676.5203681218851, -1259.1392167224028,
    771.32342877765313, -176.61502916214059, 12.507343278686905,
    -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
  ];
  if (x < 0.5) {
    return Math.log(Math.PI / Math.sin(Math.PI * x)) - logGamma(1 - x);
  }
  x -= 1;
  let sum = coefs[0]!;
  for (let i = 1; i < g + 2; i++) {
    sum += coefs[i]! / (x + i);
  }
  const t = x + g + 0.5;
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(sum);
}

function logBeta(a: number, b: number): number {
  return logGamma(a) + logGamma(b) - logGamma(a + b);
}

function betacf(a: number, b: number, x: number): number {
  const TINY = 1e-30;
  const EPS = 1e-14;
  const qab = a + b;
  const qap = a + 1;
  const qam = a - 1;

  let c = 1;
  let d = 1 - qab * x / qap;
  if (Math.abs(d) < TINY) d = TINY;
  d = 1 / d;
  let h = d;

  for (let m = 1; m <= 300; m++) {
    const m2 = 2 * m;

    let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
    d = 1 + aa * d;
    if (Math.abs(d) < TINY) d = TINY;
    c = 1 + aa / c;
    if (Math.abs(c) < TINY) c = TINY;
    d = 1 / d;
    h *= d * c;

    aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
    d = 1 + aa * d;
    if (Math.abs(d) < TINY) d = TINY;
    c = 1 + aa / c;
    if (Math.abs(c) < TINY) c = TINY;
    d = 1 / d;
    const del = d * c;
    h *= del;

    if (Math.abs(del - 1) < EPS) break;
  }

  return h;
}

function regIncBeta(x: number, a: number, b: number): number {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  if (x > (a + 1) / (a + b + 2)) return 1 - regIncBeta(1 - x, b, a);
  const lnB = logBeta(a, b);
  const front = Math.exp(Math.log(x) * a + Math.log(1 - x) * b - lnB) / a;
  return front * betacf(a, b, x);
}

function studentTCdf(x: number, nu: number): number {
  if (x === 0) return 0.5;
  const t2 = x * x;
  const betaVal = regIncBeta(nu / (nu + t2), nu / 2, 0.5);
  return x > 0 ? 1 - 0.5 * betaVal : 0.5 * betaVal;
}

/** Choose CDF based on tDof setting. */
function cdf(x: number): number {
  return tDof > 0 ? studentTCdf(x, tDof) : normalCdf(x);
}

function computeRealizedVol(history: { ts: number; price: number }[]): number {
  if (history.length < 10) return 0;

  // Sample at ~1s intervals
  const sampled: number[] = [history[0]!.price];
  let lastTs = history[0]!.ts;
  for (let i = 1; i < history.length; i++) {
    if (history[i]!.ts - lastTs >= 1000) {
      sampled.push(history[i]!.price);
      lastTs = history[i]!.ts;
    }
  }
  if (sampled.length < 5) return 0;

  const returns: number[] = [];
  for (let i = 1; i < sampled.length; i++) {
    returns.push(Math.log(sampled[i]! / sampled[i - 1]!));
  }

  const mean = returns.reduce((s, r) => s + r, 0) / returns.length;
  const variance = returns.reduce((s, r) => s + (r - mean) ** 2, 0) / (returns.length - 1);
  const secondsPerYear = 365.25 * 24 * 3600;
  return Math.sqrt(variance * secondsPerYear);
}

// ── Feed ──
const feed = new BinanceFeed("btcusdt");
let lastLogMs = 0;
let tickCount = 0;

feed.on("data", (point) => {
  const now = point.timestamp;
  tickCount++;

  // Capture strike on first tick
  if (strike === 0) {
    strike = point.price;
    marketStartMs = now;
    console.log(`\n[BS] Strike captured: $${strike.toFixed(2)}`);
    console.log(`[BS] Simulating ${durationArg} market starting now\n`);
  }

  // Add to history
  priceHistory.push({ ts: now, price: point.price });
  const cutoff = now - volWindowS * 1000;
  while (priceHistory.length > 0 && priceHistory[0]!.ts < cutoff) {
    priceHistory.shift();
  }

  // Log every 5 seconds
  if (now - lastLogMs < 5000) return;
  lastLogMs = now;

  const spot = point.price;
  const elapsedS = (now - marketStartMs) / 1000;
  const remainingS = Math.max(1, marketDurationS - elapsedS);
  const tau = remainingS / (365.25 * 24 * 3600);
  const sigma = computeRealizedVol(priceHistory);

  if (sigma <= 0) {
    console.log(`[BS] t=${elapsedS.toFixed(0)}s spot=$${spot.toFixed(2)} | building vol estimate (${priceHistory.length} samples)...`);
    return;
  }

  const sqrtTau = Math.sqrt(tau);
  const d2 = (Math.log(spot / strike) - (sigma * sigma / 2) * tau) / (sigma * sqrtTau);

  // Compare both distributions
  const fairUpNorm = normalCdf(d2);
  const fairUpT = studentTCdf(d2, tDof);
  const fairUp = cdf(d2);
  const fairDown = 1 - fairUp;
  const halfSpread = Math.max(minSpread, spreadMultiplier * sigma * sqrtTau);

  const askUp = fairUp + halfSpread;
  const askDown = fairDown + halfSpread;
  const bidUp = Math.max(0.01, fairUp - halfSpread);
  const bidDown = Math.max(0.01, fairDown - halfSpread);

  // Simulate inventory imbalance scenarios
  const imb5 = {
    askUp: askUp - skewK * 5, askDown: askDown + skewK * 5,
  };
  const imbN5 = {
    askUp: askUp - skewK * -5, askDown: askDown + skewK * -5,
  };

  const moneyness = ((spot - strike) / strike * 100);
  const pctRemaining = (remainingS / marketDurationS * 100);

  console.log(
    `[BS] t=${elapsedS.toFixed(0)}s (${pctRemaining.toFixed(0)}% left) | ` +
    `spot=$${spot.toFixed(2)} strike=$${strike.toFixed(2)} moneyness=${moneyness >= 0 ? "+" : ""}${moneyness.toFixed(3)}% | ` +
    `σ=${(sigma * 100).toFixed(1)}% d2=${d2.toFixed(3)} | ` +
    `fair=(${fairUp.toFixed(3)}/${fairDown.toFixed(3)}) spread=${(halfSpread * 2).toFixed(3)}`
  );
  console.log(
    `     asks=(${askUp.toFixed(3)}/${askDown.toFixed(3)}) combined=$${(askUp + askDown).toFixed(3)} | ` +
    `bids=(${bidUp.toFixed(3)}/${bidDown.toFixed(3)}) | ` +
    `N(d2)=${fairUpNorm.toFixed(4)} vs t(d2,ν=${tDof})=${fairUpT.toFixed(4)} Δ=${((fairUpT - fairUpNorm) * 100).toFixed(2)}%`
  );
  console.log(
    `     imb+5: asks=(${imb5.askUp.toFixed(3)}/${imb5.askDown.toFixed(3)}) | ` +
    `imb-5: asks=(${imbN5.askUp.toFixed(3)}/${imbN5.askDown.toFixed(3)})`
  );
});

await feed.start();
console.log("[BS Dry Run] Connected to Binance BTCUSDT feed. Logging every 5s. Ctrl+C to stop.\n");

// Keep alive
await new Promise(() => {});
