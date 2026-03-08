import { ClobClient } from "@polymarket/clob-client";
import { loadConfig } from "./src/config.ts";
import { Wallet } from "@ethersproject/wallet";
import { writeFileSync } from "fs";

const config = loadConfig();
const pm = config.polymarket;
const wallet = new Wallet(pm.privateKey);

console.error("Deriving API credentials...");
const clobClient = new ClobClient(pm.host, pm.chainId, wallet);
const creds = await clobClient.createOrDeriveApiKey();
console.error("Got creds, fetching trades...");

const authedClient = new ClobClient(pm.host, pm.chainId, wallet, creds);
const trades = await authedClient.getTrades({}, false);

console.error(`Fetched ${trades.length} trades`);
writeFileSync("/tmp/trades_raw.json", JSON.stringify(trades, null, 2));
console.log(`Wrote ${trades.length} trades to /tmp/trades_raw.json`);
