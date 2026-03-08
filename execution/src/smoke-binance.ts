import { BinanceFeed } from "./data/binance-feed.ts";

const feed = new BinanceFeed();
let count = 0;

feed.on("data", (point) => {
  count++;
  if (count <= 5 || count % 50 === 0) {
    console.log(
      `[${count}] price=${point.price.toFixed(2)} bid=${point.bid!.toFixed(2)} ask=${point.ask!.toFixed(2)} latency=${Date.now() - point.timestamp}ms`
    );
  }
});

console.log("Connecting to Binance BTC/USDT bookTicker...");
await feed.start();

setTimeout(async () => {
  await feed.stop();
  console.log(`\nReceived ${count} ticks in 5 seconds (${(count / 5).toFixed(1)}/sec)`);
  process.exit(0);
}, 5_000);
