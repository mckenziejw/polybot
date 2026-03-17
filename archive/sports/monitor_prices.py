"""
Monitor Polymarket CLOB orderbooks for sports markets via WebSocket.

Connects to the CLOB market WebSocket, subscribes to active sports markets,
and records periodic snapshots to parquet files.

Usage:
    python sports/monitor_prices.py [--interval 10] [--markets-file data/sports/markets.json]

Output: data/sports/snapshots/<sport>_<YYYY-MM-DD>.parquet  (appended throughout the day)
"""

import argparse
import json
import logging
import os
import signal
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import websocket  # websocket-client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "sports" / "snapshots"

CLOB_WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"


# ── Orderbook state ──────────────────────────────────────────────────────────

class OrderBook:
    """Minimal orderbook tracker from CLOB WS events."""

    def __init__(self, asset_id: str):
        self.asset_id = asset_id
        self.bids: dict[float, float] = {}  # price -> size
        self.asks: dict[float, float] = {}  # price -> size
        self.last_update: float = 0

    def apply_snapshot(self, bids: list, asks: list):
        self.bids = {float(p): float(s) for p, s in bids}
        self.asks = {float(p): float(s) for p, s in asks}
        self.last_update = time.time()

    def apply_changes(self, changes: list):
        """Apply price_change deltas: [[side, price, size], ...]"""
        for change in changes:
            side, price, size = change[0], float(change[1]), float(change[2])
            book = self.bids if side == "BUY" else self.asks
            if size == 0:
                book.pop(price, None)
            else:
                book[price] = size
        self.last_update = time.time()

    def snapshot(self) -> dict:
        """Return summary stats for this book."""
        best_bid = max(self.bids.keys()) if self.bids else 0.0
        best_ask = min(self.asks.keys()) if self.asks else 1.0
        bid_depth = sum(self.bids.values())
        ask_depth = sum(self.asks.values())
        mid = (best_bid + best_ask) / 2 if self.bids and self.asks else None
        spread = best_ask - best_bid if self.bids and self.asks else None

        return {
            "asset_id": self.asset_id,
            "best_bid": best_bid,
            "best_ask": best_ask,
            "mid": mid,
            "spread": spread,
            "bid_depth": round(bid_depth, 2),
            "ask_depth": round(ask_depth, 2),
            "n_bid_levels": len(self.bids),
            "n_ask_levels": len(self.asks),
        }


# ── Market metadata ──────────────────────────────────────────────────────────

def load_markets(path: str) -> tuple[dict[str, dict], list[str]]:
    """
    Load markets JSON.
    Returns:
        asset_map: asset_id -> {market_id, sport, title, outcome_label}
        asset_ids: list of all asset_ids to subscribe
    """
    with open(path) as f:
        data = json.load(f)

    markets = data.get("markets", [])
    asset_map: dict[str, dict] = {}
    asset_ids: list[str] = []

    for m in markets:
        # Skip settled / closed
        status = (m.get("status") or "").lower()
        if status in ("closed", "resolved", "settled"):
            continue

        market_id = m.get("market_id")
        sport = m.get("sport") or "unknown"
        title = m.get("title", "")
        token_ids = m.get("token_ids", [])

        # Try to pair token_ids with outcome labels
        outcome_prices = m.get("outcome_prices", {})
        labels = list(outcome_prices.keys()) if outcome_prices else []

        for i, tid in enumerate(token_ids):
            label = labels[i] if i < len(labels) else f"outcome_{i}"
            asset_map[tid] = {
                "market_id": market_id,
                "sport": sport,
                "title": title,
                "outcome": label,
            }
            asset_ids.append(tid)

    return asset_map, asset_ids


# ── Snapshot persistence ─────────────────────────────────────────────────────

def flush_snapshots(buffer: list[dict]):
    """Write buffered snapshots to parquet, grouped by sport+date."""
    if not buffer:
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(buffer)

    # Group by sport + date for separate files
    for (sport, date_str), group in df.groupby(["sport", "date"]):
        safe_sport = str(sport).replace("/", "_").replace(" ", "_").lower()
        fname = f"{safe_sport}_{date_str}.parquet"
        fpath = DATA_DIR / fname

        if fpath.exists():
            existing = pd.read_parquet(fpath)
            combined = pd.concat([existing, group], ignore_index=True)
        else:
            combined = group

        combined.to_parquet(fpath, index=False)
        log.info("Wrote %d rows to %s (total %d)", len(group), fpath, len(combined))

    buffer.clear()


# ── WebSocket handler ────────────────────────────────────────────────────────

class PriceMonitor:
    def __init__(self, asset_map: dict[str, dict], asset_ids: list[str],
                 snapshot_interval: float = 10.0, flush_interval: float = 60.0):
        self.asset_map = asset_map
        self.asset_ids = asset_ids
        self.snapshot_interval = snapshot_interval
        self.flush_interval = flush_interval

        self.books: dict[str, OrderBook] = {}
        self.buffer: list[dict] = []
        self.last_snapshot_time = 0.0
        self.last_flush_time = 0.0
        self.running = True

    def on_open(self, ws):
        log.info("WebSocket connected, subscribing to %d assets", len(self.asset_ids))
        # Subscribe in batches to avoid message size limits
        batch_size = 50
        for i in range(0, len(self.asset_ids), batch_size):
            batch = self.asset_ids[i:i + batch_size]
            msg = json.dumps({"assets_ids": batch, "type": "market"})
            ws.send(msg)
            log.info("Subscribed batch %d-%d", i, i + len(batch))

    def on_message(self, ws, raw_msg: str):
        try:
            msgs = json.loads(raw_msg)
        except json.JSONDecodeError:
            log.warning("Non-JSON message: %s", raw_msg[:200])
            return

        # Could be a single event or a list
        if isinstance(msgs, dict):
            msgs = [msgs]

        for msg in msgs:
            self._handle_event(msg)

        # Periodic snapshot
        now = time.time()
        if now - self.last_snapshot_time >= self.snapshot_interval:
            self._take_snapshots()
            self.last_snapshot_time = now

        # Periodic flush to disk
        if now - self.last_flush_time >= self.flush_interval:
            flush_snapshots(self.buffer)
            self.last_flush_time = now

    def _handle_event(self, msg: dict):
        event_type = msg.get("event_type") or msg.get("type", "")
        asset_id = msg.get("asset_id")

        if not asset_id:
            # Some messages don't have asset_id (e.g. heartbeat)
            return

        if asset_id not in self.books:
            self.books[asset_id] = OrderBook(asset_id)
        book = self.books[asset_id]

        if event_type == "book":
            bids = msg.get("bids", [])
            asks = msg.get("asks", [])
            book.apply_snapshot(bids, asks)
        elif event_type == "price_change":
            changes = msg.get("changes", [])
            book.apply_changes(changes)
        elif event_type in ("last_trade_price", "tick_size"):
            pass  # Informational, ignore
        else:
            log.debug("Unknown event_type=%s for asset=%s", event_type, asset_id[:12])

    def _take_snapshots(self):
        now_utc = datetime.now(timezone.utc)
        ts = now_utc.isoformat()
        date_str = now_utc.strftime("%Y-%m-%d")

        for asset_id, book in self.books.items():
            if book.last_update == 0:
                continue  # No data yet

            meta = self.asset_map.get(asset_id, {})
            snap = book.snapshot()
            snap.update({
                "timestamp": ts,
                "date": date_str,
                "market_id": meta.get("market_id"),
                "sport": meta.get("sport", "unknown"),
                "title": meta.get("title", ""),
                "outcome": meta.get("outcome", ""),
            })
            self.buffer.append(snap)

    def on_error(self, ws, error):
        log.error("WebSocket error: %s", error)

    def on_close(self, ws, close_status_code, close_msg):
        log.warning("WebSocket closed: code=%s msg=%s", close_status_code, close_msg)

    def run(self):
        signal.signal(signal.SIGINT, lambda *_: self.stop())
        signal.signal(signal.SIGTERM, lambda *_: self.stop())

        while self.running:
            log.info("Connecting to %s", CLOB_WS_URL)
            ws = websocket.WebSocketApp(
                CLOB_WS_URL,
                on_open=self.on_open,
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close,
            )
            ws.run_forever(ping_interval=30, ping_timeout=10)

            # Flush remaining data before reconnect
            flush_snapshots(self.buffer)

            if self.running:
                log.info("Reconnecting in 5 seconds...")
                time.sleep(5)

    def stop(self):
        log.info("Shutting down, flushing remaining snapshots...")
        self.running = False
        flush_snapshots(self.buffer)


def main():
    parser = argparse.ArgumentParser(description="Monitor Polymarket sports orderbooks")
    parser.add_argument("--interval", type=float, default=10.0,
                        help="Snapshot interval in seconds (default: 10)")
    parser.add_argument("--flush-interval", type=float, default=60.0,
                        help="Flush to parquet every N seconds (default: 60)")
    parser.add_argument("--markets-file", type=str,
                        default=str(PROJECT_ROOT / "data" / "sports" / "markets.json"),
                        help="Path to markets JSON from fetch_markets.py")
    args = parser.parse_args()

    if not Path(args.markets_file).exists():
        log.error("Markets file not found: %s", args.markets_file)
        log.error("Run fetch_markets.py first.")
        sys.exit(1)

    asset_map, asset_ids = load_markets(args.markets_file)
    if not asset_ids:
        log.error("No active markets with token IDs found. Nothing to monitor.")
        sys.exit(1)

    log.info("Loaded %d assets across %d markets",
             len(asset_ids), len(set(m["market_id"] for m in asset_map.values())))

    monitor = PriceMonitor(
        asset_map=asset_map,
        asset_ids=asset_ids,
        snapshot_interval=args.interval,
        flush_interval=args.flush_interval,
    )
    monitor.run()


if __name__ == "__main__":
    main()
