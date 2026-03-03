"""
metrics.py - CSV metrics writer for market maker sessions.
"""
import csv
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from execution.order_client import OrderPair, OrderStatus

SESSION_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

PAIRS_FIELDS = [
    "session_id", "market_slug", "condition_id", "outcome",
    "yes_price", "no_price", "order_size", "net_cost",
    "mid_at_placement", "mid_at_resolution", "unrealized_pnl",
    "fill_latency_s", "two_sided_gap_s",
    "placed_at", "first_fill_at", "second_fill_at",
]
PRICES_FIELDS = ["session_id", "market_slug", "ts", "mid", "best_bid", "best_ask"]


class MetricsWriter:
    def __init__(self, out_dir="data/metrics"):
        self._dir = Path(out_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._pairs_path  = self._dir / "pairs.csv"
        self._prices_path = self._dir / "prices.csv"
        self._pairs_buf   = []
        self._prices_buf  = []
        self._lock        = threading.Lock()
        self._stop_event  = threading.Event()
        self._ensure_headers()
        threading.Thread(target=self._flush_loop, daemon=True, name="metrics-flush").start()

    def record_pair(self, pair, outcome, market_slug, condition_id, mid_at_placement, mid_at_resolution):
        yes = pair.yes_order
        no  = pair.no_order
        placed_at = yes.placed_at if yes else None
        fill_times = sorted([
            o.filled_at for o in [yes, no]
            if o and o.status == OrderStatus.FILLED and o.filled_at
        ])
        first_fill_at  = fill_times[0] if fill_times else None
        second_fill_at = fill_times[1] if len(fill_times) == 2 else None
        fill_latency_s  = (first_fill_at - placed_at)      if (first_fill_at and placed_at) else None
        two_sided_gap_s = (second_fill_at - first_fill_at) if second_fill_at else None
        unrealized_pnl  = None
        if mid_at_resolution is not None:
            v = 0.0
            if yes and yes.status == OrderStatus.FILLED:
                v += yes.size * mid_at_resolution - yes.price * yes.size
            if no and no.status == OrderStatus.FILLED:
                v += no.size * (1.0 - mid_at_resolution) - no.price * no.size
            unrealized_pnl = round(v, 6)
        row = {
            "session_id": SESSION_ID, "market_slug": market_slug, "condition_id": condition_id,
            "outcome": outcome,
            "yes_price": yes.price if yes else None, "no_price": no.price if no else None,
            "order_size": yes.size if yes else None, "net_cost": round(pair.net_cost, 6),
            "mid_at_placement":  round(mid_at_placement,  4) if mid_at_placement  else None,
            "mid_at_resolution": round(mid_at_resolution, 4) if mid_at_resolution else None,
            "unrealized_pnl": unrealized_pnl,
            "fill_latency_s":  round(fill_latency_s,  2) if fill_latency_s  else None,
            "two_sided_gap_s": round(two_sided_gap_s, 2) if two_sided_gap_s else None,
            "placed_at":      round(placed_at,      3) if placed_at      else None,
            "first_fill_at":  round(first_fill_at,  3) if first_fill_at  else None,
            "second_fill_at": round(second_fill_at, 3) if second_fill_at else None,
        }
        with self._lock:
            self._pairs_buf.append(row)

    def record_price(self, mid, best_bid, best_ask, market_slug):
        with self._lock:
            self._prices_buf.append({
                "session_id": SESSION_ID, "market_slug": market_slug,
                "ts": round(time.time(), 3), "mid": round(mid, 4),
                "best_bid": round(best_bid, 4) if best_bid else None,
                "best_ask": round(best_ask, 4) if best_ask else None,
            })

    def flush(self):
        with self._lock:
            pairs  = self._pairs_buf[:]
            prices = self._prices_buf[:]
            self._pairs_buf.clear()
            self._prices_buf.clear()
        if pairs:
            self._append_csv(self._pairs_path, pairs, PAIRS_FIELDS)
        if prices:
            self._append_csv(self._prices_path, prices, PRICES_FIELDS)

    def stop(self):
        self._stop_event.set()
        self.flush()

    def _ensure_headers(self):
        for path, fields in [(self._pairs_path, PAIRS_FIELDS), (self._prices_path, PRICES_FIELDS)]:
            if not path.exists():
                with open(path, "w", newline="") as f:
                    csv.DictWriter(f, fieldnames=fields).writeheader()

    def _append_csv(self, path, rows, fields):
        with open(path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writerows(rows)

    def _flush_loop(self):
        while not self._stop_event.is_set():
            time.sleep(10)
            self.flush()