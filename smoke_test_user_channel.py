"""
smoke_test_user_channel.py

Connects to the Polymarket user channel, subscribes to the current
BTC 5-minute market, and prints raw events for 60 seconds.

No orders are placed. Use this to verify:
  1. Auth succeeds
  2. Subscription is accepted
  3. Event field names match what we expect

Usage:
    python smoke_test_user_channel.py
    python smoke_test_user_channel.py --duration 120
"""

import argparse
import json
import signal
import sys
import time

from execution.market_info import fetch_current_market

parser = argparse.ArgumentParser()
parser.add_argument("--duration", type=int, default=600)
args = parser.parse_args()

import logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import ApiCreds

with open("config.json") as f:
    config = json.load(f)
pm = config["polymarket"]

# Derive creds — the UI-provided keys are seeds, not the actual auth creds
_client = ClobClient(
    host="https://clob.polymarket.com",
    chain_id=137,
    key=pm["private_key"],
    creds=ApiCreds(
        api_key=pm["api_key"],
        api_secret=pm["api_secret"],
        api_passphrase=pm["api_passphrase"],
    ),
)
_client.set_api_creds(_client.create_or_derive_api_creds())
creds = _client.creds
print(f"Derived api_key: {creds.api_key[:16]}...")

market = fetch_current_market()
print(f"\nMarket: {market.slug}")
print(f"Condition ID: {market.condition_id}")
print(f"Subscribing for {args.duration}s...\n")

from websocket import WebSocketApp

def on_open(ws):
    print(">>> Connected")
    msg = {
        "auth": {
            "apiKey":     creds.api_key,
            "secret":     creds.api_secret,
            "passphrase": creds.api_passphrase,
        },
        "markets": [market.condition_id],
        "type": "user",
    }
    ws.send(json.dumps(msg))
    print(f">>> Subscribed to {market.condition_id[:20]}...")

def on_message(ws, message):
    if message == "PONG":
        print(">>> PONG")
        return
    try:
        parsed = json.loads(message)
        print(f"\n<<< {json.dumps(parsed, indent=2)}")
    except Exception:
        print(f"\n<<< RAW: {message[:300]}")

def on_error(ws, error):
    print(f"!!! ERROR: {error}")

def on_close(ws, code, msg):
    print(f">>> CLOSED: {code} {msg}")

ws = WebSocketApp(
    "wss://ws-subscriptions-clob.polymarket.com/ws/user",
    on_open=on_open,
    on_message=on_message,
    on_error=on_error,
    on_close=on_close,
)

def ping():
    while True:
        time.sleep(10)
        try:
            ws.send("PING")
        except Exception:
            break

import threading
threading.Thread(target=ping, daemon=True).start()

# Auto-close after duration
def timeout():
    time.sleep(args.duration)
    print(f"\n>>> {args.duration}s elapsed, closing")
    ws.close()

threading.Thread(target=timeout, daemon=True).start()

signal.signal(signal.SIGINT, lambda s, f: ws.close() or sys.exit(0))
ws.run_forever()