import requests
from collections import defaultdict
slug = "btc-updown-15m-1760028300"
resp = requests.get(
    "https://gamma-api.polymarket.com/markets/slug/" + slug, timeout=10
)
condition_id = resp.json()["conditionId"]

trades = requests.get(
    "https://data-api.polymarket.com/trades",
    params={"market": condition_id, "limit": 3},
    timeout=10,
).json()

for t in trades:
    print(t["timestamp"], t["transactionHash"])



    by_tx = defaultdict(list)
    for t in trades:
        by_tx[t["transactionHash"]].append(float(t["timestamp"]))

    for tx, timestamps in by_tx.items():
        if len(timestamps) > 1:
            print(f"TX {tx[:20]}...: {len(timestamps)} trades, "
                f"timestamps: {timestamps} "
                f"({'identical' if len(set(timestamps)) == 1 else 'different'})")