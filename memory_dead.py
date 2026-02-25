# # Run this before anything else to understand your actual data volume
# from pathlib import Path
# import pyarrow.parquet as pq

# trade_files = list(Path("data/trade_events").glob("*.parquet"))
# print(f"Total market files: {len(trade_files)}")

# # Sample 10 files to get typical row counts and memory usage
# import numpy as np
# sample = np.random.choice(trade_files, min(10, len(trade_files)), replace=False)
# row_counts = []
# for f in sample:
#     t = pq.read_table(f)
#     row_counts.append(len(t))
#     print(f"  {f.name}: {len(t):,} rows, ~{t.nbytes / 1e6:.1f} MB")

# print(f"\nMedian rows per market: {np.median(row_counts):,.0f}")
# print(f"Estimated total rows: {np.median(row_counts) * len(trade_files):,.0f}")
# print(f"Estimated total uncompressed size: {np.median(row_counts) * len(trade_files) * 200 / 1e9:.1f} GB")

# import pyarrow.parquet as pq

# # Pick the median-sized file
# t = pq.read_table("data/trade_events/btc-updown-5m-1771624500.parquet")
# df = t.to_pandas()

# print(df["side"].value_counts())
# print(f"\nUnique prices: {df['trade_price'].nunique()}")
# print(f"Rows with size == 0: {(df['size'] == 0).sum()}")
# print(f"\nTimestamp gaps (ms) between consecutive rows:")
# gaps = df["exchange_timestamp"].diff().dropna()
# print(gaps.describe())
# print(f"\nRows per second (approx): {len(df) / 300:.0f}")

# import pyarrow.parquet as pq
# import pandas as pd

# df = pq.read_table("data/trade_events/btc-updown-5m-1771624500.parquet").to_pandas()

# # Check if hash uniqueness can identify discrete trade events
# print(f"Total rows: {len(df)}")
# print(f"Unique hashes: {df['hash'].nunique()}")
# print(f"Rows with duplicate hash: {df.duplicated('hash').sum()}")

# # Look at what a 'trade' might actually be
# # Multiple price changes share a hash if they came from the same TradeEvent message
# hash_groups = df.groupby("hash").size()
# print(f"\nRows per hash:")
# print(hash_groups.describe())
# print(f"Hashes with exactly 2 rows: {(hash_groups == 2).sum()}")
# print(f"Hashes with exactly 1 row: {(hash_groups == 1).sum()}")

# # Sample a multi-row hash to see what it looks like
# multi = hash_groups[hash_groups > 1].index[0]
# print(f"\nSample multi-row hash group:")
# print(df[df["hash"] == multi][["exchange_timestamp","asset_id","side","trade_price","size","best_bid","best_ask"]].to_string())

# from pathlib import Path
# import pyarrow.parquet as pq
# import numpy as np

# book_files = list(Path("data/book_snapshots").glob("*.parquet"))
# print(f"Total book files: {len(book_files)}")

# sample = np.random.choice(book_files, min(10, len(book_files)), replace=False)
# row_counts = []
# for f in sample:
#     t = pq.read_table(f)
#     row_counts.append(len(t))
#     print(f"  {f.name}: {len(t):,} rows, ~{t.nbytes / 1e6:.1f} MB")

# print(f"\nMedian rows per market: {np.median(row_counts):,.0f}")
# print(f"Estimated total rows: {np.median(row_counts) * len(book_files):,.0f}")
# print(f"Estimated total uncompressed: {np.median(row_counts) * len(book_files) * 400 / 1e9:.2f} GB")

# # Also check what a snapshot looks like for the two tokens
# t = pq.read_table(book_files[0]).to_pandas()
# print(f"\nUnique asset_ids per market: {t['asset_id'].nunique()}")
# print(f"Rows per asset_id:\n{t['asset_id'].value_counts()}")

# gaps = t.groupby("asset_id")["exchange_timestamp"].diff().dropna()
# print(f"\nSnapshot interval (ms):")
# print(gaps.describe())

import pyarrow.parquet as pq
from pathlib import Path
import pandas as pd

f = next(Path("data/book_snapshots").glob("*.parquet"))
df = pq.read_table(f).to_pandas()

# Look at a single snapshot for one token
row = df.iloc[100]

print("mid_price:", row["mid_price"])
print("spread:   ", row["spread"])
print()

bid_cols = [(f"bid_price_{i}", f"bid_size_{i}") for i in range(1, 11)]
ask_cols = [(f"ask_price_{i}", f"ask_size_{i}") for i in range(1, 11)]

print("BIDS:")
for p, s in bid_cols:
    print(f"  {p}: {row[p]:.3f}  size: {row[s]:.2f}")

print("\nASKS:")
for p, s in ask_cols:
    print(f"  {p}: {row[p]:.3f}  size: {row[s]:.2f}")