"""Inspect data structures for timing analysis."""
import pandas as pd
import json

# Feature cache
df = pd.read_parquet('data/xgb_features_v2.parquet')
print("=== FEATURE CACHE ===")
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nobserve_time_s values: {sorted(df['observe_time_s'].unique()) if 'observe_time_s' in df.columns else 'NOT FOUND'}")
print(f"\nDtypes:\n{df.dtypes}")
print(f"\nSample rows:\n{df.head(2).to_string()}")

# Check for target/label column
label_candidates = [c for c in df.columns if any(x in c.lower() for x in ['label', 'target', 'resolv', 'outcome', 'won', 'up'])]
print(f"\nPossible label columns: {label_candidates}")

# Resolution cache
with open('data/resolution_cache.json') as f:
    rc = json.load(f)
print(f"\n=== RESOLUTION CACHE ===")
print(f"Total entries: {len(rc)}")
keys = list(rc.keys())[:3]
for k in keys:
    print(f"  {k} -> {rc[k]}")

# Raw parquet sample
import os
parquet_dir = 'data/telonex_book_snapshots'
files = sorted(os.listdir(parquet_dir))
print(f"\n=== RAW PARQUETS ===")
print(f"Total files: {len(files)}")
print(f"First 3: {files[:3]}")

sample = pd.read_parquet(os.path.join(parquet_dir, files[0]))
print(f"\nSample parquet shape: {sample.shape}")
print(f"Columns: {sample.columns.tolist()}")
print(f"Dtypes:\n{sample.dtypes}")
print(f"\nFirst 3 rows:\n{sample.head(3).to_string()}")
