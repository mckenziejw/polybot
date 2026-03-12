"""
Build sample datasets for swarm agents.
Run on the cloud instance where memory is plentiful.

Usage:
    python3 build_sample.py [--size 1000] [--truncate-sec 0]

Outputs:
    data/sample_1k/book_snapshots_sample_1k.parquet
    data/sample_1k/resolution_cache_sample.json
"""

import pandas as pd
import glob
import json
import random
import os
import time
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=1000, help="Number of markets to sample")
    parser.add_argument("--truncate-sec", type=int, default=0,
                        help="Only keep first N seconds per market (0 = full market)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", default="data/telonex_btc_5m_book_snapshots")
    parser.add_argument("--resolution-cache", default="data/resolution_cache.json")
    parser.add_argument("--output-dir", default="data/sample_1k")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.resolution_cache) as f:
        rc = json.load(f)

    files = sorted(glob.glob(f"{args.data_dir}/*.parquet"))
    print(f"Total files: {len(files)}")

    random.seed(args.seed)
    sample_files = random.sample(files, min(args.size, len(files)))
    print(f"Sampling {len(sample_files)} files (truncate={args.truncate_sec}s)...")

    out_path = os.path.join(args.output_dir, "book_snapshots_sample_1k.parquet")
    rc_path = os.path.join(args.output_dir, "resolution_cache_sample.json")

    # Process in batches to control memory
    batch_size = 100
    all_chunks = []
    sample_slugs = []
    skipped = 0
    t0 = time.time()

    for batch_start in range(0, len(sample_files), batch_size):
        batch = sample_files[batch_start:batch_start + batch_size]
        dfs = []
        for f in batch:
            df = pd.read_parquet(f)
            slug = df["slug"].iloc[0]
            if slug not in rc:
                skipped += 1
                continue

            if args.truncate_sec > 0:
                epoch_ms = int(slug.split("-")[-1]) * 1000
                df = df[df["exchange_timestamp"] <= epoch_ms + args.truncate_sec * 1000]

            dfs.append(df)
            sample_slugs.append(slug)

        if dfs:
            all_chunks.append(pd.concat(dfs, ignore_index=True))
            del dfs

        done = min(batch_start + batch_size, len(sample_files))
        print(f"  Processed {done}/{len(sample_files)}, kept {len(sample_slugs)}")

    merged = pd.concat(all_chunks, ignore_index=True)
    del all_chunks

    elapsed = time.time() - t0
    print(f"\nLoaded {len(sample_slugs)} markets ({skipped} skipped) in {elapsed:.1f}s")
    print(f"Shape: {merged.shape}")

    merged.to_parquet(out_path)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"Saved to {out_path} ({size_mb:.1f} MB)")

    sample_rc = {s: rc[s] for s in sample_slugs if s in rc}
    with open(rc_path, "w") as f:
        json.dump(sample_rc, f)
    print(f"Resolution cache: {len(sample_rc)} entries -> {rc_path}")


if __name__ == "__main__":
    main()
