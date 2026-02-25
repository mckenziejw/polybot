import pandas as pd

df = pd.read_parquet("https://api.telonex.io/v1/datasets/polymarket/markets")

# Find your BTC 5-minute markets
btc_5m = df[df['slug'].str.contains('btc-updown-5m', na=False)]
print(f"BTC 5m markets found: {len(btc_5m)}")
print(f"Status breakdown:\n{btc_5m['status'].value_counts()}")
print(f"\nData availability:")
print(f"  With book_snapshot_5:  {(btc_5m['book_snapshot_5_from'] != '').sum()}")
print(f"  With book_snapshot_25: {(btc_5m['book_snapshot_25_from'] != '').sum()}")
print(f"  With trades:           {(btc_5m['trades_from'] != '').sum()}")
print(f"\nDate range:")
has_data = btc_5m[btc_5m['book_snapshot_5_from'] != '']
if len(has_data):
    print(f"  Snapshots from: {has_data['book_snapshot_5_from'].min()}")
    print(f"  Snapshots to:   {has_data['book_snapshot_5_to'].max()}")