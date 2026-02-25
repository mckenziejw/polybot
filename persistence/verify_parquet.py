import pyarrow.parquet as pq
import os

# Find the most recent files
book_files = sorted(os.listdir("data/book_snapshots"))
trade_files = sorted(os.listdir("data/trade_events"))

print(f"Book files: {book_files}")
print(f"Trade files: {trade_files}")

if book_files:
    book = pq.read_table(f"data/book_snapshots/{book_files[-1]}")
    print(f"\nBook snapshot rows: {len(book)}")
    print(f"Schema:\n{book.schema}")
    print(f"\nFirst row:\n{book.slice(0, 1).to_pydict()}")

if trade_files:
    trades = pq.read_table(f"data/trade_events/{trade_files[-1]}")
    print(f"\nTrade event rows: {len(trades)}")
    print(f"\nFirst row:\n{trades.slice(0, 1).to_pydict()}")