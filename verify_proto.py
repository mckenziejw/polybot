import json
import time
from polymarket_pb2 import (
    MarketEvent,
    BookSnapshot,
    TradeEvent,
    PriceChange,
    PriceLevel,
    Side,
)

# Real book snapshot message from your WebSocket output
raw_book = {
    "market": "0x0fca400440bf422a2f00c02240514d5916d47623cd7d8799183e2643c8edfc61",
    "asset_id": "41912778467520096380265408402892156614066447290800509965994604458487573204992",
    "timestamp": "1771335440042",
    "hash": "0483e44625fb19875584ed8913a0f76dc1a8399c",
    "bids": [
        {"price": "0.71", "size": "91.18"},
        {"price": "0.70", "size": "72"},
    ],
    "asks": [
        {"price": "0.73", "size": "34.99"},
        {"price": "0.74", "size": "449"},
    ],
    "tick_size": "0.01",
    "last_trade_price": "0.270",
    "event_type": "book"
}

# Real price_change message from your WebSocket output
raw_trade = {
    "market": "0x0fca400440bf422a2f00c02240514d5916d47623cd7d8799183e2643c8edfc61",
    "timestamp": "1771335440265",
    "event_type": "price_change",
    "price_changes": [
        {
            "asset_id": "41912778467520096380265408402892156614066447290800509965994604458487573204992",
            "price": "0.73",
            "size": "34.99",
            "side": "SELL",
            "hash": "0483e44625fb19875584ed8913a0f76dc1a8399c",
            "best_bid": "0.71",
            "best_ask": "0.73"
        }
    ]
}


def parse_book(raw: dict, received_ts: int) -> MarketEvent:
    snapshot = BookSnapshot(
        market=raw["market"],
        asset_id=raw["asset_id"],
        exchange_timestamp=int(raw["timestamp"]),
        hash=raw["hash"],
        bids=[PriceLevel(price=float(b["price"]), size=float(b["size"])) for b in raw["bids"]],
        asks=[PriceLevel(price=float(a["price"]), size=float(a["size"])) for a in raw["asks"]],
        tick_size=float(raw.get("tick_size", 0)),
        last_trade_price=float(raw.get("last_trade_price", 0)),
    )
    return MarketEvent(received_timestamp=received_ts, book=snapshot)


def parse_trade(raw: dict, received_ts: int) -> MarketEvent:
    def parse_side(side_str: str) -> Side:
        if side_str == "BUY":
            return Side.BUY
        elif side_str == "SELL":
            return Side.SELL
        return Side.SIDE_UNKNOWN

    price_changes = [
        PriceChange(
            asset_id=pc["asset_id"],
            price=float(pc["price"]),
            size=float(pc["size"]),
            side=parse_side(pc["side"]),
            hash=pc["hash"],
            best_bid=float(pc["best_bid"]),
            best_ask=float(pc["best_ask"]),
        )
        for pc in raw["price_changes"]
    ]
    trade = TradeEvent(
        market=raw["market"],
        exchange_timestamp=int(raw["timestamp"]),
        price_changes=price_changes,
    )
    return MarketEvent(received_timestamp=received_ts, trade=trade)


def verify_round_trip(event: MarketEvent, label: str):
    # Serialize to bytes
    serialized = event.SerializeToString()

    # Deserialize back
    restored = MarketEvent()
    restored.ParseFromString(serialized)

    # Verify key fields
    assert event.received_timestamp == restored.received_timestamp, "received_timestamp mismatch"
    assert event.WhichOneof("payload") == restored.WhichOneof("payload"), "payload type mismatch"

    print(f"\n--- {label} ---")
    print(f"Serialized size:     {len(serialized)} bytes")
    print(f"Payload type:        {event.WhichOneof('payload')}")
    print(f"received_timestamp:  {event.received_timestamp}")

    if event.HasField("book"):
        assert event.book.market == restored.book.market
        assert event.book.asset_id == restored.book.asset_id
        assert event.book.exchange_timestamp == restored.book.exchange_timestamp
        assert len(event.book.bids) == len(restored.book.bids)
        assert len(event.book.asks) == len(restored.book.asks)
        print(f"exchange_timestamp:  {event.book.exchange_timestamp}")
        print(f"market:              {event.book.market}")
        print(f"asset_id:            {event.book.asset_id}")
        print(f"bids:                {[(b.price, b.size) for b in event.book.bids]}")
        print(f"asks:                {[(a.price, a.size) for a in event.book.asks]}")
        print(f"tick_size:           {event.book.tick_size}")
        print(f"last_trade_price:    {event.book.last_trade_price}")

    elif event.HasField("trade"):
        assert event.trade.market == restored.trade.market
        assert event.trade.exchange_timestamp == restored.trade.exchange_timestamp
        assert len(event.trade.price_changes) == len(restored.trade.price_changes)
        print(f"exchange_timestamp:  {event.trade.exchange_timestamp}")
        print(f"market:              {event.trade.market}")
        print(f"price_changes:       {len(event.trade.price_changes)}")
        for pc in event.trade.price_changes:
            print(f"  asset_id:          {pc.asset_id}")
            print(f"  price:             {pc.price}")
            print(f"  size:              {pc.size}")
            print(f"  side:              {Side.Name(pc.side)}")
            print(f"  best_bid:          {pc.best_bid}")
            print(f"  best_ask:          {pc.best_ask}")

    print(f"Round-trip OK ✓")


if __name__ == "__main__":
    received_ts = int(time.time() * 1000)

    book_event = parse_book(raw_book, received_ts)
    trade_event = parse_trade(raw_trade, received_ts)

    verify_round_trip(book_event, "BookSnapshot")
    verify_round_trip(trade_event, "TradeEvent")

    # Sanity check: serialized bytes are meaningfully smaller than equivalent JSON
    book_json_size = len(json.dumps(raw_book).encode())
    book_proto_size = len(book_event.SerializeToString())
    print(f"\n--- Size comparison (book) ---")
    print(f"JSON:    {book_json_size} bytes")
    print(f"Protobuf: {book_proto_size} bytes")
    print(f"Ratio:   {book_json_size / book_proto_size:.1f}x smaller")