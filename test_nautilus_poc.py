"""
NautilusTrader proof-of-concept: load a Polymarket BTC 5m market and run a trivial strategy.
"""

import asyncio

from nautilus_trader.adapters.polymarket.loaders import PolymarketDataLoader
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.config import BacktestEngineConfig, LoggingConfig, StrategyConfig
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import QuoteTick, TradeTick
from nautilus_trader.model.enums import AccountType, BookType, OmsType, OrderSide
from nautilus_trader.model.identifiers import InstrumentId, Venue
from nautilus_trader.model.objects import Money, Price, Quantity
from nautilus_trader.trading.strategy import Strategy


# --- Trivial strategy: buy on every Nth trade tick ---

class BuyEveryNConfig(StrategyConfig, frozen=True):
    instrument_id: str
    trade_interval: int = 50
    quantity: float = 10.0


class BuyEveryN(Strategy):
    """Dead-simple strategy: place a market buy every N trade ticks."""

    def __init__(self, config: BuyEveryNConfig) -> None:
        super().__init__(config)
        self.instrument_id_str = config.instrument_id
        self.trade_interval = config.trade_interval
        self.quantity = config.quantity
        self.tick_count = 0

    def on_start(self) -> None:
        iid = InstrumentId.from_str(self.instrument_id_str)
        self.subscribe_trade_ticks(iid)
        self.log.info(f"Strategy started, buying every {self.trade_interval} ticks")

    def on_trade_tick(self, tick: TradeTick) -> None:
        self.tick_count += 1
        if self.tick_count % self.trade_interval == 0:
            order = self.order_factory.market(
                instrument_id=tick.instrument_id,
                order_side=OrderSide.BUY,
                quantity=Quantity(self.quantity, precision=6),
            )
            self.submit_order(order)
            self.log.info(
                f"Tick #{self.tick_count}: BUY {self.quantity} @ {tick.price}"
            )

    def on_stop(self) -> None:
        self.log.info(f"Strategy stopped. Total ticks: {self.tick_count}")


def trades_to_quotes(trades: list[TradeTick], spread_ticks: int = 1) -> list[QuoteTick]:
    """Convert trade ticks to quote ticks by creating a synthetic spread around trade price."""
    quotes = []
    for t in trades:
        price_val = float(str(t.price))
        # Create a 1-tick spread around the trade price
        bid = price_val
        ask = price_val + 0.001 * spread_ticks  # 1 tick = 0.001 for 3-decimal precision
        quotes.append(
            QuoteTick(
                instrument_id=t.instrument_id,
                bid_price=Price(bid, precision=3),
                ask_price=Price(ask, precision=3),
                bid_size=Quantity(100, precision=6),
                ask_size=Quantity(100, precision=6),
                ts_event=t.ts_event,
                ts_init=t.ts_init,
            )
        )
    return quotes


async def main():
    print("Fetching BTC 5m market from Polymarket API...")

    loader = await PolymarketDataLoader.from_market_slug(
        slug="btc-updown-5m-1773013800",
        token_index=0,  # Up token
    )

    instrument = loader.instrument
    print(f"Instrument: {instrument.id}")
    print(f"  Outcome: {instrument.outcome}")
    print(f"  Price precision: {instrument.price_precision}")
    print(f"  Size precision: {instrument.size_precision}")
    print(f"  Min quantity: {instrument.min_quantity}")
    print(f"  Maker fee: {instrument.maker_fee}")
    print(f"  Taker fee: {instrument.taker_fee}")

    # Load historical trades
    print("\nLoading historical trades...")
    trades = await loader.load_trades()
    print(f"Loaded {len(trades)} trades")

    if len(trades) < 10:
        print(f"Only {len(trades)} trades — not enough for backtest")
        return

    print(f"Time range: {trades[0].ts_event} -> {trades[-1].ts_event}")
    print(f"Price range: {min(float(str(t.price)) for t in trades):.3f} -> {max(float(str(t.price)) for t in trades):.3f}")

    # Convert trades to quotes so the matching engine has bid/ask
    quotes = trades_to_quotes(trades)

    # Set up backtest engine
    print("\nConfiguring backtest engine...")

    engine = BacktestEngine(
        config=BacktestEngineConfig(
            logging=LoggingConfig(log_level="WARNING"),
        ),
    )

    POLYMARKET = Venue("POLYMARKET")
    engine.add_venue(
        venue=POLYMARKET,
        oms_type=OmsType.NETTING,
        account_type=AccountType.CASH,
        starting_balances=[Money(1000, USD)],
        book_type=BookType.L1_MBP,
    )

    engine.add_instrument(instrument)
    engine.add_data(trades)
    engine.add_data(quotes)

    trade_interval = max(len(trades) // 10, 1)
    strategy = BuyEveryN(
        config=BuyEveryNConfig(
            instrument_id=str(instrument.id),
            trade_interval=trade_interval,
            quantity=10.0,
        ),
    )
    engine.add_strategy(strategy)

    # Run
    print(f"Running backtest ({len(trades)} trades, buying every {trade_interval})...")
    engine.run()

    # Results
    print("\n=== RESULTS ===")
    print(f"Total orders: {engine.trader.generate_order_fills_report().shape[0]}")
    print(f"Total positions: {engine.trader.generate_positions_report().shape[0]}")

    fills = engine.trader.generate_order_fills_report()
    if not fills.empty:
        print(f"\nOrder Fills:")
        print(fills.to_string())

    positions = engine.trader.generate_positions_report()
    if not positions.empty:
        print(f"\nPositions:")
        print(positions.to_string())

    account = engine.trader.generate_account_report(POLYMARKET)
    if not account.empty:
        print(f"\nAccount:")
        print(account.to_string())


if __name__ == "__main__":
    asyncio.run(main())
