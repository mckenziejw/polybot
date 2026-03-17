"""
Load historical Binance quote data from parquet files into NautilusTrader types.

Converts our 1-second bid/ask parquet data into QuoteTick objects and optionally
creates Bar aggregations for indicator computation.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import BarType, QuoteTick
from nautilus_trader.model.enums import (
    AccountType,
    AggregationSource,
    BarAggregation,
    OmsType,
    PriceType,
)
from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue
from nautilus_trader.model.instruments import CurrencyPair
from nautilus_trader.model.objects import Currency, Money, Price, Quantity


BINANCE = Venue("BINANCE")

# Asset configs: (symbol, base_currency, quote file path pattern)
ASSETS = {
    "BTCUSDT": {"base": "BTC", "file": "btcusdt_quotes.parquet", "dir": "btc_quotes"},
    "ETHUSDT": {"base": "ETH", "file": "ethusdt_quotes.parquet", "dir": "eth_quotes"},
    "SOLUSDT": {"base": "SOL", "file": "solusdt_quotes.parquet", "dir": "sol_quotes"},
    "XRPUSDT": {"base": "XRP", "file": "xrpusdt_quotes.parquet", "dir": "xrp_quotes"},
}

# Precision configs per asset
PRECISION = {
    "BTCUSDT": {"price": 2, "size": 5},   # BTC: $84,123.45, 0.00001 BTC
    "ETHUSDT": {"price": 2, "size": 4},   # ETH: $2,345.67, 0.0001 ETH
    "SOLUSDT": {"price": 2, "size": 2},   # SOL: $145.23, 0.01 SOL
    "XRPUSDT": {"price": 4, "size": 1},   # XRP: $0.5432, 0.1 XRP
}


def create_instrument(symbol: str) -> CurrencyPair:
    """Create a NautilusTrader CurrencyPair instrument for a Binance spot pair."""
    prec = PRECISION[symbol]
    asset_info = ASSETS[symbol]

    base_currency = Currency.from_str(asset_info["base"])
    quote_currency = Currency.from_str("USDT")

    return CurrencyPair(
        instrument_id=InstrumentId(Symbol(symbol), BINANCE),
        raw_symbol=Symbol(symbol),
        base_currency=base_currency,
        quote_currency=quote_currency,
        price_precision=prec["price"],
        size_precision=prec["size"],
        price_increment=Price(10 ** -prec["price"], precision=prec["price"]),
        size_increment=Quantity(10 ** -prec["size"], precision=prec["size"]),
        lot_size=None,
        max_quantity=None,
        min_quantity=Quantity(10 ** -prec["size"], precision=prec["size"]),
        max_notional=None,
        min_notional=Money(10, quote_currency),
        max_price=None,
        min_price=Price(10 ** -prec["price"], precision=prec["price"]),
        margin_init=None,
        margin_maint=None,
        maker_fee=Decimal("0.0001"),
        taker_fee=Decimal("0.0004"),
        ts_event=0,
        ts_init=0,
    )


def load_quotes_from_parquet(
    data_dir: Path,
    symbol: str,
    start_ms: int | None = None,
    end_ms: int | None = None,
) -> tuple[CurrencyPair, list[QuoteTick]]:
    """
    Load historical Binance quotes from parquet into NautilusTrader QuoteTicks.

    Parameters
    ----------
    data_dir : Path
        Root data directory (e.g., Path("data/"))
    symbol : str
        Trading pair symbol (e.g., "BTCUSDT")
    start_ms : int, optional
        Start timestamp in milliseconds (inclusive)
    end_ms : int, optional
        End timestamp in milliseconds (inclusive)

    Returns
    -------
    tuple[CurrencyPair, list[QuoteTick]]
        The instrument and list of quote ticks sorted chronologically.
    """
    asset_info = ASSETS[symbol]
    prec = PRECISION[symbol]
    parquet_path = data_dir / asset_info["dir"] / asset_info["file"]

    df = pd.read_parquet(parquet_path)

    # Apply time filters
    if start_ms is not None:
        df = df[df["timestamp_ms"] >= start_ms]
    if end_ms is not None:
        df = df[df["timestamp_ms"] <= end_ms]

    # Create instrument
    instrument = create_instrument(symbol)
    instrument_id = instrument.id
    price_prec = prec["price"]
    size_prec = prec["size"]
    n = len(df)

    # Use Cython batch constructor for ~100x speedup over Python loop
    ts_nanos = (df["timestamp_ms"].values * 1_000_000).astype(np.uint64)

    quotes = QuoteTick.from_raw_arrays_to_list(
        instrument_id=instrument_id,
        price_prec=price_prec,
        size_prec=size_prec,
        bid_prices_raw=np.ascontiguousarray(df["bid_price"].values, dtype=np.float64),
        ask_prices_raw=np.ascontiguousarray(df["ask_price"].values, dtype=np.float64),
        bid_sizes_raw=np.ascontiguousarray(np.ones(n), dtype=np.float64),
        ask_sizes_raw=np.ascontiguousarray(np.ones(n), dtype=np.float64),
        ts_events=np.ascontiguousarray(ts_nanos),
        ts_inits=np.ascontiguousarray(ts_nanos),
    )

    return instrument, quotes


def load_multi_asset(
    data_dir: Path,
    symbols: list[str] | None = None,
    start_ms: int | None = None,
    end_ms: int | None = None,
) -> dict[str, tuple[CurrencyPair, list[QuoteTick]]]:
    """
    Load multiple assets from parquet files.

    Parameters
    ----------
    data_dir : Path
        Root data directory
    symbols : list[str], optional
        List of symbols to load. Defaults to all available.
    start_ms : int, optional
        Start timestamp filter
    end_ms : int, optional
        End timestamp filter

    Returns
    -------
    dict[str, tuple[CurrencyPair, list[QuoteTick]]]
        Mapping of symbol -> (instrument, quotes)
    """
    if symbols is None:
        symbols = list(ASSETS.keys())

    results = {}
    for symbol in symbols:
        if symbol not in ASSETS:
            raise ValueError(f"Unknown symbol: {symbol}. Available: {list(ASSETS.keys())}")
        results[symbol] = load_quotes_from_parquet(data_dir, symbol, start_ms, end_ms)

    return results


def setup_backtest_engine(
    engine: BacktestEngine,
    data_dir: Path,
    symbols: list[str] | None = None,
    start_ms: int | None = None,
    end_ms: int | None = None,
    starting_balance: float = 10_000.0,
) -> dict[str, CurrencyPair]:
    """
    Load Binance data into a BacktestEngine.

    Adds the BINANCE venue, instruments, and quote tick data.
    Returns the loaded instruments for strategy configuration.

    Parameters
    ----------
    engine : BacktestEngine
        The engine to populate
    data_dir : Path
        Root data directory
    symbols : list[str], optional
        Symbols to load (default: all)
    start_ms : int, optional
        Start timestamp filter
    end_ms : int, optional
        End timestamp filter
    starting_balance : float
        Starting USDT balance

    Returns
    -------
    dict[str, CurrencyPair]
        Loaded instruments keyed by symbol
    """
    # Add venue
    usdt = Currency.from_str("USDT")
    engine.add_venue(
        venue=BINANCE,
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        starting_balances=[Money(starting_balance, usdt)],
    )

    # Load data for each symbol
    all_data = load_multi_asset(data_dir, symbols, start_ms, end_ms)

    instruments = {}
    for symbol, (instrument, quotes) in all_data.items():
        engine.add_instrument(instrument)
        engine.add_data(quotes, sort=False)
        instruments[symbol] = instrument

    engine.sort_data()

    return instruments


def make_bar_type(
    symbol: str,
    period_minutes: int = 1,
    price_type: PriceType = PriceType.MID,
) -> BarType:
    """
    Create a BarType for bar aggregation from quote ticks.

    Parameters
    ----------
    symbol : str
        Trading pair (e.g., "BTCUSDT")
    period_minutes : int
        Bar period in minutes (1, 5, 15, 60, etc.)
    price_type : PriceType
        Which price to aggregate (MID, BID, ASK)

    Returns
    -------
    BarType
    """
    from nautilus_trader.model.data import BarSpecification

    instrument_id = InstrumentId(Symbol(symbol), BINANCE)

    # NautilusTrader requires HOUR aggregation for >= 60 min periods
    if period_minutes >= 60 and period_minutes % 60 == 0:
        hours = period_minutes // 60
        spec = BarSpecification(hours, BarAggregation.HOUR, price_type)
    else:
        spec = BarSpecification(period_minutes, BarAggregation.MINUTE, price_type)

    return BarType(instrument_id, spec, AggregationSource.INTERNAL)
