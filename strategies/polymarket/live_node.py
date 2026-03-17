"""Live TradingNode for the Polymarket 5-minute sniper strategy.

Wires up:
  - Binance Futures data client (read-only, no API key needed)
  - Polymarket data + execution clients (needs wallet + API creds)
  - SniperStrategy with XGBoost models

Usage (run from project root):
    # Dry run (log predictions, no orders):
    .venv/bin/python -m strategies.polymarket.live_node

    # Paper trade with small size:
    .venv/bin/python -m strategies.polymarket.live_node --bet-dollars 5 --no-dry-run

    # Live:
    .venv/bin/python -m strategies.polymarket.live_node --bet-dollars 10 --no-dry-run --confidence 0.55
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from nautilus_trader.adapters.binance.common.enums import BinanceAccountType
from nautilus_trader.adapters.binance.config import BinanceDataClientConfig
from nautilus_trader.adapters.binance.factories import BinanceLiveDataClientFactory
from nautilus_trader.adapters.polymarket.config import (
    PolymarketDataClientConfig,
    PolymarketExecClientConfig,
    PolymarketInstrumentProviderConfig,
)
from nautilus_trader.adapters.polymarket.factories import (
    PolymarketLiveDataClientFactory,
    PolymarketLiveExecClientFactory,
)
from nautilus_trader.config import (
    InstrumentProviderConfig,
    LoggingConfig,
    TradingNodeConfig,
)
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.live.node import TradingNode

from .sniper_strategy import SniperStrategy, SniperStrategyConfig

# Resolve project root: live_node.py lives at strategies/polymarket/live_node.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _resolve_path(p: str) -> str:
    """Resolve a path relative to the project root."""
    path = Path(p)
    if path.is_absolute():
        return str(path)
    return str(PROJECT_ROOT / path)


def load_config(config_path: str = "config.json") -> dict:
    """Load credentials from config.json."""
    path = Path(_resolve_path(config_path))
    if not path.exists():
        print(f"Config file not found: {path}")
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


def build_node(args: argparse.Namespace) -> TradingNode:
    """Build a configured TradingNode."""
    cfg = load_config(args.config)
    pm = cfg["polymarket"]

    # Polymarket instrument provider with slug builder for UpDown 5m markets
    instrument_config = PolymarketInstrumentProviderConfig(
        event_slug_builder="strategies.polymarket.slug_builder:build_btc_5m_slugs",
    )

    # Polymarket data client
    pm_data_config = PolymarketDataClientConfig(
        instrument_config=instrument_config,
        private_key=pm["private_key"],
        funder=pm["proxy_wallet"],
        api_key=pm["api_key"],
        api_secret=pm["api_secret"],
        passphrase=pm["api_passphrase"],
        update_instruments_interval_mins=1,  # refresh every minute to catch new 5m markets
        ws_connection_initial_delay_secs=3,
    )

    # Binance Futures data client (read-only, public data)
    # Live adapter names instruments as BTCUSDT-PERP.BINANCE (not BTCUSDT.BINANCE)
    binance_data_config = BinanceDataClientConfig(
        account_type=BinanceAccountType.USDT_FUTURES,
        instrument_provider=InstrumentProviderConfig(
            load_ids=frozenset([InstrumentId.from_str("BTCUSDT-PERP.BINANCE")]),
        ),
    )

    # Resolve model directory to absolute path and validate
    model_dir = _resolve_path(args.model_dir)
    model_path = Path(model_dir)
    if not model_path.is_dir():
        print(f"Model directory not found: {model_path}")
        sys.exit(1)
    expected_model = model_path / f"window_{args.model_window}_seed_0.json"
    if not expected_model.exists():
        print(f"Model file not found: {expected_model}")
        sys.exit(1)

    # Strategy config
    strategy_config = SniperStrategyConfig(
        strategy_id="SNIPER-001",
        binance_bar_type="BTCUSDT-PERP.BINANCE-1-MINUTE-LAST-INTERNAL",
        model_dir=model_dir,
        model_window=args.model_window,
        confidence_threshold=args.confidence,
        max_entry_price=args.max_entry,
        bet_dollars=args.bet_dollars,
        daily_loss_limit=args.daily_loss_limit,
        dry_run=args.dry_run,
        min_warmup_candles=args.min_warmup,
    )

    # Build client configs — skip exec client in dry-run mode (avoids auth issues)
    data_clients = {
        "POLYMARKET": pm_data_config,
        "BINANCE": binance_data_config,
    }
    exec_clients = {}

    if not args.dry_run:
        pm_exec_config = PolymarketExecClientConfig(
            instrument_config=instrument_config,
            private_key=pm["private_key"],
            funder=pm["proxy_wallet"],
            api_key=pm["api_key"],
            api_secret=pm["api_secret"],
            passphrase=pm["api_passphrase"],
        )
        exec_clients["POLYMARKET"] = pm_exec_config

    # Log directory
    log_dir = _resolve_path("logs")

    # TradingNode
    node_config = TradingNodeConfig(
        trader_id="SNIPER-001",
        logging=LoggingConfig(
            log_level="INFO",
            log_level_file="DEBUG",
            log_directory=log_dir,
            log_file_format="json",
        ),
        data_clients=data_clients,
        exec_clients=exec_clients,
    )

    node = TradingNode(config=node_config)

    # Register adapter factories
    node.add_data_client_factory("POLYMARKET", PolymarketLiveDataClientFactory)
    node.add_data_client_factory("BINANCE", BinanceLiveDataClientFactory)
    if not args.dry_run:
        node.add_exec_client_factory("POLYMARKET", PolymarketLiveExecClientFactory)

    # Add strategy
    strategy = SniperStrategy(config=strategy_config)
    node.trader.add_strategy(strategy)

    return node


def main():
    parser = argparse.ArgumentParser(
        description="Polymarket 5-minute BTC sniper — NautilusTrader live node"
    )
    parser.add_argument("--config", default="config.json", help="Path to config.json")
    parser.add_argument("--model-dir", default="data/polymarket_5m_models_v2",
                        help="Directory with XGBoost model files")
    parser.add_argument("--model-window", type=int, default=31,
                        help="Walk-forward window index to load (default: 31 = latest)")
    parser.add_argument("--confidence", type=float, default=0.55,
                        help="Minimum prediction confidence to trade")
    parser.add_argument("--max-entry", type=float, default=0.60,
                        help="Maximum ask price to accept")
    parser.add_argument("--bet-dollars", type=float, default=10.0,
                        help="Bet size in dollars per trade")
    parser.add_argument("--daily-loss-limit", type=float, default=50.0,
                        help="Stop trading after this much daily loss")
    parser.add_argument("--min-warmup", type=int, default=60,
                        help="Minimum candles before trading (default: 60)")
    parser.add_argument("--dry-run", action="store_true", default=True,
                        help="Log predictions without placing orders (default)")
    parser.add_argument("--no-dry-run", dest="dry_run", action="store_false",
                        help="Enable live order placement")

    args = parser.parse_args()

    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    print(f"Model: {_resolve_path(args.model_dir)}/window_{args.model_window}_seed_*.json")
    print(f"Confidence: >= {args.confidence}")
    print(f"Max entry: <= ${args.max_entry}")
    print(f"Bet size: ${args.bet_dollars}")
    print(f"Daily loss limit: ${args.daily_loss_limit}")

    node = build_node(args)
    node.build()

    try:
        node.run()
    finally:
        node.dispose()


if __name__ == "__main__":
    main()
