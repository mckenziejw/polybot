"""Configuration for BTC directional classifier.

All hyperparameters in one place. Supports both 200ms LOB snapshots (v2)
and 1-minute resampled bars (v3) with regime features and triple-barrier
labeling.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DirectionalConfig:
    # ── Data ──────────────────────────────────────────────────────────
    snapshot_interval_ms: int = 200       # 200ms snapshots (matches Bybit WS)
    n_levels: int = 20                    # top 20 of 200 available levels

    # ── Prediction horizons (in snapshots) ────────────────────────────
    # 3=600ms, 5=1s, 25=5s, 150=30s
    horizons: list = field(default_factory=lambda: [3, 5, 25, 150])
    primary_horizon: int = 25             # 5s = primary trading horizon

    # ── Labels (fixed-horizon, kept for backward compat) ─────────────
    label_threshold_sigma: float = 0.5    # vol-adjusted up/down/flat threshold
    label_vol_window: int = 1500          # 5 min at 200ms for rolling vol

    # ── Triple Barrier ───────────────────────────────────────────────
    tp_sigma: float = 1.0                 # take-profit in rolling vol units
    sl_sigma: float = 1.0                 # stop-loss in rolling vol units
    max_barrier_horizon: int = 1500       # 5 min at 200ms
    barrier_vol_window: int = 1500        # backward-looking vol for thresholds

    # ── Regime Features ──────────────────────────────────────────────
    hurst_windows: list = field(default_factory=lambda: [150, 300, 1500])  # 30s, 60s, 5min

    # ── Walk-forward ──────────────────────────────────────────────────
    train_days: int = 90                  # shorter: only ~10 months total
    test_days: int = 7
    purge_seconds: int = 60              # 1 min gap train/test

    # ── Ensemble ──────────────────────────────────────────────────────
    n_seeds: int = 5
    seeds: list = field(default_factory=lambda: [42, 123, 456, 789, 1024])
    confidence_threshold: float = 0.60

    # ── XGBoost ───────────────────────────────────────────────────────
    max_depth: int = 3
    learning_rate: float = 0.01
    n_estimators: int = 2000
    early_stopping: int = 50
    subsample: float = 0.8
    colsample_bytree: float = 0.7
    min_child_weight: int = 50            # higher for dense data

    # ── Costs (Bybit VIP0 perps) ─────────────────────────────────────
    maker_fee: float = 0.00018
    taker_fee: float = 0.000495
    slippage_bps: float = 0.0

    # ── Backtest ──────────────────────────────────────────────────────
    notional_per_trade: float = 1000.0
    cooldown_snapshots: int = 25          # 5s between trades
    confidence_sweep: list = field(
        default_factory=lambda: [0.50, 0.55, 0.60, 0.65, 0.70]
    )

    # ── Paths ─────────────────────────────────────────────────────────
    orderbook_dir: Path = Path("data/bybit_orderbook")
    orderbook_raw_dir: Path = Path("data/bybit_orderbook_raw")
    trades_dir: Path = Path("data/bybit_trades_200ms")
    model_dir: Path = Path("data/directional_models")
    results_dir: Path = Path("data/directional_results")

    # ── Training subsample ────────────────────────────────────────────
    train_subsample: int = 5              # use every Nth snapshot for training
    # 1 = full 200ms, 5 = 1s, 25 = 5s

    def xgb_params(self, seed: int = 42) -> dict:
        """Return XGBoost parameter dict."""
        return {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "n_estimators": self.n_estimators,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "min_child_weight": self.min_child_weight,
            "seed": seed,
            "verbosity": 0,
            "tree_method": "hist",
            "n_jobs": -1,
        }

    def snapshots_per_day(self) -> int:
        """Number of 200ms snapshots in a day."""
        return 86400 * 1000 // self.snapshot_interval_ms  # 432,000

    def horizon_seconds(self, h: int) -> float:
        """Convert horizon in snapshots to seconds."""
        return h * self.snapshot_interval_ms / 1000.0

    @classmethod
    def config_1m(cls) -> "DirectionalConfig":
        """Config for 1-minute resampled bars with longer horizons.

        All window parameters are in 1-min bar units.
        Target horizons: 15m, 30m, 60m where moves >> fees.
        """
        return cls(
            snapshot_interval_ms=60_000,
            n_levels=20,
            # Horizons in 1-min bars: 15m, 30m, 60m
            horizons=[15, 30, 60],
            primary_horizon=30,
            # Labels
            label_threshold_sigma=0.5,
            label_vol_window=60,           # 1h lookback for flat-zone vol
            # Triple barrier: 1h max horizon at 1-min bars
            tp_sigma=1.0,
            sl_sigma=1.0,
            max_barrier_horizon=60,        # 1 hour
            barrier_vol_window=60,         # 1h lookback for barrier vol
            # Regime: 15m, 30m, 1h windows
            hurst_windows=[15, 30, 60],
            # Walk-forward
            train_days=90,
            test_days=7,
            purge_seconds=3600,            # 1h purge gap
            # Ensemble (same)
            n_seeds=5,
            seeds=[42, 123, 456, 789, 1024],
            confidence_threshold=0.60,
            # XGBoost (same)
            max_depth=3,
            learning_rate=0.01,
            n_estimators=2000,
            early_stopping=50,
            subsample=0.8,
            colsample_bytree=0.7,
            min_child_weight=50,
            # Costs (same)
            maker_fee=0.00018,
            taker_fee=0.000495,
            slippage_bps=0.0,
            # Backtest
            notional_per_trade=1000.0,
            cooldown_snapshots=5,          # 5 min cooldown
            confidence_sweep=[0.50, 0.55, 0.60, 0.65, 0.70],
            # Paths — v3 dirs
            orderbook_dir=Path("data/bybit_orderbook_raw"),
            orderbook_raw_dir=Path("data/bybit_orderbook_raw"),
            trades_dir=Path("data/bybit_trades_200ms"),
            model_dir=Path("data/directional_models_v3"),
            results_dir=Path("data/directional_results_v3"),
            # No subsampling needed — already at 1-min
            train_subsample=1,
        )
