#!/bin/bash
# Backtest DYDX and CRV directional predictions using candle-based price simulation.
#
# Binance Futures fees (VIP0): maker 0.02%, taker 0.05%
# We use candle close prices with synthetic spread.
#
# Usage: bash run_alt_backtest.sh

set -euo pipefail
cd /home/ubuntu
source venv/bin/activate
export PYTHONUNBUFFERED=1

echo "=== Alt-Coin Directional Backtests ==="
echo "Started: $(date -u)"
echo ""

# Binance Futures VIP0 fees
TAKER_FEE=0.0005
MAKER_FEE=0.0002

for symbol_pair in "DYDX-USDT-PERP:dydx:3.0" "CRV-USDT-PERP:crv:3.0"; do
    IFS=: read -r symbol sym_short spread_bps <<< "$symbol_pair"

    results_dir="data/directional_results_alt_${sym_short}"

    echo "============================================================"
    echo "Backtest: $symbol (spread=${spread_bps}bps)"
    echo "============================================================"

    if [ ! -f "${results_dir}/predictions.parquet" ]; then
        echo "  No predictions found at ${results_dir}, skipping."
        continue
    fi

    python3 -m strategies.directional.backtest \
        --results-dir "$results_dir" \
        --data-dir data/cryptolake \
        --candles \
        --symbol "$symbol" \
        --spread-bps "$spread_bps" \
        --resolution 1m \
        --taker-fee "$TAKER_FEE" \
        --maker-fee "$MAKER_FEE" \
        2>&1

    echo ""
done

echo "=== Backtests complete at $(date -u) ==="
