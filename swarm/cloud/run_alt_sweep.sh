#!/bin/bash
# Alt-coin sweep: preprocess + train for multiple Binance Futures symbols.
#
# Tests whether mid/low-cap alts have more predictable directional signal
# than BTC using the same OHLCV + derivatives feature pipeline.
#
# Usage: nohup bash run_alt_sweep.sh &

set -euo pipefail
cd /home/ubuntu
source venv/bin/activate
export PYTHONUNBUFFERED=1

TOTAL_CORES=$(nproc)
SYMBOLS=(PENDLE-USDT-PERP DYDX-USDT-PERP CRV-USDT-PERP JASMY-USDT-PERP)
TRAIN_DAYS=60

echo "=== Alt-Coin Directional Sweep ==="
echo "Symbols: ${SYMBOLS[*]}"
echo "Cores: $TOTAL_CORES"
echo "Training window: ${TRAIN_DAYS} days"
echo "Started: $(date -u)"
echo ""

for symbol in "${SYMBOLS[@]}"; do
    sym_short=$(echo "$symbol" | cut -d- -f1 | tr '[:upper:]' '[:lower:]')

    echo "============================================================"
    echo "Symbol: $symbol ($sym_short)"
    echo "Started: $(date -u)"
    echo "============================================================"

    # Step 1: Preprocess (OHLCV + derivatives, no cross-asset for alts)
    echo "  Preprocessing..."
    python3 -m pipelines.preprocess_features \
        --variant ohlcv \
        --symbol "$symbol" \
        --no-resume \
        2>&1 | tail -10
    echo ""

    # Step 2: Train
    features_dir="data/directional_features_v3_ohlcv_${sym_short}"
    results_dir="data/directional_results_alt_${sym_short}"
    model_dir="data/directional_models_alt_${sym_short}"

    echo "  Training (${TRAIN_DAYS}d windows)..."
    python3 -m strategies.directional.train \
        --variant ohlcv \
        --resolution 1m \
        --train-days "$TRAIN_DAYS" \
        --n-jobs "$TOTAL_CORES" \
        --features-dir "$features_dir" \
        --results-dir "$results_dir" \
        --model-dir "$model_dir" \
        > "/tmp/train_alt_${sym_short}.log" 2>&1

    # Extract results
    acc=$(grep "Overall accuracy" "/tmp/train_alt_${sym_short}.log" 2>/dev/null | awk '{print $NF}')
    auc=$(grep "Overall AUC" "/tmp/train_alt_${sym_short}.log" 2>/dev/null | awk '{print $NF}')
    n=$(grep "Total test" "/tmp/train_alt_${sym_short}.log" 2>/dev/null | awk '{print $NF}')

    echo "  Results: acc=${acc:-N/A} auc=${auc:-N/A} n=${n:-N/A}"
    echo "  Finished: $(date -u)"
    echo ""
done

# Summary
echo ""
echo "=========================================="
echo "=== ALT-COIN DIRECTIONAL SWEEP SUMMARY ==="
echo "=========================================="
echo ""
printf "%-20s %8s %8s %12s %8s\n" "Symbol" "Acc" "AUC" "N" "Windows"
printf "%-20s %8s %8s %12s %8s\n" "--------------------" "--------" "--------" "------------" "--------"

for symbol in "${SYMBOLS[@]}"; do
    sym_short=$(echo "$symbol" | cut -d- -f1 | tr '[:upper:]' '[:lower:]')
    log="/tmp/train_alt_${sym_short}.log"
    if [ -f "$log" ]; then
        acc=$(grep "Overall accuracy" "$log" | awk '{print $NF}')
        auc=$(grep "Overall AUC" "$log" | awk '{print $NF}')
        n=$(grep "Total test" "$log" | awk '{print $NF}')
        nw=$(grep -c "^  W" "$log" 2>/dev/null || echo "?")
        printf "%-20s %8s %8s %12s %8s\n" "$symbol" "${acc:-N/A}" "${auc:-N/A}" "${n:-N/A}" "$nw"
    fi
done

# Also print top features per symbol
echo ""
echo "=== Top 5 features per symbol ==="
for symbol in "${SYMBOLS[@]}"; do
    sym_short=$(echo "$symbol" | cut -d- -f1 | tr '[:upper:]' '[:lower:]')
    log="/tmp/train_alt_${sym_short}.log"
    if [ -f "$log" ]; then
        echo ""
        echo "$symbol:"
        # Get feature importance from the last window
        grep -A 6 "Top 15 features" "$log" | tail -6 | head -5
    fi
done

touch /tmp/alt_sweep_complete
echo ""
echo "=== Alt sweep complete at $(date -u) ==="
