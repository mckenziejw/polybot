#!/bin/bash
# Tier 1+2 DeFi alt sweep: download + preprocess + train + backtest.
#
# Tests DeFi protocol tokens similar to DYDX (70% acc) and CRV (62% acc).
#
# Usage: nohup bash run_alt_sweep_t2.sh &

set -euo pipefail
cd /home/ubuntu
source venv/bin/activate
export PYTHONUNBUFFERED=1

TOTAL_CORES=$(nproc)
TRAIN_DAYS=60
TAKER_FEE=0.0005
MAKER_FEE=0.0002
SPREAD_BPS=3.0

# Tier 1: closest to DYDX/CRV profile
# Tier 2: strong DeFi but higher volume
SYMBOLS=(
    SNX-USDT-PERP
    COMP-USDT-PERP
    SUSHI-USDT-PERP
    BAL-USDT-PERP
    RUNE-USDT-PERP
    FXS-USDT-PERP
    MKR-USDT-PERP
    LDO-USDT-PERP
    CAKE-USDT-PERP
    UNI-USDT-PERP
    AAVE-USDT-PERP
    1INCH-USDT-PERP
    YFI-USDT-PERP
)

echo "=== DeFi Alt-Coin Sweep (Tier 1+2) ==="
echo "Symbols: ${SYMBOLS[*]}"
echo "Cores: $TOTAL_CORES"
echo "Training window: ${TRAIN_DAYS} days"
echo "Started: $(date -u)"
echo ""

# ── Step 1: Download data ──────────────────────────────────────────
echo "=== Step 1: Downloading CryptoLake data ==="
python3 -m pipelines.download_cryptolake \
    --symbols ${SYMBOLS[*]} \
    --data-types candles funding open_interest \
    --start 2025-06-01 \
    --end 2026-03-17 \
    --output-dir data/cryptolake \
    2>&1 | tail -30
echo ""

# ── Step 2: Preprocess + Train + Backtest each symbol ──────────────
declare -A RESULTS

for symbol in "${SYMBOLS[@]}"; do
    sym_short=$(echo "$symbol" | cut -d- -f1 | tr '[:upper:]' '[:lower:]')
    features_dir="data/directional_features_v3_ohlcv_${sym_short}"
    results_dir="data/directional_results_alt_${sym_short}"
    model_dir="data/directional_models_alt_${sym_short}"

    echo "============================================================"
    echo "Symbol: $symbol ($sym_short)"
    echo "Started: $(date -u)"
    echo "============================================================"

    # Preprocess
    echo "  Preprocessing..."
    rm -rf "$features_dir"
    python3 -m pipelines.preprocess_features \
        --variant ohlcv \
        --symbol "$symbol" \
        --no-resume \
        2>&1 | tail -5

    # Train
    echo "  Training..."
    python3 -m strategies.directional.train \
        --variant ohlcv \
        --resolution 1m \
        --train-days "$TRAIN_DAYS" \
        --n-jobs "$TOTAL_CORES" \
        --features-dir "$features_dir" \
        --results-dir "$results_dir" \
        --model-dir "$model_dir" \
        > "/tmp/train_alt_${sym_short}.log" 2>&1

    acc=$(grep "Overall accuracy" "/tmp/train_alt_${sym_short}.log" 2>/dev/null | awk '{print $NF}')
    auc=$(grep "Overall AUC" "/tmp/train_alt_${sym_short}.log" 2>/dev/null | awk '{print $NF}')
    n=$(grep "Total test" "/tmp/train_alt_${sym_short}.log" 2>/dev/null | awk '{print $NF}')
    RESULTS[$symbol]="acc=${acc:-N/A} auc=${auc:-N/A} n=${n:-N/A}"
    echo "  Train results: ${RESULTS[$symbol]}"

    # Backtest (only if accuracy > 55%)
    if [ -f "${results_dir}/predictions.parquet" ]; then
        acc_num=$(echo "${acc:-0}" | tr -d ' ')
        if (( $(echo "$acc_num > 0.55" | bc -l 2>/dev/null || echo 0) )); then
            echo "  Backtesting (acc > 55%)..."
            python3 -m strategies.directional.backtest \
                --results-dir "$results_dir" \
                --data-dir data/cryptolake \
                --candles \
                --symbol "$symbol" \
                --spread-bps "$SPREAD_BPS" \
                --resolution 1m \
                --taker-fee "$TAKER_FEE" \
                --maker-fee "$MAKER_FEE" \
                2>&1 | grep -E "(Conf|0\.[567]0)" || true
        else
            echo "  Skipping backtest (acc <= 55%)"
        fi
    fi

    echo "  Finished: $(date -u)"
    echo ""
done

# ── Summary ────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "=== DEFI ALT SWEEP SUMMARY ==="
echo "=========================================="
echo ""
printf "%-20s %8s %8s %12s\n" "Symbol" "Acc" "AUC" "N"
printf "%-20s %8s %8s %12s\n" "--------------------" "--------" "--------" "------------"

for symbol in "${SYMBOLS[@]}"; do
    sym_short=$(echo "$symbol" | cut -d- -f1 | tr '[:upper:]' '[:lower:]')
    log="/tmp/train_alt_${sym_short}.log"
    if [ -f "$log" ]; then
        acc=$(grep "Overall accuracy" "$log" | awk '{print $NF}')
        auc=$(grep "Overall AUC" "$log" | awk '{print $NF}')
        n=$(grep "Total test" "$log" | awk '{print $NF}')
        printf "%-20s %8s %8s %12s\n" "$symbol" "${acc:-N/A}" "${auc:-N/A}" "${n:-N/A}"
    fi
done

# Top features for winners
echo ""
echo "=== Top 5 features for symbols with acc > 55% ==="
for symbol in "${SYMBOLS[@]}"; do
    sym_short=$(echo "$symbol" | cut -d- -f1 | tr '[:upper:]' '[:lower:]')
    log="/tmp/train_alt_${sym_short}.log"
    if [ -f "$log" ]; then
        acc=$(grep "Overall accuracy" "$log" | awk '{print $NF}')
        acc_num=$(echo "${acc:-0}" | tr -d ' ')
        if (( $(echo "$acc_num > 0.55" | bc -l 2>/dev/null || echo 0) )); then
            echo ""
            echo "$symbol (acc=$acc):"
            grep -A 6 "Top 15 features" "$log" | tail -6 | head -5
        fi
    fi
done

touch /tmp/alt_sweep_t2_complete
echo ""
echo "=== Sweep complete at $(date -u) ==="
