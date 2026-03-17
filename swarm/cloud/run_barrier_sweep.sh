#!/bin/bash
# Run barrier sweep: preprocess features once (OHLCV+derivatives with new higher-order features),
# then train across multiple asymmetric barrier configs.
#
# Usage: nohup bash run_barrier_sweep.sh &

set -euo pipefail
cd /home/ubuntu
source venv/bin/activate
export PYTHONUNBUFFERED=1

TOTAL_CORES=$(nproc)
echo "=== Barrier Sweep ==="
echo "Cores: $TOTAL_CORES"
echo "Started: $(date -u)"
echo ""

# ── Step 1: Preprocess OHLCV features (with new higher-order derivatives) ──
echo "=== Step 1: Preprocessing OHLCV features ==="
rm -rf data/directional_features_v3_ohlcv
python3 -m pipelines.preprocess_features --variant ohlcv --no-resume 2>&1 | tail -5
echo ""

# ── Step 2: Run barrier sweep ──────────────────────────────────────────────
# Each config runs sequentially (labels recomputed per config).
# Within each run, XGBoost gets full cores since only one runs at a time.

# Barrier configs: (tp_sigma, sl_sigma, description)
# Symmetric baselines
CONFIGS=(
    "1.0:1.0:symmetric_1.0"
    "1.5:1.5:symmetric_1.5"
    "2.0:2.0:symmetric_2.0"
    # Asymmetric: wide TP, tight SL (trend following — need to be right less often)
    "2.0:0.5:trend_2.0_0.5"
    "2.0:1.0:trend_2.0_1.0"
    "3.0:1.0:trend_3.0_1.0"
    # Asymmetric: tight TP, wide SL (mean reversion — frequent small wins)
    "0.5:2.0:revert_0.5_2.0"
    "1.0:2.0:revert_1.0_2.0"
    "1.0:3.0:revert_1.0_3.0"
)

echo "=== Step 2: Training ${#CONFIGS[@]} barrier configs ==="
echo ""

for config in "${CONFIGS[@]}"; do
    IFS=':' read -r tp sl desc <<< "$config"
    echo "──────────────────────────────────────────────────────────"
    echo "Config: $desc (tp=${tp}σ, sl=${sl}σ)"
    echo "Started: $(date -u)"
    echo "──────────────────────────────────────────────────────────"

    python3 -m strategies.directional.train \
        --variant ohlcv \
        --resolution 1m \
        --tp-sigma "$tp" \
        --sl-sigma "$sl" \
        --n-jobs "$TOTAL_CORES" \
        > "/tmp/train_${desc}.log" 2>&1

    # Print summary
    echo "Results:"
    grep -E "(Overall accuracy|Overall AUC|Total test)" "/tmp/train_${desc}.log" || echo "  No aggregate results"
    echo "Finished: $(date -u)"
    echo ""
done

# ── Step 3: Summary ───────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "=== BARRIER SWEEP SUMMARY ==="
echo "=========================================="
echo ""
printf "%-25s %8s %8s %12s\n" "Config" "Acc" "AUC" "N"
printf "%-25s %8s %8s %12s\n" "-------------------------" "--------" "--------" "------------"

for config in "${CONFIGS[@]}"; do
    IFS=':' read -r tp sl desc <<< "$config"
    log="/tmp/train_${desc}.log"
    if [ -f "$log" ]; then
        acc=$(grep "Overall accuracy" "$log" | awk '{print $NF}')
        auc=$(grep "Overall AUC" "$log" | awk '{print $NF}')
        n=$(grep "Total test" "$log" | awk '{print $NF}')
        printf "%-25s %8s %8s %12s\n" "$desc" "${acc:-N/A}" "${auc:-N/A}" "${n:-N/A}"
    fi
done

echo ""
# Also print confidence-filtered results for each
echo ""
echo "=== Confidence-filtered (>0.60) ==="
printf "%-25s %8s %8s\n" "Config" "Acc@0.60" "N@0.60"
printf "%-25s %8s %8s\n" "-------------------------" "--------" "--------"

for config in "${CONFIGS[@]}"; do
    IFS=':' read -r tp sl desc <<< "$config"
    log="/tmp/train_${desc}.log"
    if [ -f "$log" ]; then
        line=$(grep "0.60" "$log" | tail -1)
        if [ -n "$line" ]; then
            acc60=$(echo "$line" | awk '{print $2}')
            n60=$(echo "$line" | awk '{print $3}')
            printf "%-25s %8s %8s\n" "$desc" "$acc60" "$n60"
        fi
    fi
done

touch /tmp/training_complete
echo ""
echo "=== Sweep complete at $(date -u) ==="
