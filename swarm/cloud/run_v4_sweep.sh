#!/bin/bash
# V4 sweep: cross-asset features + training window length comparison.
#
# Preprocesses OHLCV + derivatives + cross-asset features once,
# then trains across 4 training window lengths (30, 45, 60, 90 days).
#
# Usage: nohup bash run_v4_sweep.sh &

set -euo pipefail
cd /home/ubuntu
source venv/bin/activate
export PYTHONUNBUFFERED=1

TOTAL_CORES=$(nproc)
echo "=== V4 Cross-Asset Sweep ==="
echo "Cores: $TOTAL_CORES"
echo "Started: $(date -u)"
echo ""

# ── Step 1: Preprocess OHLCV + cross-asset features ──────────────────
echo "=== Step 1: Preprocessing OHLCV + cross-asset features ==="
rm -rf data/directional_features_v3_ohlcv
python3 -m pipelines.preprocess_features \
    --variant ohlcv \
    --no-resume \
    2>&1 | tail -20
echo ""

# ── Step 2: Training window sweep ────────────────────────────────────
TRAIN_DAYS=(30 45 60 90)

echo "=== Step 2: Training ${#TRAIN_DAYS[@]} window configs ==="
echo ""

for days in "${TRAIN_DAYS[@]}"; do
    echo "──────────────────────────────────────────────────────────"
    echo "Training window: ${days} days"
    echo "Started: $(date -u)"
    echo "──────────────────────────────────────────────────────────"

    python3 -m strategies.directional.train \
        --variant ohlcv \
        --resolution 1m \
        --train-days "$days" \
        --n-jobs "$TOTAL_CORES" \
        --results-dir "data/directional_results_v4_${days}d" \
        --model-dir "data/directional_models_v4_${days}d" \
        > "/tmp/train_v4_${days}d.log" 2>&1

    echo "Results:"
    grep -E "(Overall accuracy|Overall AUC|Total test)" "/tmp/train_v4_${days}d.log" || echo "  No aggregate results"
    echo "Finished: $(date -u)"
    echo ""
done

# ── Step 3: Summary ──────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "=== V4 TRAINING WINDOW SWEEP SUMMARY ==="
echo "=========================================="
echo ""
printf "%-15s %8s %8s %12s %8s\n" "Window" "Acc" "AUC" "N" "Windows"
printf "%-15s %8s %8s %12s %8s\n" "---------------" "--------" "--------" "------------" "--------"

for days in "${TRAIN_DAYS[@]}"; do
    log="/tmp/train_v4_${days}d.log"
    if [ -f "$log" ]; then
        acc=$(grep "Overall accuracy" "$log" | awk '{print $NF}')
        auc=$(grep "Overall AUC" "$log" | awk '{print $NF}')
        n=$(grep "Total test" "$log" | awk '{print $NF}')
        nw=$(grep -c "^  W" "$log" || echo "?")
        printf "%-15s %8s %8s %12s %8s\n" "${days}d" "${acc:-N/A}" "${auc:-N/A}" "${n:-N/A}" "$nw"
    fi
done

# Also print first-half vs second-half AUC for each
echo ""
echo "=== Temporal stability (first vs second half AUC) ==="
for days in "${TRAIN_DAYS[@]}"; do
    log="/tmp/train_v4_${days}d.log"
    if [ -f "$log" ]; then
        # Extract per-window AUCs
        aucs=$(grep "^  W" "$log" | sed 's/.*auc=\([0-9.]*\).*/\1/')
        n_windows=$(echo "$aucs" | wc -l)
        half=$((n_windows / 2))
        if [ "$half" -gt 0 ]; then
            first_half=$(echo "$aucs" | head -"$half" | awk '{s+=$1} END {printf "%.4f", s/NR}')
            second_half=$(echo "$aucs" | tail -"$half" | awk '{s+=$1} END {printf "%.4f", s/NR}')
            printf "%-15s first_half=%s  second_half=%s\n" "${days}d" "$first_half" "$second_half"
        fi
    fi
done

touch /tmp/training_complete
echo ""
echo "=== Sweep complete at $(date -u) ==="
