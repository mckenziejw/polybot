#!/bin/bash
# Run both v3 training variants in parallel with proper thread allocation,
# then signal completion. Designed to run on EC2 via nohup.
#
# Usage: nohup bash run_v3_training.sh &

set -euo pipefail
cd /home/ubuntu
source venv/bin/activate
export PYTHONUNBUFFERED=1

TOTAL_CORES=$(nproc)
N_RUNS=2
# Sequential seed training (no ProcessPoolExecutor fork), so only 1 model
# trains at a time per run. Each model gets half the machine.
JOBS_PER_SEED=$(( TOTAL_CORES / N_RUNS ))
JOBS_PER_SEED=$(( JOBS_PER_SEED > 0 ? JOBS_PER_SEED : 1 ))

echo "=== V3 Training Launcher ==="
echo "Cores: $TOTAL_CORES, Parallel runs: $N_RUNS"
echo "Threads per XGBoost model: $JOBS_PER_SEED"
echo "Started: $(date -u)"
echo ""

# Launch both variants in background
python3 -m strategies.directional.train \
    --variant ob \
    --resolution 1m \
    --n-jobs "$JOBS_PER_SEED" \
    > /tmp/train_ob.log 2>&1 &
PID_OB=$!

python3 -m strategies.directional.train \
    --variant ohlcv \
    --resolution 1m \
    --n-jobs "$JOBS_PER_SEED" \
    > /tmp/train_ohlcv.log 2>&1 &
PID_OHLCV=$!

echo "OB PID: $PID_OB"
echo "OHLCV PID: $PID_OHLCV"

# Wait for both to finish
wait $PID_OB
OB_EXIT=$?
echo "OB finished with exit code $OB_EXIT at $(date -u)"

wait $PID_OHLCV
OHLCV_EXIT=$?
echo "OHLCV finished with exit code $OHLCV_EXIT at $(date -u)"

# Print summary
echo ""
echo "=== OB Results ==="
grep -E "(Overall accuracy|Overall AUC|Total test)" /tmp/train_ob.log || echo "No aggregate results"
echo ""
echo "=== OHLCV Results ==="
grep -E "(Overall accuracy|Overall AUC|Total test)" /tmp/train_ohlcv.log || echo "No aggregate results"

# Signal completion
touch /tmp/training_complete
echo ""
echo "=== Training complete at $(date -u) ==="
