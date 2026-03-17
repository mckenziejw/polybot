#!/bin/bash
# Full directional classifier pipeline: wait for download, preprocess, train.
#
# Runs autonomously overnight:
#   1. Poll current instance until download completes
#   2. Tear down download instance (r8g.2xlarge)
#   3. Launch compute instance (c8g.12xlarge) with same EBS volume
#   4. Run preprocessing (all 3 variants)
#   5. Run training (all 3 variants)
#   6. Pull results locally
#   7. Tear down compute instance
#
# Usage:
#   bash swarm/cloud/run_directional_pipeline.sh 2>&1 | tee pipeline.log

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

LAUNCH="bash swarm/cloud/launch.sh"
STATE_FILE="swarm/cloud/.instance-state"
KEY_FILE="$HOME/.ssh/polybot-swarm.pem"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

ssh_cmd() {
    source "$STATE_FILE"
    ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 \
        -i "$KEY_FILE" "ubuntu@$PUBLIC_IP" "$@"
}

# ── Phase 1: Wait for download to complete ───────────────────────────

log "=== Phase 1: Download remaining orderbook data ==="

# Launch download instance if not running
if [ ! -f "$STATE_FILE" ]; then
    log "No instance running. Launching r8g.2xlarge for download..."
    $LAUNCH launch-directional r8g.2xlarge
    $LAUNCH sync-directional
    $LAUNCH download-bybit 2025-05-01 2026-03-16 --resume
    sleep 30  # give downloads time to start
fi

log "Waiting for download to complete..."

while true; do
    ob_count=$(ssh_cmd "ls /data/bybit_orderbook/*.parquet 2>/dev/null | wc -l" 2>/dev/null || echo "0")
    raw_count=$(ssh_cmd "ls /data/bybit_orderbook_raw/*.parquet 2>/dev/null | wc -l" 2>/dev/null || echo "0")
    tr_count=$(ssh_cmd "ls /data/bybit_trades_200ms/*.parquet 2>/dev/null | wc -l" 2>/dev/null || echo "0")
    running=$(ssh_cmd 'pgrep -f "bybit_orderbook_download|bybit_trades_download" | wc -l' 2>/dev/null || echo "0")

    log "Progress: OB=$ob_count RAW=$raw_count TR=$tr_count (processes running: $running)"

    # Only break when no download processes AND we have close to full data
    if [ "$running" = "0" ] && [ "$ob_count" -gt 300 ]; then
        log "Download processes finished. OB=$ob_count RAW=$raw_count TR=$tr_count"
        break
    fi

    # Also break if file count hasn't changed in 3 consecutive polls (process died)
    if [ "${prev_ob_count:-0}" = "$ob_count" ] && [ "${prev2_ob_count:-0}" = "$ob_count" ] && [ "$ob_count" -gt 100 ]; then
        log "WARNING: Download appears stalled at OB=$ob_count (no progress in 15 min). Proceeding."
        break
    fi
    prev2_ob_count="${prev_ob_count:-0}"
    prev_ob_count="$ob_count"

    sleep 300  # check every 5 minutes
done

# Verify we got a reasonable number of files
if [ "$ob_count" -lt 300 ]; then
    log "WARNING: Only $ob_count orderbook files downloaded (expected ~318). Check logs."
    log "Continuing anyway..."
fi

# Check if trades also completed
if [ "$tr_count" -lt 300 ]; then
    log "Trades incomplete ($tr_count files). The download script runs trades after orderbook."
    log "Checking if trades download is still needed..."
    # The background SSH command runs both sequentially, so if orderbook process
    # is done, trades should be too (or failed). Check the log.
    ssh_cmd "tail -5 /data/trades_download.log 2>/dev/null || echo 'No trades log found'"
    ssh_cmd "tail -5 /data/orderbook_download.log 2>/dev/null || echo 'No orderbook log found'"
fi

log "Download disk usage:"
ssh_cmd "du -sh /data/bybit_orderbook /data/bybit_orderbook_raw /data/bybit_trades_200ms 2>/dev/null"

# ── Phase 2: Swap to compute instance ────────────────────────────────

log ""
log "=== Phase 2: Swapping to c8g.12xlarge for compute ==="

# Tear down download instance
log "Tearing down download instance..."
$LAUNCH teardown
sleep 10

# Launch compute instance
log "Launching c8g.12xlarge..."
$LAUNCH launch-directional c8g.12xlarge

log "Compute instance ready."

# ── Phase 3: Preprocess features ─────────────────────────────────────

log ""
log "=== Phase 3: Preprocessing features (all 3 variants) ==="

# Sync latest code (includes updated preprocess_features.py)
$LAUNCH sync-directional

# Run preprocessing — sequentially per variant to manage memory
for variant in causal raw paper; do
    log "Preprocessing variant: $variant"

    ssh_cmd "cd ~/wss_test && \
        mkdir -p data && \
        for dir in bybit_orderbook bybit_orderbook_raw bybit_trades_200ms \
                   directional_features_paper directional_features_causal directional_features_raw; do \
            mkdir -p /data/\$dir; \
            ln -sfn /data/\$dir data/\$dir; \
        done && \
        echo \"Instance: \$(uname -m), \$(nproc) vCPU, \$(free -h | awk '/Mem:/{print \$2}') RAM\" && \
        echo \"Free mem before: \$(free -h | awk '/Mem:/{print \$4}')\" && \
        echo '' && \
        ~/venv/bin/python -m pipelines.preprocess_features --variant $variant 2>&1 && \
        echo '' && \
        echo \"Free mem after: \$(free -h | awk '/Mem:/{print \$4}')\" && \
        echo \"Feature files: \$(ls /data/directional_features_${variant}/*.parquet 2>/dev/null | wc -l)\" && \
        du -sh /data/directional_features_${variant} 2>/dev/null" \
    || log "ERROR: Preprocessing $variant failed"

    log "Variant $variant done."
done

log "All preprocessing complete."
ssh_cmd "for v in paper causal raw; do echo \"\$v: \$(ls /data/directional_features_\${v}/*.parquet 2>/dev/null | wc -l) files, \$(du -sh /data/directional_features_\${v} 2>/dev/null | awk '{print \$1}')\"; done"

# ── Phase 4: Training ────────────────────────────────────────────────

log ""
log "=== Phase 4: Walk-forward training (all 3 variants) ==="

# Need xgboost installed
ssh_cmd "~/venv/bin/pip install -q xgboost scikit-learn"

for variant in causal raw paper; do
    log "Training variant: $variant"

    ssh_cmd "cd ~/wss_test && \
        mkdir -p data && \
        for dir in bybit_orderbook bybit_orderbook_raw bybit_trades_200ms \
                   directional_features_paper directional_features_causal directional_features_raw \
                   directional_models_paper directional_models_causal directional_models_raw \
                   directional_results_paper directional_results_causal directional_results_raw; do \
            mkdir -p /data/\$dir; \
            ln -sfn /data/\$dir data/\$dir; \
        done && \
        echo \"Training $variant...\" && \
        echo \"Instance: \$(uname -m), \$(nproc) vCPU, \$(free -h | awk '/Mem:/{print \$2}') RAM\" && \
        echo '' && \
        ~/venv/bin/python -m strategies.directional.train --variant $variant 2>&1 && \
        echo '' && \
        echo 'Results:' && \
        ls -la /data/directional_results_${variant}/ 2>/dev/null" \
    || log "ERROR: Training $variant failed"

    log "Variant $variant training done."
done

log "All training complete."

# ── Phase 5: Pull results ────────────────────────────────────────────

log ""
log "=== Phase 5: Pulling results ==="

source "$STATE_FILE"

for variant in paper causal raw; do
    for dir_type in results models; do
        remote_dir="/data/directional_${dir_type}_${variant}/"
        local_dir="$PROJECT_DIR/data/directional_${dir_type}_${variant}/"
        mkdir -p "$local_dir"

        log "Pulling directional_${dir_type}_${variant}..."
        rsync -avz \
            -e "ssh -o StrictHostKeyChecking=no -i $KEY_FILE" \
            "ubuntu@$PUBLIC_IP:$remote_dir" \
            "$local_dir" 2>/dev/null || log "WARNING: Failed to pull $dir_type/$variant"
    done
done

# Also pull any logs
mkdir -p "$PROJECT_DIR/data/pipeline_logs"
scp -o StrictHostKeyChecking=no -i "$KEY_FILE" \
    "ubuntu@$PUBLIC_IP:/data/*.log" \
    "$PROJECT_DIR/data/pipeline_logs/" 2>/dev/null || true

log "Results pulled."

# ── Phase 6: Tear down ───────────────────────────────────────────────

log ""
log "=== Phase 6: Tearing down compute instance ==="
$LAUNCH teardown

log ""
log "=========================================="
log "Pipeline complete!"
log "=========================================="
log ""
log "Results in:"
for variant in paper causal raw; do
    results_dir="$PROJECT_DIR/data/directional_results_${variant}"
    if [ -d "$results_dir" ]; then
        n_files=$(ls "$results_dir"/*.json "$results_dir"/*.csv 2>/dev/null | wc -l)
        log "  $variant: $n_files result files in $results_dir"
    else
        log "  $variant: NO RESULTS"
    fi
done
log ""
log "EBS volume preserved for future use."
log "To delete: bash swarm/cloud/launch.sh delete-volume"
