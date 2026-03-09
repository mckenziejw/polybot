#!/bin/bash
# Launch multi-asset trading instances.
# Each asset gets its own model server + execution engine.
#
# Usage:
#   ./start_all.sh              # start all configured assets
#   ./start_all.sh btc eth      # start specific assets only
#
# Stop: Ctrl+C (sends SIGINT to all children)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Asset → model server port mapping
declare -A PORTS=(
  [btc]=8000
  [eth]=8001
  [sol]=8002
  [xrp]=8003
)

# Default: all assets with config files present
if [ $# -eq 0 ]; then
  ASSETS=()
  for asset in btc eth sol xrp; do
    cfg="config.json"
    [ "$asset" != "btc" ] && cfg="config.${asset}.json"
    [ -f "$cfg" ] && ASSETS+=("$asset")
  done
else
  ASSETS=("$@")
fi

PIDS=()

cleanup() {
  echo ""
  echo "[Launcher] Shutting down all processes..."
  for pid in "${PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait
  echo "[Launcher] All processes stopped."
}
trap cleanup SIGINT SIGTERM

echo "[Launcher] Starting assets: ${ASSETS[*]}"
echo ""

for asset in "${ASSETS[@]}"; do
  port="${PORTS[$asset]}"
  model_path="data/xgb_model_${asset}.json"
  meta_path="data/xgb_model_meta_${asset}.json"
  feature_cache="data/xgb_features_v3_${asset}.parquet"

  # BTC uses default paths (backward compatible)
  if [ "$asset" = "btc" ]; then
    model_path="data/xgb_model.json"
    meta_path="data/xgb_model_meta.json"
    feature_cache="data/xgb_features_v3.parquet"
  fi

  spot_quotes="data/${asset}_quotes/${asset}usdt_quotes.parquet"
  # BTC quotes are in a different directory
  [ "$asset" = "btc" ] && spot_quotes="data/btc_quotes/btcusdt_quotes.parquet"

  if [ ! -f "$model_path" ]; then
    echo "[Launcher] WARNING: No model found at $model_path — skipping $asset"
    continue
  fi

  cfg="config.json"
  [ "$asset" != "btc" ] && cfg="config.${asset}.json"

  echo "[Launcher] Starting $asset model server on port $port..."
  python3 model_server.py \
    --port "$port" \
    --model-path "$model_path" \
    --meta-path "$meta_path" \
    --spot-quotes "$spot_quotes" &
  PIDS+=($!)

  sleep 1

  echo "[Launcher] Starting $asset execution engine (config: $cfg)..."
  CONFIG="$cfg" bun run execution/src/index.ts &
  PIDS+=($!)

  echo ""
done

echo "[Launcher] All instances started. Press Ctrl+C to stop."
wait
