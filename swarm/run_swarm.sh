#!/bin/bash
# Strategy Iteration Swarm — Coordinator
# Usage: ./swarm/run_swarm.sh "seed prompt" [num_agents]
#
# Examples:
#   ./swarm/run_swarm.sh "explore orderbook microstructure signals" 4
#   ./swarm/run_swarm.sh "continue"  # iterate on previous best

set -euo pipefail

# Allow nested Claude sessions
unset CLAUDECODE 2>/dev/null || true

SEED="${1:?Usage: run_swarm.sh <seed-prompt|continue> [num_agents]}"
NUM_AGENTS="${2:-3}"
SWARM_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SWARM_DIR")"

# Determine generation number
GENERATION=$(find "$SWARM_DIR" -maxdepth 1 -name 'generation_*' -type d 2>/dev/null | wc -l)
GEN_DIR="$SWARM_DIR/generation_$GENERATION"

echo "=== Swarm Generation $GENERATION ==="
echo "Seed: $SEED"
echo "Agents: $NUM_AGENTS"
echo ""

# Step 1: Generate hypotheses
# If "continue", read leaderboard and mutate. Otherwise, generate from seed.
if [ "$SEED" = "continue" ] && [ -f "$SWARM_DIR/leaderboard.json" ]; then
    CONTINUE_MODE=true
    CONTEXT="$(cat "$SWARM_DIR/leaderboard.json")"
else
    CONTINUE_MODE=false
    CONTEXT="$SEED"
fi

echo "Generating $NUM_AGENTS strategy hypotheses..."
HYPO_PROMPT_FILE="$GEN_DIR/hypothesis_prompt.txt"
mkdir -p "$GEN_DIR"

if [ "$CONTINUE_MODE" = true ]; then
cat > "$HYPO_PROMPT_FILE" << HEOF
You are a quantitative trading researcher iterating on a PROVEN signal.

The following strategy showed positive results in backtesting:
$CONTEXT

Your job is to generate exactly $NUM_AGENTS VARIATIONS of this winning strategy. Do NOT invent entirely new strategies. Instead, mutate the winning approach:
- Try different parameter values (thresholds, window sizes, level selections)
- Try combining it with additional filters (spread, time-of-day, volatility regime)
- Try the inverse or complementary signal
- Try different entry timing (earlier/later snapshots)
- Try restricting to subsets of markets where the signal is stronger

Each variation should be a concrete, testable modification of the original winning strategy.
Output ONLY a JSON array of strings, each string being one hypothesis (2-3 sentences). No other text.

Available data: 5m orderbook snapshots (7,187 files, 5 levels deep), trade events (5,339 files), 4h orderbook (798 files), resolution cache. Raw data includes pre/post market open snapshots.
HEOF
else
cat > "$HYPO_PROMPT_FILE" << HEOF
You are a quantitative trading researcher. Generate exactly $NUM_AGENTS diverse, novel strategy hypotheses for Polymarket 5-minute BTC Up/Down binary prediction markets.

Context: $CONTEXT

IMPORTANT — these have already been tested and FAILED:
- Mid-price drift, BTC momentum, whale directional signals, XGBoost confidence, fading trade flow, sports longshot bias

Generate genuinely novel ideas. For each, output ONLY a JSON array of strings, each string being one hypothesis (2-3 sentences). No other text.

Available data: 5m orderbook snapshots (7,187 files, 5 levels deep), trade events (5,339 files), 4h orderbook (798 files), resolution cache. Raw data includes pre/post market open snapshots.
HEOF
fi
HYPO_RAW_FILE="$GEN_DIR/hypotheses_raw.txt"
HYPO_JSON_FILE="$GEN_DIR/hypotheses.json"

claude -p "$(cat "$HYPO_PROMPT_FILE")" \
    --dangerously-skip-permissions --max-turns 3 --output-format text \
    > "$HYPO_RAW_FILE" 2>/dev/null

# Extract JSON array from output (may have markdown fences, multi-line)
.venv/bin/python3 -c "
import json, re, sys
raw = open('$HYPO_RAW_FILE').read()
# Strip markdown fences
raw = re.sub(r'\`\`\`json\s*', '', raw)
raw = re.sub(r'\`\`\`\s*', '', raw)
arr = json.loads(raw.strip())
assert isinstance(arr, list) and len(arr) >= $NUM_AGENTS, f'Expected {$NUM_AGENTS} hypotheses, got {len(arr)}'
json.dump(arr, open('$HYPO_JSON_FILE', 'w'))
for i, h in enumerate(arr):
    print(f'  {i+1}. {h}')
"

if [ ! -f "$HYPO_JSON_FILE" ]; then
    echo "ERROR: Failed to generate hypotheses. Raw output:"
    cat "$HYPO_RAW_FILE"
    exit 1
fi

echo ""

# Step 2: Launch agents in parallel
PIDS=()
for i in $(seq 0 $((NUM_AGENTS - 1))); do
    AGENT_DIR="$GEN_DIR/agent_$i"
    mkdir -p "$AGENT_DIR"

    # Extract hypothesis for this agent
    HYPOTHESIS=$(.venv/bin/python3 -c "import json; print(json.load(open('$HYPO_JSON_FILE'))[$i])")

    echo "Launching agent $i: $HYPOTHESIS"

    AGENT_PROMPT="You are a quantitative trading strategy researcher. Your task:

1. Implement and backtest a specific strategy against Polymarket 5-minute BTC Up/Down binary prediction markets.

2. STRATEGY HYPOTHESIS: $HYPOTHESIS

3. DATA — USE THE SAMPLE DATASET FOR INITIAL TESTING:
   - SAMPLE book snapshots (USE THIS FIRST): data/sample_1k/book_snapshots_sample_1k.parquet
     Single file, 917 markets, first 60s of orderbook per market. Load with pd.read_parquet().
   - SAMPLE resolution cache: data/sample_1k/resolution_cache_sample.json
     Columns: exchange_timestamp, received_timestamp, slug, market, asset_id, token_label (Up/Down), mid_price, spread, book_imbalance, bid/ask_price/size_1-5
   - Full dataset (only if sample shows promise): data/telonex_book_snapshots/ (7,187 parquet files)
   - Full resolution cache: data/resolution_cache.json (maps slug -> winner Up/Down)
   - 5m trade events: data/trade_events/ (5,339 parquet files)
   - 4h orderbook: data/telonex_btc_4h_book_snapshots/ (798 files)
   - Raw book snapshots (includes pre/post market): datasets/telonex_5m_raw/

4. RULES:
   - Use $PROJECT_DIR/.venv/bin/python3 for all Python execution
   - Use ONLY vectorized pandas/numpy operations — NO iterrows(), NO row-by-row loops
   - Account for Polymarket fees: fee = 0.0222 * p * 0.25 * (p*(1-p))^2
   - Minimum order size: 5 tokens
   - slug format: btc-updown-5m-{unix_epoch_seconds}
   - token_label Up = BTC goes up in the 5m window
   - Markets resolve to 1.00 (winner) or 0.00 (loser)
   - Entry price cap: never buy above 0.85 (payoff asymmetry)

5. OUTPUT:
   - Write your strategy implementation to: $AGENT_DIR/strategy.py
   - Write standardized results JSON to: $AGENT_DIR/results.json
   - results.json schema: {\"strategy_name\": \"\", \"description\": \"\", \"hypothesis\": \"\", \"metrics\": {\"total_trades\": 0, \"win_rate\": 0.0, \"pnl_total\": 0.0, \"pnl_per_trade\": 0.0, \"max_drawdown\": 0.0, \"sharpe\": 0.0}, \"parameters\": {}, \"verdict\": \"promising|marginal|failed\", \"next_steps\": \"\"}

6. Be rigorous. If the strategy doesn't work, say so honestly. Don't cherry-pick parameters. Write results.json even if the strategy fails.

7. CRITICAL SAFETY RULES:
   - NEVER execute live trades, place orders, or submit any transaction
   - NEVER send transactions on-chain or call any smart contract
   - NEVER use the execution/ TypeScript code or any CLOB/exchange client
   - Read-only API queries (e.g. Gamma API, public market data) are ALLOWED
   - Backtesting must use historical parquet data, not live prices"

    AGENT_PROMPT_FILE="$AGENT_DIR/prompt.txt"
    echo "$AGENT_PROMPT" > "$AGENT_PROMPT_FILE"

    # Each agent gets its own .claude.json copy (avoids write races)
    # but shares everything else via the real HOME
    AGENT_CONF="$AGENT_DIR/.claude.json"
    cp "$HOME/.claude.json" "$AGENT_CONF"

    # Lower CPU priority; memory capped by Docker container (24GB total)
    nice -n 10 \
        bash -c "
            AGENT_HOME='$AGENT_DIR/.home'
            mkdir -p \"\$AGENT_HOME\"
            cp '$AGENT_CONF' \"\$AGENT_HOME/.claude.json\"
            ln -sfn '$HOME/.claude' \"\$AGENT_HOME/.claude\"
            ln -sfn '$HOME/.anthropic' \"\$AGENT_HOME/.anthropic\" 2>/dev/null
            export HOME=\"\$AGENT_HOME\"
            exec claude -p \"\$(cat '$AGENT_PROMPT_FILE')\" \
                --dangerously-skip-permissions \
                --max-turns 30 \
                --output-format text
        " > "$AGENT_DIR/agent_log.txt" 2>&1 &

    PIDS+=($!)
done

echo ""
echo "All $NUM_AGENTS agents launched. Waiting for completion..."
echo "Monitor progress: tail -f $GEN_DIR/agent_*/agent_log.txt"
echo ""

# Step 3: Wait for all agents
FAILED=0
for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
        echo "Agent $i completed successfully."
    else
        echo "Agent $i failed (exit code $?)."
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "=== Generation $GENERATION Results ==="
echo ""

# Step 4: Collect and rank results
export GEN_DIR
.venv/bin/python3 << 'PYEOF'
import json, glob, os

gen_dir = os.environ.get("GEN_DIR", "")
if not gen_dir:
    import sys
    sys.exit(1)

results = []
for f in sorted(glob.glob(f"{gen_dir}/agent_*/results.json")):
    try:
        with open(f) as fh:
            r = json.load(fh)
            r["_source"] = f
            results.append(r)
    except Exception as e:
        print(f"  WARN: Could not read {f}: {e}")

if not results:
    print("  No results collected. Check agent logs.")
    import sys
    sys.exit(0)

# Rank by PnL per trade
results.sort(key=lambda r: r.get("metrics", {}).get("pnl_per_trade", -999), reverse=True)

print(f"{'Rank':<5} {'Strategy':<30} {'Trades':<8} {'WR%':<8} {'PnL/Trade':<12} {'Total PnL':<12} {'Verdict'}")
print("-" * 95)
for i, r in enumerate(results):
    m = r.get("metrics", {})
    print(f"{i+1:<5} {r.get('strategy_name', '?'):<30} {m.get('total_trades', 0):<8} {m.get('win_rate', 0)*100:<8.1f} ${m.get('pnl_per_trade', 0):<11.4f} ${m.get('pnl_total', 0):<11.2f} {r.get('verdict', '?')}")

# Update leaderboard
swarm_dir = os.path.dirname(gen_dir)
lb_path = os.path.join(swarm_dir, "leaderboard.json")
try:
    with open(lb_path) as f:
        leaderboard = json.load(f)
except FileNotFoundError:
    leaderboard = []

# Add promising results
for r in results:
    if r.get("verdict") in ("promising", "marginal"):
        entry = {k: v for k, v in r.items() if k != "_source"}
        entry["generation"] = int(os.path.basename(gen_dir).split("_")[1])
        leaderboard.append(entry)

# Keep top 10
leaderboard.sort(key=lambda r: r.get("metrics", {}).get("pnl_per_trade", -999), reverse=True)
leaderboard = leaderboard[:10]

with open(lb_path, "w") as f:
    json.dump(leaderboard, f, indent=2)

print(f"\nLeaderboard updated: {lb_path} ({len(leaderboard)} entries)")
PYEOF

echo ""
echo "Done. Run './swarm/run_swarm.sh continue' to iterate on the best results."
