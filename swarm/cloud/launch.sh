#!/bin/bash
# Launch a spot instance for swarm runs, rsync data, run swarm, pull results.
#
# Usage:
#   ./swarm/cloud/launch.sh                          # launch instance
#   ./swarm/cloud/launch.sh run "seed prompt" 3      # launch + run swarm
#   ./swarm/cloud/launch.sh teardown                 # terminate instance
#   ./swarm/cloud/launch.sh ssh                      # ssh into instance
#
# Prerequisites:
#   - AWS CLI configured (aws configure)
#   - SSH key pair created in AWS (default: polybot-swarm)
#   - Private key at ~/.ssh/polybot-swarm.pem

set -euo pipefail

# === Configuration ===
INSTANCE_TYPE="r8g.2xlarge"
AMI_REGION="us-east-1"
KEY_NAME="${POLYBOT_KEY_NAME:-polybot-swarm}"
KEY_FILE="${POLYBOT_KEY_FILE:-$HOME/.ssh/polybot-swarm.pem}"
SECURITY_GROUP="${POLYBOT_SG:-polybot-swarm-sg}"
IAM_PROFILE="${POLYBOT_IAM_PROFILE:-polybot-swarm-s3}"
S3_BUCKET="${POLYBOT_S3_BUCKET:-polybot-swarm-data}"
SPOT_MAX_PRICE="0.25"  # safety cap, actual spot is ~$0.18
STATE_FILE="swarm/cloud/.instance-state"
PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"

# Ubuntu 24.04 ARM64 AMI (us-east-1) — update if using different region
# This will be looked up dynamically below
AMI_ID=""

log() { echo "[$(date +%H:%M:%S)] $*"; }

# === Functions ===

lookup_ami() {
    log "Looking up latest Ubuntu 24.04 ARM64 AMI..."
    AMI_ID=$(aws ec2 describe-images \
        --region "$AMI_REGION" \
        --owners 099720109477 \
        --filters "Name=name,Values=ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-arm64-server-*" \
                  "Name=state,Values=available" \
        --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId' \
        --output text)
    log "AMI: $AMI_ID"
}

ensure_security_group() {
    if ! aws ec2 describe-security-groups --region "$AMI_REGION" \
        --group-names "$SECURITY_GROUP" &>/dev/null; then
        log "Creating security group $SECURITY_GROUP..."
        SG_ID=$(aws ec2 create-security-group \
            --region "$AMI_REGION" \
            --group-name "$SECURITY_GROUP" \
            --description "Polybot swarm instances" \
            --query 'GroupId' --output text)
        aws ec2 authorize-security-group-ingress \
            --region "$AMI_REGION" \
            --group-id "$SG_ID" \
            --protocol tcp --port 22 --cidr "0.0.0.0/0"
        log "Security group created: $SG_ID"
    fi
}

launch_instance() {
    lookup_ami
    ensure_security_group

    log "Requesting spot instance ($INSTANCE_TYPE, max \$$SPOT_MAX_PRICE/hr)..."

    INSTANCE_ID=$(aws ec2 run-instances \
        --region "$AMI_REGION" \
        --image-id "$AMI_ID" \
        --instance-type "$INSTANCE_TYPE" \
        --key-name "$KEY_NAME" \
        --security-groups "$SECURITY_GROUP" \
        --iam-instance-profile "Name=$IAM_PROFILE" \
        --instance-market-options '{"MarketType":"spot","SpotOptions":{"MaxPrice":"'"$SPOT_MAX_PRICE"'","SpotInstanceType":"one-time"}}' \
        --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":50,"VolumeType":"gp3"}}]' \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=polybot-swarm}]" \
        --query 'Instances[0].InstanceId' \
        --output text)

    log "Instance requested: $INSTANCE_ID"
    log "Waiting for instance to be running..."

    aws ec2 wait instance-running --region "$AMI_REGION" --instance-ids "$INSTANCE_ID"

    PUBLIC_IP=$(aws ec2 describe-instances \
        --region "$AMI_REGION" \
        --instance-ids "$INSTANCE_ID" \
        --query 'Reservations[0].Instances[0].PublicIpAddress' \
        --output text)

    log "Instance running at $PUBLIC_IP"

    # Save state
    mkdir -p "$(dirname "$STATE_FILE")"
    cat > "$STATE_FILE" << EOF
INSTANCE_ID=$INSTANCE_ID
PUBLIC_IP=$PUBLIC_IP
REGION=$AMI_REGION
EOF

    log "Waiting for SSH to be ready..."
    for i in $(seq 1 30); do
        if ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 \
            -i "$KEY_FILE" "ubuntu@$PUBLIC_IP" "echo ready" &>/dev/null; then
            break
        fi
        sleep 5
    done

    log "SSH ready. Setting up instance..."
    setup_instance
}

setup_instance() {
    load_state
    SSH="ssh -o StrictHostKeyChecking=no -i $KEY_FILE ubuntu@$PUBLIC_IP"
    SCP="scp -o StrictHostKeyChecking=no -i $KEY_FILE"

    # Install dependencies
    $SSH << 'SETUP'
set -e
sudo apt-get update -qq
sudo apt-get install -y -qq curl git jq python3 python3-pip python3-venv unzip

# Bun
curl -fsSL https://bun.sh/install | bash
# Persist bun in PATH for non-interactive shells
echo 'export PATH="$HOME/.bun/bin:$PATH"' >> ~/.profile

# Node (for Claude Code)
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo bash -
sudo apt-get install -y -qq nodejs

# AWS CLI v2 (ARM64 instance)
curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-aarch64.zip" -o /tmp/awscliv2.zip
unzip -q /tmp/awscliv2.zip -d /tmp/
sudo /tmp/aws/install
rm -rf /tmp/aws /tmp/awscliv2.zip

# Claude Code
sudo npm install -g @anthropic-ai/claude-code

echo "Setup complete."
SETUP

    log "Instance setup complete."
}

stage_s3() {
    log "Staging data to s3://$S3_BUCKET/ ..."
    log "Uploading curated datasets (~27GB total)."

    # Create bucket if needed
    if ! aws s3api head-bucket --bucket "$S3_BUCKET" 2>/dev/null; then
        log "Creating S3 bucket $S3_BUCKET..."
        aws s3api create-bucket --bucket "$S3_BUCKET" --region "$AMI_REGION" \
            --create-bucket-configuration LocationConstraint="$AMI_REGION" 2>/dev/null || \
        aws s3api create-bucket --bucket "$S3_BUCKET" --region "$AMI_REGION"
    fi

    # Normalized 5m book snapshots (all assets) + 4h + resolution cache
    PIDS=()
    for ASSET in btc eth sol xrp; do
        ASSET_DIR="$PROJECT_DIR/data/telonex_${ASSET}_5m_book_snapshots"
        if [ -d "$ASSET_DIR" ]; then
            log "Syncing ${ASSET^^} 5m book snapshots..."
            aws s3 sync "$ASSET_DIR/" \
                "s3://$S3_BUCKET/telonex_${ASSET}_5m_book_snapshots/" --quiet &
            PIDS+=($!)
        fi
    done

    log "Syncing BTC 4h book snapshots..."
    aws s3 sync "$PROJECT_DIR/data/telonex_btc_4h_book_snapshots/" \
        "s3://$S3_BUCKET/telonex_btc_4h_book_snapshots/" --quiet &
    PIDS+=($!)

    aws s3 cp "$PROJECT_DIR/data/resolution_cache.json" \
        "s3://$S3_BUCKET/resolution_cache.json" --quiet &
    PIDS+=($!)

    for pid in "${PIDS[@]}"; do
        wait "$pid"
    done

    log "All datasets staged."
    log "Run 'aws s3 ls s3://$S3_BUCKET/data/ --summarize' to check."
}

sync_data() {
    load_state
    SSH="ssh -o StrictHostKeyChecking=no -i $KEY_FILE ubuntu@$PUBLIC_IP"

    log "Syncing code to instance via rsync..."

    # Sync code only (lightweight) — data comes from S3
    rsync -avz --progress \
        -e "ssh -o StrictHostKeyChecking=no -i $KEY_FILE" \
        --exclude '.git' \
        --exclude 'node_modules' \
        --exclude '.venv' \
        --exclude 'execution/node_modules' \
        --exclude 'datasets/' \
        --exclude 'data/' \
        --exclude 'swarm/generation_*/agent_*/.home' \
        --exclude '*.ipynb' \
        --exclude 'nohup.out' \
        "$PROJECT_DIR/" "ubuntu@$PUBLIC_IP:~/wss_test/"

    log "Pulling data from S3 on instance..."

    $SSH << S3PULL
set -e
cd ~/wss_test
mkdir -p data

# Pull normalized 5m book snapshots (all assets) + 4h + resolution cache
for ASSET in btc eth sol xrp; do
    echo "Pulling \${ASSET^^} 5m book snapshots from S3..."
    aws s3 sync "s3://$S3_BUCKET/telonex_\${ASSET}_5m_book_snapshots/" "data/telonex_\${ASSET}_5m_book_snapshots/" --quiet &
done

echo "Pulling BTC 4h orderbook from S3..."
aws s3 sync "s3://$S3_BUCKET/telonex_btc_4h_book_snapshots/" data/telonex_btc_4h_book_snapshots/ --quiet &

echo "Pulling resolution cache..."
aws s3 cp "s3://$S3_BUCKET/resolution_cache.json" data/resolution_cache.json --quiet &

wait
echo "All data synced from S3."

# Build full sample dataset (no truncation, we have the memory)
python3 -m venv .venv 2>/dev/null || true
.venv/bin/pip install -q pandas numpy pyarrow scikit-learn xgboost 2>/dev/null
echo "Building full sample dataset..."
.venv/bin/python3 swarm/cloud/build_sample.py --truncate-sec 0
echo "Sample dataset ready."
S3PULL

    log "Data sync complete."
}

sync_auth() {
    load_state
    log "Syncing Claude auth to instance..."

    SCP="scp -o StrictHostKeyChecking=no -i $KEY_FILE"
    SSH="ssh -o StrictHostKeyChecking=no -i $KEY_FILE ubuntu@$PUBLIC_IP"

    # Copy auth files
    $SCP "$HOME/.claude.json" "ubuntu@$PUBLIC_IP:~/.claude.json"
    $SSH "mkdir -p ~/.claude"

    # Copy just the essential claude config (not conversation history)
    for f in settings.json; do
        if [ -f "$HOME/.claude/$f" ]; then
            $SCP "$HOME/.claude/$f" "ubuntu@$PUBLIC_IP:~/.claude/$f"
        fi
    done

    log "Auth synced."
}

run_swarm_remote() {
    load_state
    local seed="${1:-explore novel orderbook signals}"
    local agents="${2:-3}"

    SSH="ssh -o StrictHostKeyChecking=no -i $KEY_FILE ubuntu@$PUBLIC_IP"

    log "Running swarm on remote instance: seed='$seed', agents=$agents"

    $SSH << REMOTE
cd ~/wss_test
export PATH="\$HOME/.bun/bin:\$PATH"
unset CLAUDECODE
bash swarm/run_swarm.sh "$seed" $agents
REMOTE

    log "Swarm complete. Pulling results..."
    pull_results
}

pull_results() {
    load_state
    log "Pulling swarm results from instance..."

    rsync -avz \
        -e "ssh -o StrictHostKeyChecking=no -i $KEY_FILE" \
        --include='*/' \
        --include='results.json' \
        --include='strategy.py' \
        --include='trades.csv' \
        --include='leaderboard.json' \
        --include='hypotheses.json' \
        --include='agent_log.txt' \
        --exclude='*' \
        "ubuntu@$PUBLIC_IP:~/wss_test/swarm/" "$PROJECT_DIR/swarm/"

    log "Results pulled."
}

teardown() {
    load_state
    log "Terminating instance $INSTANCE_ID..."
    aws ec2 terminate-instances --region "$REGION" --instance-ids "$INSTANCE_ID" >/dev/null
    rm -f "$STATE_FILE"
    log "Instance terminated."
}

do_ssh() {
    load_state
    exec ssh -o StrictHostKeyChecking=no -i "$KEY_FILE" "ubuntu@$PUBLIC_IP"
}

load_state() {
    if [ ! -f "$STATE_FILE" ]; then
        echo "ERROR: No instance state found. Run 'launch.sh' first."
        exit 1
    fi
    source "$STATE_FILE"
}

show_status() {
    if [ -f "$STATE_FILE" ]; then
        source "$STATE_FILE"
        echo "Instance: $INSTANCE_ID"
        echo "IP: $PUBLIC_IP"
        echo "Region: $REGION"
        STATE=$(aws ec2 describe-instances --region "$REGION" \
            --instance-ids "$INSTANCE_ID" \
            --query 'Reservations[0].Instances[0].State.Name' \
            --output text 2>/dev/null || echo "unknown")
        echo "State: $STATE"
        if [ "$STATE" = "running" ]; then
            SPOT_PRICE=$(aws ec2 describe-spot-price-history \
                --region "$REGION" \
                --instance-types "$INSTANCE_TYPE" \
                --product-descriptions "Linux/UNIX" \
                --max-items 1 \
                --query 'SpotPriceHistory[0].SpotPrice' \
                --output text 2>/dev/null || echo "?")
            echo "Current spot price: \$$SPOT_PRICE/hr"
        fi
    else
        echo "No active instance."
    fi
}

# === Main ===

CMD="${1:-launch}"
shift || true

case "$CMD" in
    stage)
        stage_s3
        ;;
    launch)
        launch_instance
        sync_auth
        sync_data
        log "Instance ready. Use './swarm/cloud/launch.sh run \"prompt\" 3' to start a swarm."
        ;;
    run)
        SEED="${1:-explore novel orderbook signals}"
        AGENTS="${2:-3}"
        run_swarm_remote "$SEED" "$AGENTS"
        ;;
    sync)
        sync_data
        sync_auth
        ;;
    pull)
        pull_results
        ;;
    ssh)
        do_ssh
        ;;
    status)
        show_status
        ;;
    teardown)
        teardown
        ;;
    *)
        echo "Usage: $0 {stage|launch|run|sync|pull|ssh|status|teardown}"
        echo ""
        echo "  stage               Upload data to S3 (one-time, from local machine)"
        echo "  launch              Spin up spot instance and set up environment"
        echo "  run \"prompt\" N      Run swarm on remote instance"
        echo "  sync                Re-sync code and pull data from S3"
        echo "  pull                Pull latest results from instance"
        echo "  ssh                 SSH into instance"
        echo "  status              Show instance status and spot price"
        echo "  teardown            Terminate instance"
        exit 1
        ;;
esac
