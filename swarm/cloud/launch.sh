#!/bin/bash
# Launch a spot instance for swarm runs or GA evolution.
#
# Usage:
#   ./swarm/cloud/launch.sh launch              # launch instance (swarm mode)
#   ./swarm/cloud/launch.sh launch-evolve       # launch instance (evolution mode)
#   ./swarm/cloud/launch.sh evolve 50 20        # run GA evolution (50 pop, 20 gen)
#   ./swarm/cloud/launch.sh run "prompt" 3      # run swarm
#   ./swarm/cloud/launch.sh teardown            # terminate instance
#   ./swarm/cloud/launch.sh ssh                 # ssh into instance
#
# For larger instances:
#   POLYBOT_INSTANCE_TYPE=r8g.4xlarge ./swarm/cloud/launch.sh launch-evolve
#
# Prerequisites:
#   - AWS CLI configured (aws configure)
#   - SSH key pair created in AWS (default: polybot-swarm)
#   - Private key at ~/.ssh/polybot-swarm.pem

set -euo pipefail

# === Configuration ===
INSTANCE_TYPE="${POLYBOT_INSTANCE_TYPE:-r8g.2xlarge}"
AMI_REGION="us-east-1"
KEY_NAME="${POLYBOT_KEY_NAME:-polybot-swarm}"
KEY_FILE="${POLYBOT_KEY_FILE:-$HOME/.ssh/polybot-swarm.pem}"
SECURITY_GROUP="${POLYBOT_SG:-polybot-swarm-sg}"
IAM_PROFILE="${POLYBOT_IAM_PROFILE:-polybot-swarm-s3}"
S3_BUCKET="${POLYBOT_S3_BUCKET:-polybot-swarm-data}"
SPOT_MAX_PRICE="${POLYBOT_SPOT_MAX:-0.75}"  # safety cap for spot mode
USE_SPOT="${POLYBOT_USE_SPOT:-false}"       # default to on-demand (spot gets reclaimed)
STATE_FILE="swarm/cloud/.instance-state"
VOLUME_STATE_FILE="swarm/cloud/.volume-state"
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
    # Use custom AMI if set (pre-baked with deps), otherwise look up Ubuntu base
    if [ -n "${POLYBOT_AMI:-}" ]; then
        AMI_ID="$POLYBOT_AMI"
        log "Using custom AMI: $AMI_ID"
    else
        lookup_ami
    fi
    ensure_security_group

    SPOT_ARGS=""
    if [ "$USE_SPOT" = "true" ]; then
        log "Requesting SPOT instance ($INSTANCE_TYPE, max \$$SPOT_MAX_PRICE/hr)..."
        SPOT_ARGS="--instance-market-options {\"MarketType\":\"spot\",\"SpotOptions\":{\"MaxPrice\":\"${SPOT_MAX_PRICE}\",\"SpotInstanceType\":\"one-time\"}}"
    else
        log "Requesting ON-DEMAND instance ($INSTANCE_TYPE)..."
    fi

    # Build run-instances command
    RUN_CMD=(aws ec2 run-instances
        --region "$AMI_REGION"
        --image-id "$AMI_ID"
        --instance-type "$INSTANCE_TYPE"
        --key-name "$KEY_NAME"
        --security-groups "$SECURITY_GROUP"
        --iam-instance-profile "Name=$IAM_PROFILE"
        --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":50,"VolumeType":"gp3"}}]'
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=polybot-swarm}]"
        --query 'Instances[0].InstanceId'
        --output text
    )
    if [ "$USE_SPOT" = "true" ]; then
        RUN_CMD+=(--instance-market-options '{"MarketType":"spot","SpotOptions":{"MaxPrice":"'"$SPOT_MAX_PRICE"'","SpotInstanceType":"one-time"}}')
    fi

    INSTANCE_ID=$("${RUN_CMD[@]}")

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

# AWS CLI v2 (auto-detect architecture)
ARCH=$(uname -m)
if [ "$ARCH" = "aarch64" ]; then
    AWS_CLI_URL="https://awscli.amazonaws.com/awscli-exe-linux-aarch64.zip"
else
    AWS_CLI_URL="https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip"
fi
curl -fsSL "$AWS_CLI_URL" -o /tmp/awscliv2.zip
unzip -q /tmp/awscliv2.zip -d /tmp/
sudo /tmp/aws/install
rm -rf /tmp/aws /tmp/awscliv2.zip

# Claude Code
sudo npm install -g @anthropic-ai/claude-code

echo "Setup complete."
SETUP

    log "Instance setup complete."
}

setup_evolution_instance() {
    # Lightweight setup — only what's needed for evolution (no Bun, Node, Claude Code)
    load_state
    SSH="ssh -o StrictHostKeyChecking=no -i $KEY_FILE ubuntu@$PUBLIC_IP"

    $SSH << 'SETUP'
set -e
sudo apt-get update -qq
sudo apt-get install -y -qq python3 python3-pip python3-venv unzip

# AWS CLI v2 (auto-detect architecture)
if ! command -v aws &>/dev/null; then
    ARCH=$(uname -m)
    if [ "$ARCH" = "aarch64" ]; then
        AWS_CLI_URL="https://awscli.amazonaws.com/awscli-exe-linux-aarch64.zip"
    else
        AWS_CLI_URL="https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip"
    fi
    curl -fsSL "$AWS_CLI_URL" -o /tmp/awscliv2.zip
    unzip -q /tmp/awscliv2.zip -d /tmp/
    sudo /tmp/aws/install
    rm -rf /tmp/aws /tmp/awscliv2.zip
fi

echo "Evolution base setup complete."
SETUP

    log "Evolution instance setup complete."
}

# === Persistent EBS Volume ===

create_volume() {
    local size="${1:-200}"
    log "Creating ${size}GB gp3 EBS volume in ${AMI_REGION}..."

    # Pick AZ — use us-east-1a by default (must match instance AZ)
    local az="${AMI_REGION}a"

    VOLUME_ID=$(aws ec2 create-volume \
        --region "$AMI_REGION" \
        --availability-zone "$az" \
        --size "$size" \
        --volume-type gp3 \
        --tag-specifications "ResourceType=volume,Tags=[{Key=Name,Value=polybot-directional-data}]" \
        --query 'VolumeId' \
        --output text)

    log "Volume created: $VOLUME_ID (${size}GB gp3 in $az)"

    mkdir -p "$(dirname "$VOLUME_STATE_FILE")"
    cat > "$VOLUME_STATE_FILE" << EOF
VOLUME_ID=$VOLUME_ID
VOLUME_AZ=$az
EOF

    log "Waiting for volume to be available..."
    aws ec2 wait volume-available --region "$AMI_REGION" --volume-ids "$VOLUME_ID"
    log "Volume ready. Use 'launch-directional' to spin up an instance with it attached."
}

load_volume_state() {
    if [ ! -f "$VOLUME_STATE_FILE" ]; then
        echo "ERROR: No volume state found. Run 'create-volume' first."
        exit 1
    fi
    source "$VOLUME_STATE_FILE"
}

attach_and_mount_volume() {
    load_state
    load_volume_state
    SSH="ssh -o StrictHostKeyChecking=no -i $KEY_FILE ubuntu@$PUBLIC_IP"

    log "Attaching volume $VOLUME_ID to instance $INSTANCE_ID..."
    aws ec2 attach-volume \
        --region "$AMI_REGION" \
        --volume-id "$VOLUME_ID" \
        --instance-id "$INSTANCE_ID" \
        --device /dev/xvdf \
        --output text > /dev/null

    log "Waiting for volume to attach..."
    aws ec2 wait volume-in-use --region "$AMI_REGION" --volume-ids "$VOLUME_ID"
    sleep 5  # give kernel time to see the device

    # Mount (format on first use)
    $SSH << 'MOUNT'
set -e
DEVICE=""
for d in /dev/xvdf /dev/nvme1n1; do
    if [ -b "$d" ]; then
        DEVICE="$d"
        break
    fi
done

if [ -z "$DEVICE" ]; then
    echo "ERROR: No attached device found"
    exit 1
fi

# Check if already formatted
if ! sudo blkid "$DEVICE" | grep -q ext4; then
    echo "First use — formatting $DEVICE as ext4..."
    sudo mkfs.ext4 -q "$DEVICE"
fi

sudo mkdir -p /data
sudo mount "$DEVICE" /data
sudo chown ubuntu:ubuntu /data
echo "Volume mounted at /data ($(df -h /data | tail -1 | awk '{print $2}') total)"
MOUNT

    log "Volume mounted at /data on instance."
}

detach_volume() {
    load_volume_state
    log "Detaching volume $VOLUME_ID..."
    aws ec2 detach-volume \
        --region "$AMI_REGION" \
        --volume-id "$VOLUME_ID" \
        --output text > /dev/null 2>&1 || true
    log "Volume detached (persists for next use)."
}

delete_volume() {
    load_volume_state
    log "WARNING: This will permanently delete volume $VOLUME_ID and all data on it."
    read -p "Type 'yes' to confirm: " confirm
    if [ "$confirm" != "yes" ]; then
        log "Aborted."
        return
    fi

    # Ensure detached first
    aws ec2 detach-volume \
        --region "$AMI_REGION" \
        --volume-id "$VOLUME_ID" \
        --output text > /dev/null 2>&1 || true
    aws ec2 wait volume-available --region "$AMI_REGION" --volume-ids "$VOLUME_ID" 2>/dev/null || true

    aws ec2 delete-volume --region "$AMI_REGION" --volume-id "$VOLUME_ID"
    rm -f "$VOLUME_STATE_FILE"
    log "Volume deleted."
}

volume_status() {
    if [ ! -f "$VOLUME_STATE_FILE" ]; then
        echo "No persistent volume configured."
        return
    fi
    load_volume_state
    STATE=$(aws ec2 describe-volumes --region "$AMI_REGION" \
        --volume-ids "$VOLUME_ID" \
        --query 'Volumes[0].[State,Size,VolumeType,AvailabilityZone]' \
        --output text 2>/dev/null || echo "not-found")
    echo "Volume: $VOLUME_ID"
    echo "Status: $STATE"
}

launch_directional() {
    local instance_type="${1:-${POLYBOT_INSTANCE_TYPE:-r8g.2xlarge}}"
    load_volume_state

    # Force instance into same AZ as volume
    INSTANCE_TYPE="$instance_type"
    log "Launching $INSTANCE_TYPE in $VOLUME_AZ (same AZ as data volume)..."

    if [ -n "${POLYBOT_AMI:-}" ]; then
        AMI_ID="$POLYBOT_AMI"
        log "Using custom AMI: $AMI_ID"
    else
        lookup_ami
    fi
    ensure_security_group

    # Launch with placement in volume's AZ
    RUN_CMD=(aws ec2 run-instances
        --region "$AMI_REGION"
        --image-id "$AMI_ID"
        --instance-type "$INSTANCE_TYPE"
        --key-name "$KEY_NAME"
        --security-groups "$SECURITY_GROUP"
        --iam-instance-profile "Name=$IAM_PROFILE"
        --placement "AvailabilityZone=$VOLUME_AZ"
        --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":30,"VolumeType":"gp3"}}]'
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=polybot-directional}]"
        --query 'Instances[0].InstanceId'
        --output text
    )
    if [ "$USE_SPOT" = "true" ]; then
        RUN_CMD+=(--instance-market-options '{"MarketType":"spot","SpotOptions":{"MaxPrice":"'"$SPOT_MAX_PRICE"'","SpotInstanceType":"one-time"}}')
    fi

    INSTANCE_ID=$("${RUN_CMD[@]}")
    log "Instance requested: $INSTANCE_ID"
    log "Waiting for instance to be running..."

    aws ec2 wait instance-running --region "$AMI_REGION" --instance-ids "$INSTANCE_ID"

    PUBLIC_IP=$(aws ec2 describe-instances \
        --region "$AMI_REGION" \
        --instance-ids "$INSTANCE_ID" \
        --query 'Reservations[0].Instances[0].PublicIpAddress' \
        --output text)

    log "Instance running at $PUBLIC_IP"

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

    # Lightweight setup (just Python + AWS CLI, no Bun/Node/Claude)
    setup_directional_instance

    # Attach and mount data volume
    attach_and_mount_volume

    # Sync pipeline code
    sync_directional

    log "Instance ready. Data volume mounted at /data."
    log "Run: ./swarm/cloud/launch.sh download-bybit"
}

setup_directional_instance() {
    load_state
    SSH="ssh -o StrictHostKeyChecking=no -i $KEY_FILE ubuntu@$PUBLIC_IP"

    $SSH << 'SETUP'
set -e
sudo apt-get update -qq
sudo apt-get install -y -qq python3 python3-pip python3-venv unzip

# AWS CLI v2
if ! command -v aws &>/dev/null; then
    ARCH=$(uname -m)
    if [ "$ARCH" = "aarch64" ]; then
        AWS_CLI_URL="https://awscli.amazonaws.com/awscli-exe-linux-aarch64.zip"
    else
        AWS_CLI_URL="https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip"
    fi
    curl -fsSL "$AWS_CLI_URL" -o /tmp/awscliv2.zip
    unzip -q /tmp/awscliv2.zip -d /tmp/
    sudo /tmp/aws/install
    rm -rf /tmp/aws /tmp/awscliv2.zip
fi

# Python venv with dependencies
python3 -m venv ~/venv
~/venv/bin/pip install -q pandas pyarrow requests scipy numpy

echo "Directional instance setup complete."
SETUP

    log "Directional instance setup complete."
}

sync_directional() {
    load_state
    SSH="ssh -o StrictHostKeyChecking=no -i $KEY_FILE ubuntu@$PUBLIC_IP"

    log "Syncing pipeline code to instance..."
    $SSH "mkdir -p ~/wss_test/pipelines ~/wss_test/strategies/directional"

    scp -o StrictHostKeyChecking=no -i "$KEY_FILE" \
        "$PROJECT_DIR/pipelines/__init__.py" \
        "$PROJECT_DIR/pipelines/bybit_orderbook_download.py" \
        "$PROJECT_DIR/pipelines/bybit_trades_download.py" \
        "$PROJECT_DIR/pipelines/bybit_download_to_s3.py" \
        "$PROJECT_DIR/pipelines/preprocess_features.py" \
        "ubuntu@$PUBLIC_IP:~/wss_test/pipelines/" 2>/dev/null

    # Strategy code needed for feature preprocessing
    scp -o StrictHostKeyChecking=no -i "$KEY_FILE" \
        "$PROJECT_DIR/strategies/__init__.py" \
        "ubuntu@$PUBLIC_IP:~/wss_test/strategies/" 2>/dev/null

    scp -o StrictHostKeyChecking=no -i "$KEY_FILE" \
        "$PROJECT_DIR/strategies/directional/__init__.py" \
        "$PROJECT_DIR/strategies/directional/config.py" \
        "$PROJECT_DIR/strategies/directional/features.py" \
        "$PROJECT_DIR/strategies/directional/labels.py" \
        "$PROJECT_DIR/strategies/directional/train.py" \
        "$PROJECT_DIR/strategies/directional/backtest.py" \
        "ubuntu@$PUBLIC_IP:~/wss_test/strategies/directional/" 2>/dev/null

    log "Code synced."
}

run_download_bybit() {
    load_state
    SSH="ssh -o StrictHostKeyChecking=no -i $KEY_FILE ubuntu@$PUBLIC_IP"

    local start="${1:-2025-05-01}"
    local end="${2:-2026-03-16}"
    local extra_args="${3:---resume}"

    log "Starting Bybit download on instance ($start -> $end)..."
    log "Output goes to /data on the persistent EBS volume."

    # Set up symlinks
    $SSH "cd ~/wss_test && \
        mkdir -p data /data/bybit_orderbook /data/bybit_orderbook_raw /data/bybit_trades_200ms && \
        ln -sfn /data/bybit_orderbook data/bybit_orderbook && \
        ln -sfn /data/bybit_orderbook_raw data/bybit_orderbook_raw && \
        ln -sfn /data/bybit_trades_200ms data/bybit_trades_200ms && \
        echo \"Instance: \$(uname -m), \$(nproc) vCPU, \$(free -h | awk '/Mem:/{print \$2}') RAM\" && \
        echo \"Volume: \$(df -h /data | tail -1 | awk '{print \$4}') free\""

    # Launch downloads with nohup so they survive SSH disconnection
    log "Starting orderbook download (nohup)..."
    $SSH "cd ~/wss_test && nohup ~/venv/bin/python -u -m pipelines.bybit_orderbook_download $start $end \
        --out-dir data/bybit_orderbook \
        --raw-dir data/bybit_orderbook_raw \
        $extra_args \
        > /data/orderbook_download.log 2>&1 &"

    log "Starting trades download (nohup)..."
    $SSH "cd ~/wss_test && nohup ~/venv/bin/python -u -m pipelines.bybit_trades_download $start $end \
        --out-dir data/bybit_trades_200ms \
        --resume \
        > /data/trades_download.log 2>&1 &"

    log "Downloads launched in background on instance. Monitor with:"
    log "  ssh -i $KEY_FILE ubuntu@\$PUBLIC_IP 'ls /data/bybit_orderbook/*.parquet | wc -l'"
    log "  ssh -i $KEY_FILE ubuntu@\$PUBLIC_IP 'tail -5 /data/orderbook_download.log'"
}

run_preprocess_remote() {
    load_state
    SSH="ssh -o StrictHostKeyChecking=no -i $KEY_FILE ubuntu@$PUBLIC_IP"

    local extra_args="--resume"
    local start="${1:-}"
    local end="${2:-}"
    if [ -n "$start" ]; then
        extra_args="$extra_args --start $start"
    fi
    if [ -n "$end" ]; then
        extra_args="$extra_args --end $end"
    fi

    log "Running v2 feature preprocessing on instance..."

    $SSH << PREPROCESS
set -e
cd ~/wss_test

# Symlink data dirs to persistent volume
mkdir -p data
for dir in bybit_orderbook_raw bybit_trades_200ms directional_features_v2; do
    mkdir -p /data/\$dir
    ln -sfn /data/\$dir data/\$dir
done

echo "Instance: \$(uname -m), \$(nproc) vCPU, \$(free -h | awk '/Mem:/{print \$2}') RAM"
echo "Orderbook files: \$(ls /data/bybit_orderbook_raw/*.parquet 2>/dev/null | wc -l)"
echo "Trade files: \$(ls /data/bybit_trades_200ms/*.parquet 2>/dev/null | wc -l)"
echo ""

nohup ~/venv/bin/python -u -m pipelines.preprocess_features \
    --ob-dir data/bybit_orderbook_raw \
    --trades-dir data/bybit_trades_200ms \
    --output-dir data/directional_features_v2 \
    $extra_args \
    > /data/preprocess_v2.log 2>&1 &
PID=\$!
echo "Preprocessing launched in background (PID \$PID)."
echo "Monitor: tail -f /data/preprocess_v2.log"

# Wait a few seconds to catch early errors
sleep 5
if ! kill -0 \$PID 2>/dev/null; then
    echo "ERROR: Process exited early. Last 20 lines:"
    tail -20 /data/preprocess_v2.log
    exit 1
fi
echo "Process running OK. Disconnect safely with ctrl-c."
PREPROCESS

    log "Preprocessing launched on instance. Monitor with:"
    log "  ./swarm/cloud/launch.sh ssh"
    log "  tail -f /data/preprocess_v2.log"
}

ensure_s3_bucket() {
    if ! aws s3api head-bucket --bucket "$S3_BUCKET" 2>/dev/null; then
        log "Creating S3 bucket $S3_BUCKET..."
        aws s3api create-bucket --bucket "$S3_BUCKET" --region "$AMI_REGION" \
            --create-bucket-configuration LocationConstraint="$AMI_REGION" 2>/dev/null || \
        aws s3api create-bucket --bucket "$S3_BUCKET" --region "$AMI_REGION"
    fi
}

stage_s3() {
    log "Staging data to s3://$S3_BUCKET/ ..."
    log "Uploading curated datasets (~27GB total)."

    ensure_s3_bucket

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
    log "Run 'aws s3 ls s3://$S3_BUCKET/ --summarize' to check."
}

stage_bybit() {
    log "Staging Bybit orderbook + trades data to s3://$S3_BUCKET/ ..."
    ensure_s3_bucket

    PIDS=()

    # Smoothed orderbook (~41GB)
    if [ -d "$PROJECT_DIR/data/bybit_orderbook" ]; then
        log "Syncing bybit_orderbook (smoothed)..."
        aws s3 sync "$PROJECT_DIR/data/bybit_orderbook/" \
            "s3://$S3_BUCKET/bybit_orderbook/" --quiet &
        PIDS+=($!)
    fi

    # Raw orderbook (~12GB)
    if [ -d "$PROJECT_DIR/data/bybit_orderbook_raw" ]; then
        log "Syncing bybit_orderbook_raw..."
        aws s3 sync "$PROJECT_DIR/data/bybit_orderbook_raw/" \
            "s3://$S3_BUCKET/bybit_orderbook_raw/" --quiet &
        PIDS+=($!)
    fi

    # 200ms-binned trades (~950MB)
    if [ -d "$PROJECT_DIR/data/bybit_trades_200ms" ]; then
        log "Syncing bybit_trades_200ms..."
        aws s3 sync "$PROJECT_DIR/data/bybit_trades_200ms/" \
            "s3://$S3_BUCKET/bybit_trades_200ms/" --quiet &
        PIDS+=($!)
    fi

    # Pre-computed features (if they exist)
    for VARIANT in paper causal raw; do
        FEAT_DIR="$PROJECT_DIR/data/directional_features_${VARIANT}"
        if [ -d "$FEAT_DIR" ]; then
            log "Syncing directional_features_${VARIANT}..."
            aws s3 sync "$FEAT_DIR/" \
                "s3://$S3_BUCKET/directional_features_${VARIANT}/" --quiet &
            PIDS+=($!)
        fi
    done

    for pid in "${PIDS[@]}"; do
        wait "$pid"
    done

    # Show sizes
    log "Upload complete. Sizes:"
    for DIR in bybit_orderbook bybit_orderbook_raw bybit_trades_200ms \
               directional_features_paper directional_features_causal directional_features_raw; do
        SIZE=$(aws s3 ls "s3://$S3_BUCKET/${DIR}/" --summarize 2>/dev/null | grep "Total Size" | awk '{print $3}')
        if [ -n "$SIZE" ] && [ "$SIZE" != "0" ]; then
            SIZE_GB=$(echo "scale=1; $SIZE / 1073741824" | bc 2>/dev/null || echo "?")
            log "  $DIR: ${SIZE_GB} GB"
        fi
    done
}

stage_quotes() {
    log "Staging Binance quote parquets to s3://$S3_BUCKET/ ..."
    ensure_s3_bucket

    PIDS=()
    for ASSET in btc eth sol xrp; do
        QUOTES_DIR="$PROJECT_DIR/data/${ASSET}_quotes"
        if [ -d "$QUOTES_DIR" ]; then
            log "Syncing ${ASSET^^} quotes..."
            aws s3 sync "$QUOTES_DIR/" \
                "s3://$S3_BUCKET/${ASSET}_quotes/" --quiet &
            PIDS+=($!)
        fi
    done

    for pid in "${PIDS[@]}"; do
        wait "$pid"
    done

    log "Quote data staged. Checking sizes:"
    for ASSET in btc eth sol xrp; do
        SIZE=$(aws s3 ls "s3://$S3_BUCKET/${ASSET}_quotes/" --summarize 2>/dev/null | grep "Total Size" | awk '{print $3}')
        log "  ${ASSET^^}: ${SIZE:-0} bytes"
    done
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

setup_evolution() {
    load_state
    SSH="ssh -o StrictHostKeyChecking=no -i $KEY_FILE ubuntu@$PUBLIC_IP"

    log "Installing evolution dependencies on instance..."
    $SSH << 'EVOLVE_SETUP'
set -e
cd ~/wss_test

# Create venv if needed
python3 -m venv .venv 2>/dev/null || true

# Core GA deps + NautilusTrader
.venv/bin/pip install -q \
    nautilus_trader==1.224.0 \
    pyarrow pandas numpy scikit-learn \
    2>&1 | tail -5

echo "NautilusTrader version:"
.venv/bin/python -c "import nautilus_trader; print(nautilus_trader.__version__)"
echo "Evolution deps installed."
EVOLVE_SETUP

    log "Evolution setup complete."
}

sync_evolution() {
    load_state
    SSH="ssh -o StrictHostKeyChecking=no -i $KEY_FILE ubuntu@$PUBLIC_IP"

    log "Syncing evolution code to instance..."

    # Step 1: Sync only the evolution/ directory (the main source code we need)
    # This avoids the rsync include/exclude ordering bug that drops evolution subdirs
    rsync -az \
        -e "ssh -o StrictHostKeyChecking=no -i $KEY_FILE" \
        --exclude '__pycache__/' \
        --exclude 'runs/' \
        --exclude 'test_runs/' \
        "$PROJECT_DIR/evolution/" "ubuntu@$PUBLIC_IP:~/wss_test/evolution/"

    # Step 2: Sync only the minimal top-level files needed (requirements, etc.)
    rsync -az \
        -e "ssh -o StrictHostKeyChecking=no -i $KEY_FILE" \
        --include 'requirements.txt' \
        --include 'CLAUDE.md' \
        --exclude '*' \
        "$PROJECT_DIR/" "ubuntu@$PUBLIC_IP:~/wss_test/"

    # Ensure __init__.py files exist
    $SSH "touch ~/wss_test/evolution/__init__.py ~/wss_test/evolution/data/__init__.py ~/wss_test/evolution/ga/__init__.py ~/wss_test/evolution/strategies/__init__.py 2>/dev/null || true"

    log "Code synced."
}

sync_quote_data() {
    # Sync quote parquets — tries S3 first (fast from EC2), falls back to rsync
    load_state
    SSH="ssh -o StrictHostKeyChecking=no -i $KEY_FILE ubuntu@$PUBLIC_IP"

    log "Syncing quote data..."

    # Check if instance can reach S3 (has IAM role)
    if $SSH "aws s3 ls s3://$S3_BUCKET/ --max-items 1" &>/dev/null; then
        log "Pulling quote data from S3 (fast path)..."
        $SSH << S3PULL
set -e
cd ~/wss_test
mkdir -p data
for ASSET in btc eth sol xrp; do
    mkdir -p "data/\${ASSET}_quotes"
    aws s3 sync "s3://$S3_BUCKET/\${ASSET}_quotes/" "data/\${ASSET}_quotes/" --quiet &
done
wait
echo "S3 sync complete."
S3PULL
    else
        log "No S3 access — rsyncing quote data directly (~86MB)..."
        for ASSET in btc eth sol xrp; do
            rsync -az \
                -e "ssh -o StrictHostKeyChecking=no -i $KEY_FILE" \
                "$PROJECT_DIR/data/${ASSET}_quotes/" \
                "ubuntu@$PUBLIC_IP:~/wss_test/data/${ASSET}_quotes/" &
        done
        wait
        log "Quote data rsynced."
    fi
}

run_evolution_remote() {
    load_state
    SSH="ssh -o StrictHostKeyChecking=no -i $KEY_FILE ubuntu@$PUBLIC_IP"

    local pop_size="${1:-50}"
    local generations="${2:-20}"
    local symbols="${3:-BTCUSDT}"
    local workers="${4:-}"
    local min_trades="${5:-20}"
    local start_date="${6:-}"
    local end_date="${7:-}"

    log "Running GA evolution: pop=$pop_size, gen=$generations, symbols=$symbols"

    # Build optional args
    local extra_args=""
    if [ -n "$workers" ]; then
        extra_args="$extra_args --workers $workers"
    fi
    if [ -n "$start_date" ]; then
        extra_args="$extra_args --start-date $start_date"
        log "  Training window: $start_date → ${end_date:-end}"
    fi
    if [ -n "$end_date" ]; then
        extra_args="$extra_args --end-date $end_date"
    fi

    $SSH << EVOLVE
set -e
cd ~/wss_test
export PATH="\$HOME/.bun/bin:\$PATH"

echo "Instance: \$(uname -m), \$(nproc) vCPU, \$(free -h | awk '/Mem:/{print \$2}') RAM"

# Run evolution
.venv/bin/python3 -m evolution.ga.runner \
    --data-dir data/ \
    --output-dir evolution/runs/ \
    --symbols $symbols \
    --pop-size $pop_size \
    --generations $generations \
    --min-trades $min_trades \
    $extra_args \
    2>&1 | tee evolution/runs/evolution.log

echo ""
echo "Evolution complete. Uploading results to S3..."
aws s3 sync evolution/runs/ "s3://$S3_BUCKET/evolution_runs/latest/" --quiet
echo "Results uploaded to s3://$S3_BUCKET/evolution_runs/latest/"
EVOLVE

    log "Evolution finished. Pulling results..."
    pull_evolution
}

pull_evolution() {
    load_state
    log "Pulling evolution results from instance..."

    mkdir -p "$PROJECT_DIR/evolution/runs"

    rsync -avz \
        -e "ssh -o StrictHostKeyChecking=no -i $KEY_FILE" \
        "ubuntu@$PUBLIC_IP:~/wss_test/evolution/runs/" \
        "$PROJECT_DIR/evolution/runs/"

    log "Results pulled to evolution/runs/"

    # Show summary if state file exists
    if [ -f "$PROJECT_DIR/evolution/runs/evolution_state.json" ]; then
        log "Evolution summary:"
        python3 -c "
import json
state = json.loads(open('$PROJECT_DIR/evolution/runs/evolution_state.json').read())
print(f\"  Run: {state['run_id']}\")
print(f\"  Generations: {state['current_generation'] + 1}\")
print(f\"  Best fitness: {state['best_fitness_ever']:.3f}\")
for g in state.get('generation_history', []):
    print(f\"    Gen {g['generation']}: best={g['best_fitness']:.3f}, avg={g['avg_fitness']:.3f}, trades={g['n_trades_best']}, pnl=\${g['total_pnl_best']:.2f}\")
"
    fi
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
    create-volume)
        SIZE="${1:-200}"
        create_volume "$SIZE"
        ;;
    delete-volume)
        delete_volume
        ;;
    volume-status)
        volume_status
        ;;
    launch-directional)
        ITYPE="${1:-r8g.2xlarge}"
        launch_directional "$ITYPE"
        ;;
    download-bybit)
        START="${1:-2025-05-01}"
        END="${2:-2026-03-16}"
        EXTRA="${3:---resume}"
        run_download_bybit "$START" "$END" "$EXTRA"
        ;;
    preprocess)
        START="${1:-}"
        END="${2:-}"
        run_preprocess_remote "$START" "$END"
        ;;
    sync-directional)
        sync_directional
        ;;
    stage-bybit)
        stage_bybit
        ;;
    stage-quotes)
        stage_quotes
        ;;
    launch)
        launch_instance
        sync_auth
        sync_data
        log "Instance ready. Use './swarm/cloud/launch.sh run \"prompt\" 3' to start a swarm."
        ;;
    launch-evolve)
        # Default to Graviton4 compute-optimized (48 vCPU, 96GB, ~$1.04/hr on-demand)
        INSTANCE_TYPE="${POLYBOT_INSTANCE_TYPE:-c8g.12xlarge}"
        launch_instance
        if [ -z "${POLYBOT_AMI:-}" ]; then
            setup_evolution_instance
            setup_evolution
        else
            log "Custom AMI — skipping base setup."
        fi
        sync_evolution
        sync_quote_data
        log "Instance ready for evolution. Use './swarm/cloud/launch.sh evolve 100 30' to start."
        ;;
    create-ami)
        load_state
        AMI_NAME="polybot-evolution-$(date +%Y%m%d)"
        log "Creating AMI '$AMI_NAME' from instance $INSTANCE_ID (no reboot)..."
        AMI_ID=$(aws ec2 create-image \
            --region "$AMI_REGION" \
            --instance-id "$INSTANCE_ID" \
            --name "$AMI_NAME" \
            --description "Ubuntu 24.04 ARM64 + Python venv + nautilus_trader + AWS CLI" \
            --no-reboot \
            --query 'ImageId' \
            --output text)
        log "AMI creation started: $AMI_ID"
        log "Wait for it: aws ec2 describe-images --image-ids $AMI_ID --query 'Images[0].State'"
        log "Once available, set POLYBOT_AMI=$AMI_ID to skip setup_instance on future launches."
        ;;
    run)
        SEED="${1:-explore novel orderbook signals}"
        AGENTS="${2:-3}"
        run_swarm_remote "$SEED" "$AGENTS"
        ;;
    evolve)
        POP="${1:-50}"
        GENS="${2:-20}"
        SYMS="${3:-BTCUSDT}"
        WORKERS="${4:-}"
        MIN_TRADES="${5:-20}"
        START_DATE="${6:-}"
        END_DATE="${7:-}"
        run_evolution_remote "$POP" "$GENS" "$SYMS" "$WORKERS" "$MIN_TRADES" "$START_DATE" "$END_DATE"
        ;;
    sync)
        sync_data
        sync_auth
        ;;
    sync-evolve)
        sync_evolution
        ;;
    pull)
        pull_results
        ;;
    pull-evolve)
        pull_evolution
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
        echo "Usage: $0 <command> [args...]"
        echo ""
        echo "Swarm commands:"
        echo "  stage               Upload orderbook data to S3"
        echo "  launch              Spin up instance for swarm runs"
        echo "  run \"prompt\" N      Run swarm on remote instance"
        echo "  sync                Re-sync code + swarm data from S3"
        echo "  pull                Pull swarm results from instance"
        echo ""
        echo "Directional classifier commands:"
        echo "  create-volume [GB]  Create persistent EBS volume (default: 200GB gp3)"
        echo "  delete-volume       Permanently delete the EBS volume"
        echo "  volume-status       Show volume state and size"
        echo "  launch-directional [TYPE]"
        echo "                      Launch instance with data volume at /data (default: r8g.2xlarge)"
        echo "  download-bybit [START] [END]"
        echo "                      Download Bybit orderbook+trades to /data (default: 2025-05-01 → 2026-03-16)"
        echo "  preprocess [START] [END]"
        echo "                      Run v2 feature preprocessing (regime + triple barrier)"
        echo "  sync-directional    Re-sync pipeline code to instance"
        echo "  stage-bybit         Backup /data contents to S3"
        echo ""
        echo "Evolution commands:"
        echo "  stage-quotes        Upload Binance quote parquets to S3"
        echo "  launch-evolve       Spin up instance for GA evolution"
        echo "  evolve POP GEN SYM WORKERS MIN_TRADES START END"
        echo "                      Run GA evolution (default: 50 pop, 20 gen, BTCUSDT)"
        echo "  sync-evolve         Re-sync code + quote data from S3"
        echo "  pull-evolve         Pull evolution results from instance"
        echo ""
        echo "General:"
        echo "  ssh                 SSH into instance"
        echo "  status              Show instance status and spot price"
        echo "  create-ami          Create AMI from running instance (bake deps)"
        echo "  teardown            Terminate instance"
        echo ""
        echo "Environment variables:"
        echo "  POLYBOT_INSTANCE_TYPE  Override instance type (default: r8g.2xlarge / c8g.12xlarge for evolve)"
        echo "  POLYBOT_AMI            Use custom AMI (skip dep install, ~2min faster)"
        echo "  POLYBOT_USE_SPOT       Set to 'true' for spot instances (default: on-demand)"
        echo "  POLYBOT_SPOT_MAX       Max spot price (default: 0.75)"
        echo ""
        echo "Examples:"
        echo "  $0 launch-evolve                       # launch + full setup"
        echo "  $0 evolve 100 30 BTCUSDT 48 20 2025-09-01 2025-12-01"
        echo "  $0 create-ami                          # bake deps into AMI"
        echo "  POLYBOT_AMI=ami-xxx $0 launch-evolve   # fast launch with baked AMI"
        echo "  POLYBOT_INSTANCE_TYPE=c8g.4xlarge $0 launch-evolve"
        exit 1
        ;;
esac
