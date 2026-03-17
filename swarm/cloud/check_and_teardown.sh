#!/bin/bash
# Check if training/sweep is complete, pull results, and stop the instance.
#
# Usage: bash swarm/cloud/check_and_teardown.sh

set -euo pipefail

KEY=~/.ssh/polybot-swarm.pem
INSTANCE_ID="i-0e64ffed913469cca"
LOCAL_RESULTS="data/directional_results_v4"

# Get current IP (may change after stop/start)
IP=$(aws ec2 describe-instances --instance-ids "$INSTANCE_ID" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' --output text 2>/dev/null)
HOST="ubuntu@$IP"

STATE=$(aws ec2 describe-instances --instance-ids "$INSTANCE_ID" \
    --query 'Reservations[0].Instances[0].State.Name' --output text 2>/dev/null)

if [ "$STATE" != "running" ]; then
    echo "Instance is $STATE, not running."
    exit 1
fi

echo "Checking training status on $HOST..."

COMPLETE=$(ssh -i "$KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=10 "$HOST" \
    'test -f /tmp/training_complete && echo "yes" || echo "no"' 2>/dev/null)

if [ "$COMPLETE" != "yes" ]; then
    echo "Training still in progress."
    ssh -i "$KEY" "$HOST" 'tail -20 /tmp/train_v4_*.log 2>/dev/null' 2>/dev/null
    exit 0
fi

echo "Training complete! Pulling results..."

mkdir -p "$LOCAL_RESULTS/sweep_logs"

# Pull all v4 results directories
ssh -i "$KEY" "$HOST" 'ls -d data/directional_results_v4_*' 2>/dev/null | while read dir; do
    name=$(basename "$dir")
    mkdir -p "$LOCAL_RESULTS/$name"
    scp -i "$KEY" -r "$HOST:/home/ubuntu/$dir/*" "$LOCAL_RESULTS/$name/" 2>/dev/null
    echo "  Pulled $name"
done

# Pull all training logs
scp -i "$KEY" "$HOST:/tmp/train_v4_*.log" "$LOCAL_RESULTS/sweep_logs/" 2>/dev/null

echo "Results pulled to $LOCAL_RESULTS"

# Print summary
echo ""
echo "=== RESULTS ==="
ssh -i "$KEY" "$HOST" 'grep -A 30 "TRAINING WINDOW SWEEP SUMMARY" /tmp/train_v4_*.log 2>/dev/null' 2>/dev/null

# Stop instance
echo ""
read -p "Stop instance $INSTANCE_ID? [y/N] " confirm
if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
    aws ec2 stop-instances --instance-ids "$INSTANCE_ID" > /dev/null
    echo "Instance stopping. EBS volume preserved."
else
    echo "Instance left running."
fi
