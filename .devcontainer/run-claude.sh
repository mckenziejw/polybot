#!/bin/bash
# Run Claude Code inside the devcontainer with permissions disabled.
# Usage: ./run-claude.sh [additional claude args...]
#
# Prerequisites:
#   docker build -t polybot-dev .devcontainer/
#
# Or use VSCode's "Reopen in Container" with the devcontainer.json.

set -euo pipefail

IMAGE="polybot-dev"
WORKSPACE="$(cd "$(dirname "$0")/.." && pwd)"

# Build if image doesn't exist
if ! docker image inspect "$IMAGE" &>/dev/null; then
    echo "Building $IMAGE..."
    docker build -t "$IMAGE" "$WORKSPACE/.devcontainer"
fi

# The host project path (used by Claude for project state keys)
HOST_PROJECT_KEY="-home-josh-Documents-projects-wss-test"

# Resource limits: leave headroom for host OS
docker run -it --rm \
    --memory=24g \
    --cpus=16 \
    -v "$WORKSPACE:$WORKSPACE" \
    -v "$HOME/.claude:/home/dev/.claude" \
    -v "$HOME/.claude.json:/home/dev/.claude.json" \
    -v "$HOME/.anthropic:/home/dev/.anthropic" \
    -e ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-}" \
    -e HOME="/home/dev" \
    -w "$WORKSPACE" \
    "$IMAGE" \
    claude --dangerously-skip-permissions "$@"
