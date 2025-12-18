#!/bin/bash
# Albatross Control Plane - Start Script
# Usage: bash scripts/start_control_plane.sh [port] [host]

set -e

PORT="${1:-9000}"
HOST="${2:-0.0.0.0}"

echo "=============================================="
echo "Albatross Control Plane"
echo "=============================================="
echo "Host: $HOST"
echo "Port: $PORT"
echo "=============================================="

# Activate virtual environment (optional)
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

python3 -m control_plane.main --host "$HOST" --port "$PORT"
