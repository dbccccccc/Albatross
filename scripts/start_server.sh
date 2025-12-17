#!/bin/bash
# RWKV-7 Inference Server - Start Script
# Usage: bash scripts/start_server.sh [model_path] [port]

set -e

# Default values
MODEL_PATH="${1:-models/RWKV-7-World-0.4B-v2.8-20241022-ctx4096}"
PORT="${2:-8000}"
MAX_BATCH_SIZE="${3:-64}"

echo "=============================================="
echo "RWKV-7 Inference Server"
echo "=============================================="
echo "Model: $MODEL_PATH"
echo "Port: $PORT"
echo "Max Batch Size: $MAX_BATCH_SIZE"
echo "=============================================="

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    echo "Virtual environment not found. Run setup_ubuntu.sh first."
    exit 1
fi

# Check if model exists
if [ ! -f "${MODEL_PATH}.pth" ]; then
    echo "Model not found: ${MODEL_PATH}.pth"
    echo "Available models:"
    ls -la models/*.pth 2>/dev/null || echo "No models found in models/"
    exit 1
fi

# Start server
python -m server.main \
    --model-path "$MODEL_PATH" \
    --tokenizer-path reference/rwkv_vocab_v20230424.txt \
    --host 0.0.0.0 \
    --port "$PORT" \
    --max-batch-size "$MAX_BATCH_SIZE" \
    --max-prefill-batch 16 \
    --prefill-chunk-size 512
