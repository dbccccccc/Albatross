#!/bin/bash
# RWKV-7 Inference Server - Ubuntu Setup Script
# Usage: bash scripts/setup_ubuntu.sh

set -e

echo "=============================================="
echo "RWKV-7 Inference Server - Ubuntu Setup"
echo "=============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    echo -e "${RED}Please do not run as root${NC}"
    exit 1
fi

# Check NVIDIA GPU
echo -e "${YELLOW}Checking NVIDIA GPU...${NC}"
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}nvidia-smi not found. Please install NVIDIA drivers first.${NC}"
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo -e "${GREEN}GPU detected!${NC}"

# Check Python version
echo -e "${YELLOW}Checking Python...${NC}"
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
if [[ $(echo "$PYTHON_VERSION >= 3.9" | bc -l) -eq 0 ]]; then
    echo -e "${RED}Python 3.9+ required. Found: $PYTHON_VERSION${NC}"
    exit 1
fi
echo -e "${GREEN}Python $PYTHON_VERSION detected!${NC}"

# Create virtual environment
echo -e "${YELLOW}Creating virtual environment...${NC}"
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}Virtual environment created!${NC}"
else
    echo -e "${GREEN}Virtual environment already exists!${NC}"
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo -e "${YELLOW}Upgrading pip...${NC}"
pip install --upgrade pip -q

# Install PyTorch
echo -e "${YELLOW}Installing PyTorch with CUDA support...${NC}"
pip install torch --index-url https://download.pytorch.org/whl/cu124 -q

# Verify CUDA
echo -e "${YELLOW}Verifying CUDA support...${NC}"
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'PyTorch {torch.__version__} with CUDA {torch.version.cuda}')"
if [ $? -ne 0 ]; then
    echo -e "${RED}CUDA verification failed!${NC}"
    exit 1
fi
echo -e "${GREEN}CUDA support verified!${NC}"

# Install server dependencies
echo -e "${YELLOW}Installing server dependencies...${NC}"
pip install fastapi uvicorn pydantic aiohttp requests -q
echo -e "${GREEN}Dependencies installed!${NC}"

# Check for model
echo -e "${YELLOW}Checking for model files...${NC}"
MODEL_DIR="models"
mkdir -p $MODEL_DIR

if [ -z "$(ls -A $MODEL_DIR/*.pth 2>/dev/null)" ]; then
    echo -e "${YELLOW}No model found in $MODEL_DIR/${NC}"
    echo ""
    echo "Please download a RWKV-7 model:"
    echo "  Option 1: huggingface-cli download BlinkDL/rwkv-7-world --local-dir models/"
    echo "  Option 2: wget https://huggingface.co/BlinkDL/rwkv-7-world/resolve/main/RWKV-7-World-0.4B-v2.8-20241022-ctx4096.pth -P models/"
    echo ""
else
    echo -e "${GREEN}Model files found:${NC}"
    ls -la $MODEL_DIR/*.pth
fi

# Check tokenizer
echo -e "${YELLOW}Checking tokenizer...${NC}"
if [ ! -f "reference/rwkv_vocab_v20230424.txt" ]; then
    echo -e "${RED}Tokenizer not found at reference/rwkv_vocab_v20230424.txt${NC}"
    exit 1
fi
echo -e "${GREEN}Tokenizer found!${NC}"

echo ""
echo "=============================================="
echo -e "${GREEN}Setup complete!${NC}"
echo "=============================================="
echo ""
echo "To start the server:"
echo "  source venv/bin/activate"
echo "  python -m server.main --model-path models/YOUR_MODEL_NAME --port 8000"
echo ""
echo "Example:"
echo "  python -m server.main --model-path models/RWKV-7-World-0.4B-v2.8-20241022-ctx4096 --port 8000"
echo ""
