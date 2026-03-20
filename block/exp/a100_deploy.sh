#!/bin/bash
# A100 deployment script — run ON the A100 node itself
# Usage: bash a100_deploy.sh [chunked|no_chunked]

CHUNKED=${1:-chunked}
MODEL="meta-llama/Llama-2-7b-hf"
MAX_SEQS=96
MAX_LEN=4096
export HF_HOME=/mydata/huggingface
export HF_TOKEN=
export VLLM_USE_V1=0

echo "=== A100 Deploy: $CHUNKED ==="

# Kill existing
pkill -9 -u $(whoami) -f "vllm|predictor|global_scheduler" 2>/dev/null
sleep 3

# Build chunked flag
CHUNKED_FLAG=""
if [ "$CHUNKED" = "chunked" ]; then
    CHUNKED_FLAG="--enable-chunked-prefill --max-num-batched-tokens 512"
fi

# Deploy 4 vLLM instances
cd ~/vllm
mkdir -p ~/Block/experiment_output/logs
for gpu in 0 1 2 3; do
    port=$((8000 + gpu))
    echo "Starting vLLM on GPU $gpu port $port..."
    (CUDA_VISIBLE_DEVICES=$gpu nohup python -m vllm.entrypoints.api_server \
        --model $MODEL --port $port \
        --max-num-seqs $MAX_SEQS --max-model-len $MAX_LEN \
        $CHUNKED_FLAG --swap-space 0 \
        > ~/Block/experiment_output/logs/vllm_gpu${gpu}.log 2>&1 &)
done

echo "Waiting 90s for model loading..."
sleep 90

# Verify vLLM
up=0
for port in 8000 8001 8002 8003; do
    curl -s --connect-timeout 3 http://127.0.0.1:$port/health >/dev/null 2>&1 && up=$((up+1))
done
echo "vLLM: $up/4 UP"

# Verify schedule_trace endpoint
echo "Testing schedule_trace..."
curl -s --connect-timeout 3 http://127.0.0.1:8000/schedule_trace 2>&1 | head -1
echo ""

if [ $up -lt 4 ]; then
    echo "ERROR: Not all vLLM started. Check logs."
    exit 1
fi

echo "=== vLLM deployment complete ==="
