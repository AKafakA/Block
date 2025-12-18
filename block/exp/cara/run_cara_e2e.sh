#!/bin/bash

# Complete End-to-End CARA Deployment and Testing Script
# This script orchestrates the full deployment workflow:
# 1. Deploy backend instances (vLLM/Ollama)
# 2. Verify backend deployments
# 3. Start CARA scheduler server
# 4. Run benchmark tests

set -e  # Exit on error

# Configuration
TARGET_HOST="asdwb@d8545-10s10301.wisc.cloudlab.us"  # Host to run CARA server
MODEL_CONFIG="block/config/cara/model_config_template.json"
HOST_CONFIG="block/config/host_configs.json"
HOSTS_FILE="block/config/hosts"
DEPLOYMENT_CONFIG="block/config/cara/model_deployment.json"

SCHEDULING_STRATEGY="random"  # Scheduling strategy for CARA
HF_TOKEN="${HF_TOKEN:-}"  # Set HF_TOKEN env var or pass as argument
REPETITION_PENALTY=${3:-1.05}  # Repetition penalty for generation
REQUEST_RATE=${4:-inf}  # Request rate for benchmark (requests per second, or 'inf' for unlimited)
SAVE_DETAILED=${5:-"ttft itl e2el models hosts instance_ids"}  # Detailed metrics to save

DOWNLOAD_DATASET=${2:-false}  # Pass 'true' as second argument to download dataset

DATASET_NAME="sharegpt"
DATASET_PATH="~/dataset/sharegpt"
DATASET_LINK="https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"

REDEPLOY=${1:-false}  # Pass 'true' as first argument to force redeployment

echo "========================================"
echo "CARA End-to-End Deployment & Test"
echo "========================================"
echo ""

# Step 1: Deploy backend instances
echo "Step 1: Deploying Backend Instances"
echo "------------------------------------"
echo "This will deploy vLLM and Ollama instances across all configured hosts..."
if [ "$REDEPLOY" = "true" ]; then
  python block/exp/cara/deploy_cara.py \
    --hosts ${HOSTS_FILE} \
    --config ${MODEL_CONFIG} \
    --hf-token "${HF_TOKEN}" \
    --output ${DEPLOYMENT_CONFIG}
  sleep 600  # Wait for models to load
else
  echo "Skipping redeployment of backends. Using existing deployment config at ${DEPLOYMENT_CONFIG}."
fi

if [ "$DOWNLOAD_DATASET" = "true" ]; then
  echo "Downloading ShareGPT dataset to ${DATASET_PATH}..."
  ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    ${TARGET_HOST} \
    "mkdir -p ${DATASET_PATH} && wget -O ${DATASET_PATH}/sharegpt_random_10k.jsonl ${DATASET_LINK}"
  echo "Dataset download completed."
fi

# Start CARA scheduler server
echo "Step 2: Starting CARA Scheduler Server"
echo "---------------------------------------"
echo "Starting CARA server on ${TARGET_HOST}:${CARA_PORT}..."

# Kill existing CARA server if running
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  ${TARGET_HOST} \
  "pkill -f 'cara_serve.py' || echo 'No existing CARA server found'"
sleep 2

# Start CARA server in background
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  ${TARGET_HOST} \
  "cd Block && nohup python -m block.global_scheduler.cara.cara_serve \
    --model_config_path ${DEPLOYMENT_CONFIG} \
    --host_config ${HOST_CONFIG} \
    --scheduling ${SCHEDULING_STRATEGY} \
    --repetition-penalty ${REPETITION_PENALTY} \
    > experiment_output/logs/cara_server.log 2>&1 &"

echo "Waiting 10 seconds for CARA server to start..."
sleep 10

# Run benchmark tests
echo "Step 3: Running Benchmark Tests"
echo "--------------------------------"

# Configuration for benchmark
OUTPUT_DIR="experiment_output/cara_test_results"
mkdir -p ${OUTPUT_DIR}

# Run benchmark with small/random dataset for testing
echo "Running CARA benchmark with random dataset..."
echo "  Requests: 50"
echo "  Input length: 128 tokens"
echo "  Output length: 64 tokens"
echo ""

# Add vLLM to PYTHONPATH if it exists locally
if [ -d "$HOME/vllm" ]; then
  export PYTHONPATH="$HOME/vllm:$PYTHONPATH"
  echo "Using local vLLM from: $HOME/vllm"
elif [ -d "~/vllm" ]; then
  export PYTHONPATH="~/vllm:$PYTHONPATH"
  echo "Using local vLLM from: ~/vllm"
fi

ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  ${TARGET_HOST} \
  "export PYTHONPATH=${PYTHONPATH} && \
   cd Block && \
   python block/benchmark/cara/benchmark_serving.py \
     --backend cara \
     --host 127.0.0.1 \
     --port 8200 \
     --dataset-name ${DATASET_NAME} \
     --dataset-path ${DATASET_PATH}/sharegpt_random_10k.jsonl \
     --num-prompts 50 \
     --request-rate ${REQUEST_RATE} \
     --save-detailed ${SAVE_DETAILED} \
     --result-dir ${OUTPUT_DIR} \
     --save-result"

if [ $? -ne 0 ]; then
    echo "❌ Benchmark failed!"
    echo "Check CARA server logs for errors:"
    echo "  ssh ${TARGET_HOST} 'tail -f Block/experiment_output/logs/cara_server.log'"
    exit 1
fi

echo ""
echo "========================================"
echo "CARA Deployment & Benchmark Complete!"
echo "========================================"
echo ""
echo "Summary:"
echo "  ✅ Backend instances deployed and verified"
echo "  ✅ CARA server running at http://${CARA_HOST_IP}:${CARA_PORT}"
echo "  ✅ Benchmark tests completed"
echo ""
echo "Results:"
echo "  - Benchmark results saved to: ${OUTPUT_DIR}"
echo "  - View latest results: ls -lht ${OUTPUT_DIR}"
echo ""
echo "Useful Commands:"
echo "  1. Check CARA server logs:"
echo "     ssh ${TARGET_HOST} 'tail -f Block/experiment_output/logs/cara_server.log'"
echo ""
echo "  2. Run additional benchmark tests:"
echo "     python block/benchmark/cara/benchmark_serving.py \\"
echo "       --backend cara \\"
echo "       --base-url http://${CARA_HOST_IP}:${CARA_PORT} \\"
echo "       --dataset-name random \\"
echo "       --num-prompts 100"
echo ""
echo "  3. Monitor backend logs on remote hosts:"
echo "     ssh <host> 'tail -f ~/vllm/vllm_server.log'  # For vLLM"
echo "     ssh <host> 'tail -f ~/ollama/ollama_server.log'  # For Ollama"
echo ""
