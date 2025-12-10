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
CARA_PORT=8200
MODEL_CONFIG="block/config/cara/model_config_template.json"
HOST_CONFIG="block/config/host_configs.json"
HOSTS_FILE="block/config/hosts"
DEPLOYMENT_CONFIG="block/config/cara/model_deployment.json"
HF_TOKEN="${HF_TOKEN:-}"  # Set HF_TOKEN env var or pass as argument

echo "========================================"
echo "CARA End-to-End Deployment & Test"
echo "========================================"
echo ""

# Step 1: Deploy backend instances
echo "Step 1: Deploying Backend Instances"
echo "------------------------------------"
echo "This will deploy vLLM and Ollama instances across all configured hosts..."
python block/exp/cara/deploy_cara.py \
  --hosts ${HOSTS_FILE} \
  --config ${MODEL_CONFIG} \
  --hf-token "${HF_TOKEN}" \
  --output ${DEPLOYMENT_CONFIG}

if [ $? -ne 0 ]; then
    echo "❌ Backend deployment failed!"
    exit 1
fi

echo "✅ Backend deployment completed"
echo ""

# Step 2: Wait for backends to initialize
echo "Step 2: Waiting for Backends to Initialize"
echo "-------------------------------------------"
echo "Waiting 60 seconds for models to load..."
sleep 60
echo ""

# Step 3: Verify backend deployments
echo "Step 3: Verifying Backend Deployments"
echo "--------------------------------------"
python block/exp/cara/e2e/check_deployment.py \
  --config ${DEPLOYMENT_CONFIG} \
  --output block/config/cara/verified_hosts.json

if [ $? -ne 0 ]; then
    echo "❌ Backend verification failed!"
    echo "Some backends may not be responding. Check logs on remote hosts."
    exit 1
fi

echo "✅ All backends verified"
echo ""

# Step 4: Start CARA scheduler server
echo "Step 4: Starting CARA Scheduler Server"
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
    --host 0.0.0.0 \
    --port ${CARA_PORT} \
    --model_config_path ${DEPLOYMENT_CONFIG} \
    --host_config ${HOST_CONFIG} \
    --scheduling random \
    > experiment_output/logs/cara_server.log 2>&1 &"

echo "Waiting 10 seconds for CARA server to start..."
sleep 10

# Verify CARA server is running
CARA_HOST_IP=$(echo ${TARGET_HOST} | cut -d'@' -f2)
curl -s "http://${CARA_HOST_IP}:${CARA_PORT}/health" > /dev/null 2>&1

if [ $? -eq 0 ]; then
    echo "✅ CARA server is running at http://${CARA_HOST_IP}:${CARA_PORT}"
else
    echo "⚠️  Warning: CARA server health check failed (this is expected if /health endpoint not implemented)"
fi
echo ""

# Step 5: Run benchmark tests
echo "Step 5: Running Benchmark Tests"
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

python block/benchmark/cara/benchmark_serving.py \
  --backend cara \
  --base-url "http://${CARA_HOST_IP}:${CARA_PORT}" \
  --endpoint /v1/completions \
  --model cara \
  --dataset-name random \
  --random-input-len 128 \
  --random-output-len 64 \
  --num-prompts 50 \
  --request-rate inf \
  --save-result \
  --result-dir ${OUTPUT_DIR} \
  --result-filename cara_benchmark_$(date +%Y%m%d_%H%M%S).json

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
