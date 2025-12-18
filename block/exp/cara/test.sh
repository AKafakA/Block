#!/bin/bash

# Simple CARA Test Script
# Assumes backend instances (vLLM/Ollama) are already running
# This script will:
#   1. Start CARA scheduler server
#   2. Send 100 random requests to test random scheduling
# Prerequisites: Customized vLLM with cara backend at ~/vllm

set -e  # Exit on error

# Configuration
CARA_HOST="127.0.0.1"
CARA_PORT="8200"
CARA_URL="http://${CARA_HOST}:${CARA_PORT}"
OUTPUT_DIR="experiment_output/cara_test_results"
MODEL_CONFIG="block/config/cara/model_deployment.json"
HOST_CONFIG="block/config/host_configs.json"
REPETITION_PENALTY=${1:-1.05}  # Repetition penalty for generation
REQUEST_RATE=${2:-inf}  # Request rate for benchmark
SAVE_DETAILED=${3:-"ttft itl e2el models hosts instance_ids"}  # Detailed metrics to save

echo "========================================"
echo "CARA Simple Test"
echo "========================================"
echo "CARA Server: ${CARA_URL}"
echo "Output Directory: ${OUTPUT_DIR}"
echo ""

# Step 1: Start CARA scheduler server
echo "Step 1: Starting CARA Scheduler Server"
echo "---------------------------------------"

# Kill existing CARA server if running
pkill -f 'cara_serve.py' || echo "No existing CARA server found"
sleep 2

# Create output directory
mkdir -p ${OUTPUT_DIR}
mkdir -p experiment_output/logs

# Start CARA server in background
nohup python -m block.global_scheduler.cara.cara_serve \
  --host ${CARA_HOST} \
  --port ${CARA_PORT} \
  --model_config_path ${MODEL_CONFIG} \
  --host_config ${HOST_CONFIG} \
  --scheduling random \
  --repetition-penalty ${REPETITION_PENALTY} \
  > experiment_output/logs/cara_server.log 2>&1 &

CARA_PID=$!
echo "CARA server started (PID: ${CARA_PID})"
echo "Waiting 10 seconds for CARA server to initialize..."
sleep 10

# Verify CARA server is running
if ! ps -p ${CARA_PID} > /dev/null; then
    echo "❌ CARA server failed to start! Check logs:"
    tail -n 20 experiment_output/logs/cara_server.log
    exit 1
fi

echo "✅ CARA server is running"
echo ""

# Step 2: Run benchmark test
echo "Step 2: Running Benchmark Test"
echo "-------------------------------"
echo "Sending 100 random requests to CARA..."

# Add vllm to PYTHONPATH to use the customized version
export PYTHONPATH="$HOME/vllm:$PYTHONPATH"

python block/benchmark/cara/benchmark_serving.py \
  --backend cara \
  --host ${CARA_HOST} \
  --port ${CARA_PORT} \
  --dataset-name random \
  --random-input-len 128 \
  --random-output-len 64 \
  --num-prompts 100 \
  --request-rate ${REQUEST_RATE} \
  --save-detailed ${SAVE_DETAILED} \
  --result-dir ${OUTPUT_DIR} \
  --result-filename cara_simple_test.json \
  --save-result

echo ""
echo "========================================"
echo "Test Completed!"
echo "========================================"
echo "Results saved to: ${OUTPUT_DIR}/cara_simple_test.json"
echo ""
echo "To view results:"
echo "  cat ${OUTPUT_DIR}/cara_simple_test.json | jq '.'"
echo ""
echo "Key metrics to check:"
echo "  - completed: Number of successful requests (should be 100)"
echo "  - request_throughput: Requests per second"
echo "  - mean_ttft_ms: Average time to first token"
echo "  - mean_tpot_ms: Average time per output token"
echo ""
echo "CARA server is still running (PID: ${CARA_PID})"
echo "To stop: kill ${CARA_PID}"
echo "To view logs: tail -f experiment_output/logs/cara_server.log"
