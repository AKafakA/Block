#!/bin/bash

# CARA Broadcasting Test Script for Training Data Collection
# =============================================================
# This script collects multi-model responses for the same prompts to train:
#   1. Model Quality Estimator: Predict output quality given prompt and model
#   2. Response Length Predictor: Predict response length given prompt and model
#
# How it works:
#   - Enables broadcasting mode in CARA server
#   - For each request, queries multiple models in parallel
#   - Saves all model responses with their metrics (TTFT, E2EL, output length, etc.)
#   - Results stored in per-request format with broadcast_results field
#
# Prerequisites:
#   - Backend instances (vLLM/Ollama) already running
#   - Model deployment config exists at block/config/cara/model_deployment.json
#   - Customized vLLM with cara backend at ~/vllm
#
# Usage:
#   ./test_broadcasting.sh [MODELS] [DATASET] [NUM_PROMPTS] [REQUEST_RATE] [OUTPUT_SUFFIX] [6th_unused] [CUSTOM_DATASET_PATH]
#
# Examples:
#   # Collect data from all 4 models using custom v3 dataset (default)
#   ./test_broadcasting.sh "Qwen/Qwen2.5-3B Qwen/Qwen2.5-7B Qwen/Qwen2.5-14B Qwen/Qwen2.5-72B" custom 500 inf test1 _ data/cara/best-route-v3-test-500.jsonl
#
#   # Collect data with rate limiting for full 20k run
#   ./test_broadcasting.sh "Qwen/Qwen2.5-3B Qwen/Qwen2.5-7B Qwen/Qwen2.5-14B Qwen/Qwen2.5-72B" custom 20000 4.0 full_20k _ data/cara/best-route-v3.jsonl
#
#   # Override generation params via env vars
#   FREQUENCY_PENALTY=0.0 TEMPERATURE=0.1 ./test_broadcasting.sh ...
#
#   # Use random prompts for quick testing
#   ./test_broadcasting.sh "Qwen/Qwen2.5-3B Qwen/Qwen2.5-7B" random 50 inf debug

set -e  # Exit on error

# =============================================================================
# Configuration Parameters
# =============================================================================

# Model selection (space-separated list of HuggingFace model names)
# These models will be queried in parallel for each request
BROADCAST_MODELS=${1:-"Qwen/Qwen2.5-3B Qwen/Qwen2.5-7B Qwen/Qwen2.5-14B"}

# Dataset selection
DATASET_NAME=${2:-"custom"}  # Options: custom, sharegpt, random, sonnet
CUSTOM_DATASET_PATH=${7:-"data/cara/best-route-v3-test-500.jsonl"}  # Path for custom JSONL dataset
SHAREGPT_DATASET_PATH="~/dataset/sharegpt/sharegpt_random_10k.jsonl"  # Path for sharegpt dataset

# Benchmark parameters
NUM_PROMPTS=${3:-100}  # Number of prompts to process
REQUEST_RATE=${4:-inf}  # Requests per second (inf = unlimited)
OUTPUT_SUFFIX=${5:-"broadcast_data"}  # Suffix for output filename

# CARA server configuration
CARA_HOST="127.0.0.1"
CARA_PORT="8200"
CARA_URL="http://${CARA_HOST}:${CARA_PORT}"

# File paths
MODEL_CONFIG="block/config/cara/model_deployment.json"
HOST_CONFIG="block/config/host_configs.json"
OUTPUT_DIR="experiment_output/cara_broadcast_training_data"

# Server parameters
REPETITION_PENALTY=${REPETITION_PENALTY:-1.0}
SCHEDULING_STRATEGY="random"  # Doesn't matter for broadcasting (all models queried)

# Generation parameters (passed to benchmark_serving.py)
FREQUENCY_PENALTY=${FREQUENCY_PENALTY:-1.2}
TEMPERATURE=${TEMPERATURE:-0.0}
MAX_OUTPUT_TOKENS=${MAX_OUTPUT_TOKENS:-1024}
MAX_TOTAL_LEN=${MAX_TOTAL_LEN:-2048}

# Detailed metrics to save for training
# IMPORTANT: Must include 'response' and 'prompts' to collect actual text for quality estimation
SAVE_DETAILED="prompts response ttft itl e2el input_lens output_lens models hosts instance_ids"

# Random dataset parameters (only used if DATASET_NAME=random)
RANDOM_INPUT_LEN=256
RANDOM_OUTPUT_LEN=128

# =============================================================================
# Display Configuration
# =============================================================================

echo "========================================================================"
echo "CARA Broadcasting Test - Training Data Collection"
echo "========================================================================"
echo ""
echo "PURPOSE:"
echo "  Collect multi-model responses for training:"
echo "    - Model Quality Estimator"
echo "    - Response Length Predictor"
echo ""
echo "CONFIGURATION:"
echo "  CARA Server:          ${CARA_URL}"
echo "  Broadcast Models:     ${BROADCAST_MODELS}"
echo "  Dataset:              ${DATASET_NAME}"
if [ "${DATASET_NAME}" = "custom" ]; then
    echo "  Dataset Path:         ${CUSTOM_DATASET_PATH}"
elif [ "${DATASET_NAME}" = "sharegpt" ]; then
    echo "  Dataset Path:         ${SHAREGPT_DATASET_PATH}"
fi
echo "  Generation:           temp=${TEMPERATURE}, freq_penalty=${FREQUENCY_PENALTY}, rep_penalty=${REPETITION_PENALTY}"
echo "  Max Output Tokens:    ${MAX_OUTPUT_TOKENS}"
echo "  Max Total Len:        ${MAX_TOTAL_LEN}"
echo "  Number of Prompts:    ${NUM_PROMPTS}"
echo "  Request Rate:         ${REQUEST_RATE} qps"
echo "  Output Directory:     ${OUTPUT_DIR}"
echo "  Output Suffix:        ${OUTPUT_SUFFIX}"
echo ""
echo "DATA COLLECTION:"
echo "  Each request will be broadcasted to all selected models"
echo "  Results saved in per-request format with broadcast_results field"
echo "  Format: requests[i].broadcast_results = [{model, ttft, e2el, output_len, response, ...}, ...]"
echo ""
echo "========================================================================"
echo ""

# =============================================================================
# Step 1: Prepare Environment
# =============================================================================

echo "Step 1: Preparing Environment"
echo "------------------------------"

# Create output directory
mkdir -p ${OUTPUT_DIR}
mkdir -p experiment_output/logs

# Add vLLM to PYTHONPATH for customized version
export PYTHONPATH="$HOME/vllm:$PYTHONPATH"
echo "✅ Using local vLLM from: $HOME/vllm"

# Verify model config exists
if [ ! -f "${MODEL_CONFIG}" ]; then
    echo "❌ Model deployment config not found: ${MODEL_CONFIG}"
    echo "Please deploy backends first using deploy_cara.py"
    exit 1
fi
echo "✅ Model config found: ${MODEL_CONFIG}"

# Verify dataset exists
if [ "${DATASET_NAME}" = "custom" ]; then
    EXPANDED_DATASET_PATH="${CUSTOM_DATASET_PATH/#\~/$HOME}"
    if [ ! -f "${EXPANDED_DATASET_PATH}" ]; then
        echo "❌ Custom dataset not found: ${EXPANDED_DATASET_PATH}"
        exit 1
    fi
    echo "✅ Custom dataset found: ${EXPANDED_DATASET_PATH}"
elif [ "${DATASET_NAME}" = "sharegpt" ]; then
    EXPANDED_DATASET_PATH="${SHAREGPT_DATASET_PATH/#\~/$HOME}"
    if [ ! -f "${EXPANDED_DATASET_PATH}" ]; then
        echo "❌ Dataset not found: ${EXPANDED_DATASET_PATH}"
        exit 1
    fi
    echo "✅ Dataset found: ${EXPANDED_DATASET_PATH}"
fi

echo ""

# =============================================================================
# Step 2: Start CARA Server with Broadcasting Enabled
# =============================================================================

echo "Step 2: Starting CARA Server (Broadcasting Mode)"
echo "-------------------------------------------------"

# Kill existing CARA server if running
pkill -f 'cara_serve.py' || echo "No existing CARA server found"
sleep 2

# Convert model list to array for argument passing
read -ra MODEL_ARRAY <<< "${BROADCAST_MODELS}"

# Start CARA server with broadcasting enabled
echo "Starting CARA server with broadcasting to ${#MODEL_ARRAY[@]} models..."
nohup python -m block.global_scheduler.cara.cara_serve \
  --host ${CARA_HOST} \
  --port ${CARA_PORT} \
  --model_config_path ${MODEL_CONFIG} \
  --host_config ${HOST_CONFIG} \
  --scheduling ${SCHEDULING_STRATEGY} \
  --repetition-penalty ${REPETITION_PENALTY} \
  --broadcasting \
  --selected-broadcasted-models ${BROADCAST_MODELS} \
  --enable-predictor-feedback \
  --feedback-sample-rate 1.0 \
  > experiment_output/logs/cara_server_broadcast.log 2>&1 &

CARA_PID=$!
echo "CARA server started (PID: ${CARA_PID})"
echo "Broadcasting enabled for models:"
for model in "${MODEL_ARRAY[@]}"; do
    echo "  - ${model}"
done
echo ""
echo "Waiting 15 seconds for CARA server to initialize..."
sleep 15

# Verify CARA server is running
if ! ps -p ${CARA_PID} > /dev/null; then
    echo "❌ CARA server failed to start! Check logs:"
    tail -n 30 experiment_output/logs/cara_server_broadcast.log
    exit 1
fi

echo "✅ CARA server is running with broadcasting enabled"
echo ""

# =============================================================================
# Step 3: Run Benchmark to Collect Training Data
# =============================================================================

echo "Step 3: Collecting Multi-Model Response Data"
echo "---------------------------------------------"
echo "Sending ${NUM_PROMPTS} requests with broadcasting to ${#MODEL_ARRAY[@]} models..."
echo ""

# Build dataset arguments based on dataset type
DATASET_ARGS=""
if [ "${DATASET_NAME}" = "random" ]; then
    DATASET_ARGS="--dataset-name random --random-input-len ${RANDOM_INPUT_LEN} --random-output-len ${RANDOM_OUTPUT_LEN}"
elif [ "${DATASET_NAME}" = "custom" ]; then
    DATASET_ARGS="--dataset-name custom --dataset-path ${CUSTOM_DATASET_PATH} --custom-output-len ${MAX_OUTPUT_TOKENS} --max-total-len ${MAX_TOTAL_LEN}"
elif [ "${DATASET_NAME}" = "sharegpt" ]; then
    DATASET_ARGS="--dataset-name sharegpt --dataset-path ${SHAREGPT_DATASET_PATH}"
elif [ "${DATASET_NAME}" = "sonnet" ]; then
    DATASET_ARGS="--dataset-name sonnet"
else
    echo "❌ Unknown dataset: ${DATASET_NAME}"
    echo "Supported datasets: custom, sharegpt, random, sonnet"
    exit 1
fi

# Build result filename
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_FILENAME="broadcast_${DATASET_NAME}_${NUM_PROMPTS}prompts_${OUTPUT_SUFFIX}_${TIMESTAMP}.json"

# Run benchmark
python block/benchmark/cara/benchmark_serving.py \
  --backend cara \
  --host ${CARA_HOST} \
  --port ${CARA_PORT} \
  ${DATASET_ARGS} \
  --num-prompts ${NUM_PROMPTS} \
  --request-rate ${REQUEST_RATE} \
  --temperature ${TEMPERATURE} \
  --frequency-penalty ${FREQUENCY_PENALTY} \
  --repetition-penalty ${REPETITION_PENALTY} \
  --save-detailed ${SAVE_DETAILED} \
  --result-dir ${OUTPUT_DIR} \
  --result-filename ${RESULT_FILENAME} \
  --save-result

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Benchmark failed!"
    echo "Check CARA server logs for errors:"
    echo "  tail -f experiment_output/logs/cara_server_broadcast.log"
    exit 1
fi

echo ""
echo "========================================================================"
echo "✅ Training Data Collection Complete!"
echo "========================================================================"
echo ""

# =============================================================================
# Display Results and Next Steps
# =============================================================================

RESULT_PATH="${OUTPUT_DIR}/${RESULT_FILENAME}"

echo "RESULTS SAVED TO:"
echo "  ${RESULT_PATH}"
echo ""

echo "DATA FORMAT:"
echo "  The results file contains per-request data with broadcast_results:"
echo "  {"
echo "    \"requests\": ["
echo "      {"
echo "        \"request_id\": \"1\","
echo "        \"prompt\": \"original prompt text\","
echo "        \"model\": \"Qwen/Qwen2.5-3B\",  // Primary response (randomly chosen)"
echo "        \"response\": \"generated text\","
echo "        \"ttft\": 0.5,"
echo "        \"e2el\": 2.0,"
echo "        \"output_len\": 100,"
echo "        \"broadcast_results\": [  // All model responses"
echo "          {\"model\": \"Qwen/Qwen2.5-3B\", \"ttft\": 0.5, \"e2el\": 2.0, \"output_len\": 100, \"generated_text\": \"...\", ...},"
echo "          {\"model\": \"Qwen/Qwen2.5-7B\", \"ttft\": 0.8, \"e2el\": 3.5, \"output_len\": 95, \"generated_text\": \"...\", ...},"
echo "          {\"model\": \"Qwen/Qwen2.5-14B\", \"ttft\": 1.2, \"e2el\": 5.0, \"output_len\": 105, \"generated_text\": \"...\", ...}"
echo "        ]"
echo "      },"
echo "      ..."
echo "    ]"
echo "  }"
echo ""

echo "TRAINING DATA USAGE:"
echo "  1. Model Quality Estimator Training:"
echo "     - Input: prompt + model_name"
echo "     - Output: predicted quality score (can derive from responses)"
echo "     - Extract from: requests[].broadcast_results[]"
echo ""
echo "  2. Response Length Predictor Training:"
echo "     - Input: prompt + model_name + num_prompt_tokens"
echo "     - Output: predicted output_len"
echo "     - Extract from: requests[].broadcast_results[].output_len"
echo ""

echo "QUICK INSPECTION:"
echo "  # View summary statistics"
echo "  cat ${RESULT_PATH} | jq '{completed, failed, mean_ttft_ms, mean_e2el_ms, mean_output_len}'"
echo ""
echo "  # Count requests with broadcast results"
echo "  cat ${RESULT_PATH} | jq '[.requests[] | select(.broadcast_results | length > 0)] | length'"
echo ""
echo "  # View first request with broadcast results"
echo "  cat ${RESULT_PATH} | jq '.requests[0] | {request_id, prompt, model, broadcast_results: [.broadcast_results[] | {model, ttft, e2el, output_len}]}'"
echo ""

echo "NEXT STEPS FOR TRAINING:"
echo "  1. Collect more data with different prompts:"
echo "     ./test_broadcasting.sh \"${BROADCAST_MODELS}\" sharegpt 500 2.0 batch2"
echo ""
echo "  2. Collect data with different model combinations:"
echo "     ./test_broadcasting.sh \"Qwen/Qwen2.5-3B Qwen/Qwen2.5-32B Qwen/Qwen2.5-72B\" sharegpt 200 inf large_models"
echo ""
echo "  3. Process collected data for training:"
echo "     python block/predictor/cara/process_training_data.py --input ${OUTPUT_DIR} --output training_data/"
echo ""

echo "CARA SERVER STATUS:"
echo "  PID: ${CARA_PID}"
echo "  Status: Running with broadcasting enabled"
echo ""
echo "  To stop: kill ${CARA_PID}"
echo "  To view logs: tail -f experiment_output/logs/cara_server_broadcast.log"
echo ""

echo "========================================================================"