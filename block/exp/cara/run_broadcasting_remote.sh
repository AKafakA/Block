#!/bin/bash

# Remote CARA Broadcasting Test Script for Training Data Collection
# ===================================================================
# This script runs the broadcasting test on a remote machine via SSH.
# Use this when running from a control machine (e.g., your local laptop).
#
# For running directly on the CARA server machine, use test_broadcasting.sh instead.
#
# Prerequisites:
#   - Backend instances (vLLM/Ollama) already deployed and running on remote hosts
#   - Model deployment config exists at block/config/cara/model_deployment.json
#   - SSH access to the target host
#
# Usage:
#   ./run_broadcasting_remote.sh TARGET_HOST [MODELS] [DATASET] [NUM_PROMPTS] [REQUEST_RATE] [OUTPUT_SUFFIX] [CUSTOM_DATASET_PATH]
#
# Examples:
#   # Collect data from all 4 models using v3 custom dataset (default)
#   ./run_broadcasting_remote.sh asdwb@d8545-10s10301.wisc.cloudlab.us \
#       "Qwen/Qwen2.5-3B Qwen/Qwen2.5-7B Qwen/Qwen2.5-14B Qwen/Qwen2.5-72B" \
#       custom 500 inf test1 data/cara/best-route-v3-test-500.jsonl
#
#   # Full 20k run with rate limiting
#   ./run_broadcasting_remote.sh asdwb@d8545-10s10301.wisc.cloudlab.us \
#       "Qwen/Qwen2.5-3B Qwen/Qwen2.5-7B Qwen/Qwen2.5-14B Qwen/Qwen2.5-72B" \
#       custom 20000 4.0 full_20k data/cara/best-route-v3.jsonl
#
# Environment variables for generation params (defaults shown):
#   FREQUENCY_PENALTY=1.2 TEMPERATURE=0.0 REPETITION_PENALTY=1.0
#   MAX_OUTPUT_TOKENS=1024 MAX_TOTAL_LEN=2048

set -e  # Exit on error

# =============================================================================
# Configuration
# =============================================================================

TARGET_HOST=${1:-"asdwb@d8545-10s10301.wisc.cloudlab.us"}
BROADCAST_MODELS=${2:-"Qwen/Qwen2.5-3B Qwen/Qwen2.5-7B Qwen/Qwen2.5-14B Qwen/Qwen2.5-72B"}
DATASET_NAME=${3:-"custom"}
NUM_PROMPTS=${4:-500}
REQUEST_RATE=${5:-inf}
OUTPUT_SUFFIX=${6:-"broadcast_data"}
CUSTOM_DATASET_PATH=${7:-"data/cara/best-route-v3-test-500.jsonl"}

# Remote paths
REMOTE_WORK_DIR="Block"

# Generation parameters (forwarded to test_broadcasting.sh via env vars)
FREQUENCY_PENALTY=${FREQUENCY_PENALTY:-1.2}
TEMPERATURE=${TEMPERATURE:-0.0}
REPETITION_PENALTY=${REPETITION_PENALTY:-1.0}
MAX_OUTPUT_TOKENS=${MAX_OUTPUT_TOKENS:-1024}
MAX_TOTAL_LEN=${MAX_TOTAL_LEN:-2048}

# SSH options
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"

# =============================================================================
# Display Configuration
# =============================================================================

echo "========================================================================"
echo "Remote CARA Broadcasting Test - Training Data Collection"
echo "========================================================================"
echo ""
echo "TARGET:"
echo "  Remote Host:          ${TARGET_HOST}"
echo "  Work Directory:       ${REMOTE_WORK_DIR}"
echo ""
echo "CONFIGURATION:"
echo "  Broadcast Models:     ${BROADCAST_MODELS}"
echo "  Dataset:              ${DATASET_NAME}"
if [ "${DATASET_NAME}" = "custom" ]; then
    echo "  Dataset Path:         ${CUSTOM_DATASET_PATH}"
fi
echo "  Number of Prompts:    ${NUM_PROMPTS}"
echo "  Request Rate:         ${REQUEST_RATE} qps"
echo "  Output Suffix:        ${OUTPUT_SUFFIX}"
echo "  Generation:           temp=${TEMPERATURE}, freq_penalty=${FREQUENCY_PENALTY}, rep_penalty=${REPETITION_PENALTY}"
echo "  Max Output/Total:     ${MAX_OUTPUT_TOKENS} / ${MAX_TOTAL_LEN}"
echo ""
echo "========================================================================"
echo ""

# =============================================================================
# Step 1: Verify Remote Environment
# =============================================================================

echo "Step 1: Verifying Remote Environment"
echo "-------------------------------------"

# Check if Block directory exists
if ! ssh ${SSH_OPTS} ${TARGET_HOST} "[ -d ${REMOTE_WORK_DIR} ]"; then
    echo "❌ Block directory not found on remote host: ${REMOTE_WORK_DIR}"
    exit 1
fi
echo "✅ Block directory found on remote host"

# Check if model deployment config exists
if ! ssh ${SSH_OPTS} ${TARGET_HOST} "[ -f ${REMOTE_WORK_DIR}/block/config/cara/model_deployment.json ]"; then
    echo "❌ Model deployment config not found on remote host"
    echo "Please deploy backends first using deploy_cara.py"
    exit 1
fi
echo "✅ Model deployment config found"

# Check if custom dataset exists on remote
if [ "${DATASET_NAME}" = "custom" ]; then
    if ! ssh ${SSH_OPTS} ${TARGET_HOST} "[ -f ${REMOTE_WORK_DIR}/${CUSTOM_DATASET_PATH} ]"; then
        echo "⚠️  Custom dataset not found on remote: ${REMOTE_WORK_DIR}/${CUSTOM_DATASET_PATH}"
        echo "Uploading from local..."
        scp ${SSH_OPTS} "${CUSTOM_DATASET_PATH}" "${TARGET_HOST}:${REMOTE_WORK_DIR}/${CUSTOM_DATASET_PATH}" || {
            echo "❌ Failed to upload dataset"
            exit 1
        }
        echo "✅ Dataset uploaded successfully"
    else
        echo "✅ Custom dataset found on remote"
    fi
fi

echo ""

# =============================================================================
# Step 2: Run Broadcasting Test on Remote Host
# =============================================================================

echo "Step 2: Running Broadcasting Test on Remote Host"
echo "-------------------------------------------------"
echo "Executing test_broadcasting.sh on ${TARGET_HOST}..."
echo ""

# Execute the test script remotely with generation params forwarded as env vars
ssh ${SSH_OPTS} ${TARGET_HOST} "cd ${REMOTE_WORK_DIR} && \
    FREQUENCY_PENALTY=${FREQUENCY_PENALTY} \
    TEMPERATURE=${TEMPERATURE} \
    REPETITION_PENALTY=${REPETITION_PENALTY} \
    MAX_OUTPUT_TOKENS=${MAX_OUTPUT_TOKENS} \
    MAX_TOTAL_LEN=${MAX_TOTAL_LEN} \
    bash block/exp/cara/test_broadcasting.sh \
    '${BROADCAST_MODELS}' \
    ${DATASET_NAME} \
    ${NUM_PROMPTS} \
    ${REQUEST_RATE} \
    ${OUTPUT_SUFFIX} \
    _ \
    ${CUSTOM_DATASET_PATH}"

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Remote broadcasting test failed!"
    echo ""
    echo "To debug, SSH to the remote host and check logs:"
    echo "  ssh ${TARGET_HOST}"
    echo "  cd ${REMOTE_WORK_DIR}"
    echo "  tail -f experiment_output/logs/cara_server_broadcast.log"
    exit 1
fi

echo ""
echo "========================================================================"
echo "✅ Remote Broadcasting Test Complete!"
echo "========================================================================"
echo ""

# =============================================================================
# Step 3: Display Results Location
# =============================================================================

echo "RESULTS LOCATION (on remote host):"
echo "  ${TARGET_HOST}:${REMOTE_WORK_DIR}/experiment_output/cara_broadcast_training_data/"
echo ""

echo "TO RETRIEVE RESULTS:"
echo "  # List all collected data files"
echo "  ssh ${TARGET_HOST} 'ls -lht ${REMOTE_WORK_DIR}/experiment_output/cara_broadcast_training_data/'"
echo ""
echo "  # Copy latest results to local machine"
echo "  scp ${TARGET_HOST}:${REMOTE_WORK_DIR}/experiment_output/cara_broadcast_training_data/broadcast_*.json ./"
echo ""
echo "  # Copy all results to local machine"
echo "  scp ${TARGET_HOST}:${REMOTE_WORK_DIR}/experiment_output/cara_broadcast_training_data/*.json ./training_data/"
echo ""

echo "TO VIEW RESULTS REMOTELY:"
echo "  # View summary statistics"
echo "  ssh ${TARGET_HOST} 'cd ${REMOTE_WORK_DIR} && cat experiment_output/cara_broadcast_training_data/broadcast_*.json | tail -1 | jq \"{completed, failed, mean_ttft_ms, mean_e2el_ms}\"'"
echo ""

echo "NEXT STEPS:"
echo "  1. Retrieve training data from remote host using scp commands above"
echo "  2. Collect more data with different configurations"
echo "  3. Process data for training model quality estimator and length predictor"
echo ""

echo "TO COLLECT MORE DATA:"
echo "  # Different model combinations"
echo "  ./run_broadcasting_remote.sh ${TARGET_HOST} \\"
echo "      \"Qwen/Qwen2.5-3B Qwen/Qwen2.5-32B Qwen/Qwen2.5-72B\" \\"
echo "      sharegpt 200 inf large_models"
echo ""
echo "  # More prompts with rate limiting"
echo "  ./run_broadcasting_remote.sh ${TARGET_HOST} \\"
echo "      \"${BROADCAST_MODELS}\" \\"
echo "      sharegpt 500 2.0 batch2"
echo ""

echo "CARA SERVER MANAGEMENT:"
echo "  # Check server logs"
echo "  ssh ${TARGET_HOST} 'tail -f ${REMOTE_WORK_DIR}/experiment_output/logs/cara_server_broadcast.log'"
echo ""
echo "  # Stop CARA server"
echo "  ssh ${TARGET_HOST} 'pkill -f cara_serve.py'"
echo ""

echo "========================================================================"