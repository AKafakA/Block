#!/bin/bash
# A30 vLLM deployment script
# Usage: sh block/exp/run_exp_vllm.sh <batch_cap> <model> <update_code> <vllm_version> <max_model_len> <enable_chunked_prefill> <num_workers> <max_num_batched_token> <use_process_for_frontend>
#
# IMPORTANT: VLLM_USE_V1 should be 0 for Block's get_scheduler_trace API
# See DEPLOYMENT_GUIDE.md for known issues and fixes

BATCH_CAP=$1
MODEL=$2
UPDATE_CODE=$3
VLLM_VERSION=$4  # Should be 0 for Block compatibility
MAX_MODEL_LENGTH=$5
ENABLE_CHUNKED_PREFILL=$6
NUM_WORKERS=$7
MAX_NUM_BATCHED_TOKEN=$8
USE_PROCESS_FOR_FRONTEND=$9
HUGGINGFACE_TOKEN=""

HOSTS_FILE="block/config/hosts"

echo "Starting vLLM on A30 cluster..."
echo "  Model: $MODEL"
echo "  Batch Cap: $BATCH_CAP"
echo "  Max Model Length: $MAX_MODEL_LENGTH"
echo "  Chunked Prefill: $ENABLE_CHUNKED_PREFILL"
echo "  Max Batched Tokens: $MAX_NUM_BATCHED_TOKEN"
echo "  VLLM_USE_V1: $VLLM_VERSION (should be 0 for Block)"

# Warn if VLLM_USE_V1 is not 0
if [ "$VLLM_VERSION" != "0" ]; then
    echo "WARNING: VLLM_USE_V1=$VLLM_VERSION. Block requires VLLM_USE_V1=0 for get_scheduler_trace API."
fi

if [ "$UPDATE_CODE" = "true" ]; then
    parallel-ssh -t 0 -h $HOSTS_FILE "cd vllm && sudo chown -R asdwb .git/ && git reset --hard HEAD~10 && git pull"
fi

# Build common environment variables
# NOTE: VLLM_USE_V1=0 is required for Block's get_scheduler_trace API
ENV_VARS="export LD_LIBRARY_PATH=\${LD_LIBRARY_PATH}:/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib:/usr/local/lib/python3.10/dist-packages/nvidia/nccl/lib:/usr/local/lib/python3.10/dist-packages/cusparselt/lib && export HF_TOKEN=${HUGGINGFACE_TOKEN} && export VLLM_USE_V1=${VLLM_VERSION}"

if [ "$USE_PROCESS_FOR_FRONTEND" = "false" ]; then
    parallel-ssh -t 0 -h $HOSTS_FILE "$ENV_VARS && cd vllm && mkdir -p ~/Block/experiment_output/logs && nohup python -m vllm.entrypoints.api_server --workers $NUM_WORKERS --model=$MODEL --max-num-seqs $BATCH_CAP --trust-remote-code --max_model_len $MAX_MODEL_LENGTH --enable_chunked_prefill $ENABLE_CHUNKED_PREFILL --max-num-batched-tokens $MAX_NUM_BATCHED_TOKEN --swap-space 0 --disable-frontend-multiprocessing > ~/Block/experiment_output/logs/vllm.log 2>&1 &"
else
    parallel-ssh -t 0 -h $HOSTS_FILE "$ENV_VARS && cd vllm && mkdir -p ~/Block/experiment_output/logs && nohup python -m vllm.entrypoints.api_server --workers $NUM_WORKERS --model=$MODEL --max-num-seqs $BATCH_CAP --trust-remote-code --max_model_len $MAX_MODEL_LENGTH --enable_chunked_prefill $ENABLE_CHUNKED_PREFILL --max-num-batched-tokens $MAX_NUM_BATCHED_TOKEN --swap-space 0 > ~/Block/experiment_output/logs/vllm.log 2>&1 &"
fi

echo "vLLM starting..."
