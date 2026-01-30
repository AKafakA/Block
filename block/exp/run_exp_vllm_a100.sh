#!/bin/bash
# A100 vLLM deployment script with tensor parallelism support
# Usage: sh block/exp/run_exp_vllm_a100.sh <batch_cap> <model> <max_model_len> <chunk_size> <tensor_parallel_size> <hf_token> [enable_chunked_prefill]
#
# Parameters:
#   enable_chunked_prefill: "true" (default) or "false"

BATCH_CAP=$1
MODEL=$2
MAX_MODEL_LENGTH=$3
MAX_NUM_BATCHED_TOKEN=$4
TENSOR_PARALLEL_SIZE=$5
HUGGINGFACE_TOKEN=$6
ENABLE_CHUNKED_PREFILL=${7:-true}

HOSTS_FILE="block/config/a100_hosts"

# Build chunked prefill argument
CHUNKED_PREFILL_ARG=""
if [ "$ENABLE_CHUNKED_PREFILL" = "true" ]; then
    CHUNKED_PREFILL_ARG="--enable-chunked-prefill --max-num-batched-tokens $MAX_NUM_BATCHED_TOKEN"
fi

echo "Starting vLLM on A100 cluster..."
echo "  Model: $MODEL"
echo "  Tensor Parallel Size: $TENSOR_PARALLEL_SIZE"
echo "  Max Model Length: $MAX_MODEL_LENGTH"
echo "  Batch Cap: $BATCH_CAP"
echo "  Chunked Prefill: $ENABLE_CHUNKED_PREFILL"
if [ "$ENABLE_CHUNKED_PREFILL" = "true" ]; then
    echo "  Chunk Size (max_num_batched_tokens): $MAX_NUM_BATCHED_TOKEN"
fi
echo "  Hosts: $(cat $HOSTS_FILE)"

# Kill any existing vLLM processes
parallel-ssh -t 0 -h $HOSTS_FILE "pkill -f 'vllm.entrypoints' || true"
sleep 5

# Start vLLM with tensor parallelism
# Note: VLLM_USE_V1=0 required for Block's get_scheduler_trace API
# Note: HF_HOME and download-dir set to /mydata for NVMe storage
# Note: PYTHONPATH includes ~/vllm for editable install to work with TP subprocesses
parallel-ssh -t 0 -h $HOSTS_FILE "cd ~/vllm && mkdir -p ~/Block/experiment_output/logs && \
    export HF_TOKEN=${HUGGINGFACE_TOKEN} && \
    export HF_HOME=/mydata/huggingface && \
    export VLLM_USE_V1=0 && \
    export PYTHONPATH=\$HOME/vllm:\$PYTHONPATH && \
    nohup python -m vllm.entrypoints.api_server \
    --model $MODEL \
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
    --max-num-seqs $BATCH_CAP \
    --max-model-len $MAX_MODEL_LENGTH \
    $CHUNKED_PREFILL_ARG \
    --swap-space 0 \
    --port 8000 \
    --trust-remote-code \
    --download-dir /mydata/huggingface \
    > ~/Block/experiment_output/logs/vllm.log 2>&1 &"

echo "vLLM starting... (this may take 3-5 minutes for 70B model)"
