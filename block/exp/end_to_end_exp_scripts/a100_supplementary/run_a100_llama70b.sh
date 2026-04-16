#!/bin/bash
# A100-40GB Llama-70B Experiment Runner
# This script deploys and runs Block prediction accuracy experiments on A100 cluster
#
# Usage: sh block/exp/end_to_end_exp_scripts/a100_supplementary/run_a100_llama70b.sh [enable_chunked_prefill]
#
# Parameters:
#   enable_chunked_prefill: "true" (default) or "false"
#
# Prerequisites:
#   1. Profiling data exists at data/profiling/compute/a100_40gb/meta-llama/Llama-2-70b-hf/
#   2. A100 hosts configured in block/config/a100_hosts
#   3. HuggingFace token with access to Llama-2-70B
#   4. NVMe mounted at /mydata on remote nodes

set -e

# Chunked prefill mode (pass as argument or default to true)
ENABLE_CHUNKED_PREFILL=${1:-true}

# Configuration - aligned with predictor config (llama70b_a100_40gb_config.json)
# These match the 7B model settings from llama_config.json for comparison
MODEL="meta-llama/Llama-2-70b-hf"
TENSOR_PARALLEL_SIZE=4       # matches replica_config.tensor_parallel_size
BATCH_CAP=48                 # matches replica_scheduler_config.batch_size_cap (same as 7B)
MAX_MODEL_LEN=4096
CHUNK_SIZE=512               # matches replica_scheduler_config.chunk_size (same as 7B)
NUM_PREDICTORS=4
PROFILING_SAMPLE_RATE=0.5
BACKEND_TIMEOUT=3600
PREDICTOR_TIMEOUT=2000
QPS_VALUES="2 4 6"
NUM_REQUESTS=500
DATASET="sharegpt"

# HuggingFace token
HF_TOKEN=""

# Paths
HOSTS_FILE="block/config/a100_hosts"
HOST_CONFIG="block/config/a100_host_configs.json"
PREDICTOR_CONFIG="block/config/llama70b_a100_40gb_config.json"
SCHEDULER_NAME="min_new_request_latency"

# Output prefix includes chunked prefill mode
if [ "$ENABLE_CHUNKED_PREFILL" = "true" ]; then
    OUTPUT_PREFIX="a100_40gb_llama70b_chunked"
else
    OUTPUT_PREFIX="a100_40gb_llama70b_no_chunked"
fi

# Get target host (first node in hosts file)
TARGET_HOST=$(head -1 $HOSTS_FILE)

echo "=============================================="
echo "A100-40GB Llama-70B Experiment"
echo "=============================================="
echo ""
echo "Configuration:"
echo "  Model: $MODEL"
echo "  Tensor Parallel Size: $TENSOR_PARALLEL_SIZE"
echo "  Batch Cap: $BATCH_CAP"
echo "  Chunk Size: $CHUNK_SIZE"
echo "  Chunked Prefill: $ENABLE_CHUNKED_PREFILL"
echo "  Target Host: $TARGET_HOST"
echo "  QPS values: $QPS_VALUES"
echo "  Num Requests: $NUM_REQUESTS"
echo "  Output Prefix: $OUTPUT_PREFIX"
echo ""

# Verify profiling data
PROFILE_PATH="data/profiling/compute/a100_40gb/meta-llama/Llama-2-70b-hf"
if [ ! -f "${PROFILE_PATH}/mlp.csv" ] || [ ! -f "${PROFILE_PATH}/attention.csv" ]; then
    echo "ERROR: Local profiling data not found at ${PROFILE_PATH}"
    exit 1
fi

# Check remote profiling data
echo "Verifying profiling data on remote nodes..."
parallel-ssh -h $HOSTS_FILE -i "ls ~/Block/data/profiling/compute/a100_40gb/meta-llama/Llama-2-70b-hf/*.csv"

echo ""
echo "Press Enter to start deployment or Ctrl+C to abort..."
read

# ============================================
# Phase 1: Reset and cleanup
# ============================================
echo ""
echo "=== Phase 1: Cleanup ==="
parallel-ssh -t 0 -h $HOSTS_FILE "pkill -f 'vllm.entrypoints' || true"
parallel-ssh -t 0 -h $HOSTS_FILE "pkill -f 'predictor/api_server' || true"
parallel-ssh -t 0 -h $HOSTS_FILE "pkill -f 'global_scheduler' || true"
sleep 10
echo "Cleanup complete."

# ============================================
# Phase 2: Start vLLM with tensor parallelism
# ============================================
echo ""
echo "=== Phase 2: Starting vLLM (this will take 3-5 minutes for 70B model) ==="
sh block/exp/run_exp_vllm_a100.sh $BATCH_CAP "$MODEL" $MAX_MODEL_LEN $CHUNK_SIZE $TENSOR_PARALLEL_SIZE "$HF_TOKEN" "$ENABLE_CHUNKED_PREFILL"

echo "Waiting for model to load (180 seconds)..."
sleep 180

# Verify vLLM is running
echo "Verifying vLLM..."
parallel-ssh -h $HOSTS_FILE -i "curl -s http://127.0.0.1:8000/health || echo 'vLLM not responding yet'"

echo ""
echo "Press Enter to continue to predictor deployment or Ctrl+C to abort..."
read

# ============================================
# Phase 3: Start predictors
# ============================================
echo ""
echo "=== Phase 3: Starting Predictors ==="
sh block/exp/run_exp_predictors_a100.sh "$PREDICTOR_CONFIG" "$SCHEDULER_NAME" $BATCH_CAP 1 $PREDICTOR_TIMEOUT "$ENABLE_CHUNKED_PREFILL"

echo "Waiting for predictors to initialize (60 seconds)..."
sleep 60

# Verify predictors
echo "Verifying predictors..."
parallel-ssh -h $HOSTS_FILE -i "curl -s http://127.0.0.1:8100/health || echo 'Predictor 1 not responding'"

echo ""
echo "Press Enter to continue to running experiments or Ctrl+C to abort..."
read

# ============================================
# Phase 4: Run experiments for each QPS
# ============================================
echo ""
echo "=== Phase 4: Running Experiments ==="

for qps in $QPS_VALUES; do
    echo ""
    echo "--- Running experiment with QPS=$qps ---"

    # Start global scheduler
    echo "Starting global scheduler..."
    sh block/exp/run_exp_global_scheduler_a100.sh "$TARGET_HOST" "$HOST_CONFIG" "$SCHEDULER_NAME" $NUM_PREDICTORS $PROFILING_SAMPLE_RATE $BACKEND_TIMEOUT $PREDICTOR_TIMEOUT
    sleep 15

    # Run benchmark
    OUTPUT_DIR="${OUTPUT_PREFIX}/${DATASET}/${SCHEDULER_NAME}/qps_${qps}"
    DATASET_PATH="~/Block/data/trace_data/${DATASET}/generate/llama"

    echo "Running benchmark..."
    ssh $TARGET_HOST "cd ~/Block && export PYTHONPATH=. && python block/benchmark/benchmark_serving.py \
        --ip_ports 127.0.0.1:8200 \
        --tokenizer $MODEL \
        --num_sampled_requests $NUM_REQUESTS \
        --dataset_type $DATASET \
        --dataset_path $DATASET_PATH \
        --qps $qps \
        --backend block \
        --log_filename benchmark.log \
        --output_dir $OUTPUT_DIR \
        --trust_remote_code \
        --max_request_len $MAX_MODEL_LEN \
        --timeout_in_seconds $BACKEND_TIMEOUT \
        --use_estimated_response_lens"

    # Save logs
    echo "Saving logs..."
    ssh $TARGET_HOST "cd ~/Block && mkdir -p experiment_output/$OUTPUT_DIR/running_logs && cp -r experiment_output/logs/* experiment_output/$OUTPUT_DIR/running_logs/ 2>/dev/null || true"

    # Kill global scheduler
    ssh $TARGET_HOST "pkill -f 'global_scheduler' || true"
    sleep 5

    echo "QPS=$qps experiment completed."
done

echo ""
echo "=============================================="
echo "All experiments completed!"
echo "=============================================="
echo ""
echo "Results on remote node: ~/Block/experiment_output/${OUTPUT_PREFIX}/"
echo ""
echo "To copy results locally:"
echo "  scp -r ${TARGET_HOST}:~/Block/experiment_output/${OUTPUT_PREFIX}/ experiment_output/"
