#!/bin/bash
# Block-Len-Oracle (Po2) — Full QPS Sweep
# Purpose: Run Block with N=2 using oracle (real) completion lengths.
# USE_LENGTH_ESTIMATION=false means the scheduler uses real completion_len from trace.

START_INDEX=0
BATCH_CAP=48
PREDICTOR_WORKERS=16
GLOBAL_SCHEDULER_WORKERS=1
BACKEND_WORKERS=1
MAX_MODEL_LENGTH=4096
CHUNK_SIZE=512
TIMEOUT_IN_SECONDS=1800
PREDICTOR_TIMEOUT_IN_SECONDS=1000
BATCH_SIZE_THRESHOLD_FOR_TIME_ESTIMATION="0"
BRANCH_NAME="main"
USE_PROCESS_FOR_FRONTEND=true
UPDATE_BLOCK_CODE=false
UPDATE_VLLM_CODE=false
RUN_EXP=true
RESTART_VLLM=false
WARMUP=false

ENABLE_CHUNKED_PREFILL="true"
MODEL="meta-llama/Llama-2-7b-hf"
DATASET_NAMES="sharegpt"

# Block scheduler only, N=2 (power-of-two), oracle lengths
SCHEDULER_NAME="min_new_request_latency"
N_SELECTED="2"
USE_LENGTH_ESTIMATION="false"  # Oracle = real completion lengths

# Full QPS sweep (17 levels) — adjust range after binary search to save time
# For binary search: use run_single_qps.sh instead
QPS="20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36"

PROFILING_SAMPLE_RATE=0.000
USE_FOR_PROFILING_ONLY=false
NUM_REQUEST=10000
KEEP_ALL_METRICS=false
OUTPUT_DIR_PREFIX="po2_len_oracle"

AVAILABLE_INSTANCE="12"
ENABLE_PREEMPTIVE_AUTO_PROVISIONING="false"
MAX_SLO="0"

num_qps=$(echo $QPS | wc -w)
echo "=== Block-Len-Oracle (Po2 N=2, Oracle lengths): ${num_qps} QPS levels ==="

for model in $MODEL; do
  if [ "$WARMUP" = "true" ] && [ "$RESTART_VLLM" = "true" ]; then
    echo "Running warmup script for ${model} model"
    sh block/exp/end_to_end_exp_scripts/warmup.sh ${model} > /dev/null 2>&1
  fi
  if [ "$model" = "meta-llama/Llama-2-7b-hf" ]; then
    MODEL_TYPE="llama"
  elif [ "$model" = "Qwen/Qwen2-7B" ]; then
    MODEL_TYPE="qwen"
  fi

  for dataset_name in $DATASET_NAMES; do
    for qps in $QPS; do
      dataset_path="~/Block/data/trace_data/$dataset_name/generate/$MODEL_TYPE"
      echo "=== Block-Len-Oracle (N=2, oracle): QPS=$qps ==="

      sh block/exp/experiment.sh \
        $SCHEDULER_NAME \
        $NUM_REQUEST \
        $RESTART_VLLM \
        $BATCH_CAP \
        $dataset_name \
        $dataset_path \
        $dataset_name \
        true \
        $KEEP_ALL_METRICS \
        $START_INDEX \
        $model \
        $MODEL_TYPE \
        $MAX_MODEL_LENGTH \
        $ENABLE_CHUNKED_PREFILL \
        $PREDICTOR_WORKERS \
        $GLOBAL_SCHEDULER_WORKERS \
        $BACKEND_WORKERS \
        $CHUNK_SIZE \
        $qps \
        $BRANCH_NAME \
        $BATCH_SIZE_THRESHOLD_FOR_TIME_ESTIMATION \
        $N_SELECTED \
        $PROFILING_SAMPLE_RATE \
        $TIMEOUT_IN_SECONDS \
        $USE_FOR_PROFILING_ONLY \
        $PREDICTOR_TIMEOUT_IN_SECONDS \
        $USE_PROCESS_FOR_FRONTEND \
        $UPDATE_BLOCK_CODE \
        $UPDATE_VLLM_CODE \
        $RUN_EXP \
        $USE_LENGTH_ESTIMATION \
        $OUTPUT_DIR_PREFIX \
        $AVAILABLE_INSTANCE \
        $MAX_SLO \
        $ENABLE_PREEMPTIVE_AUTO_PROVISIONING

      RESTART_VLLM=false
    done
  done
done

echo ""
echo "==========================================="
echo "Block-Len-Oracle (Po2) experiment completed!"
echo "==========================================="
echo "Results in: experiment_output/${OUTPUT_DIR_PREFIX}/"
echo "Capacity should be >= Block-Po2 (30.2 QPS) since oracle lengths are more accurate."
