#!/bin/bash
# Llumnix-- verification at QPS=32
# Uses same infrastructure as po2_len_oracle but with min_lunmnix_load scheduler

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

# Llumnix-- scheduler, N=12 (queries all instances)
SCHEDULER_NAME="min_lunmnix_load"
N_SELECTED="12"
USE_LENGTH_ESTIMATION="false"

QPS="30 32"

PROFILING_SAMPLE_RATE=0.000
USE_FOR_PROFILING_ONLY=false
NUM_REQUEST=10000
KEEP_ALL_METRICS=false
OUTPUT_DIR_PREFIX="llumnix_verify"

AVAILABLE_INSTANCE="12"
ENABLE_PREEMPTIVE_AUTO_PROVISIONING="false"
MAX_SLO="0"

echo "=== Llumnix-- verification: QPS=$QPS ==="

for model in $MODEL; do
  if [ "$model" = "meta-llama/Llama-2-7b-hf" ]; then
    MODEL_TYPE="llama"
  elif [ "$model" = "Qwen/Qwen2-7B" ]; then
    MODEL_TYPE="qwen"
  fi

  for dataset_name in $DATASET_NAMES; do
    for qps in $QPS; do
      dataset_path="~/Block/data/trace_data/$dataset_name/generate/$MODEL_TYPE"
      echo "=== Llumnix-- QPS=$qps ==="

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
    done
  done
done

echo "=== Llumnix-- verification done ==="
