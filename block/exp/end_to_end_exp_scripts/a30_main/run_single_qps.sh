#!/bin/bash
# Single QPS run for manual binary search
# Usage:
#   sh run_single_qps.sh <QPS> [oracle|estimated] [output_prefix]
#
# Examples:
#   sh run_single_qps.sh 28 oracle po2_len_oracle    # Block N=2, oracle lengths
#   sh run_single_qps.sh 30.5 oracle po2_len_oracle  # Fine-grained capacity search
#   sh run_single_qps.sh 32 estimated po2_estimated   # Block N=2, estimated lengths
#   sh run_single_qps.sh 30 oracle generality/batch24_oracle  # Generality variation
#
# First run: set RESTART_VLLM=true below and run once to deploy vLLM+predictors.
# Subsequent runs: leave RESTART_VLLM=false (default) to reuse existing deployment.
#
# After each run, check TTFT P99 in the output to decide next QPS for binary search.

QPS=${1:?Usage: $0 <QPS> [oracle|estimated] [output_prefix] [dataset] [model] [batch_cap] [chunk_size]}
LEN_MODE=${2:-oracle}       # "oracle" or "estimated"
OUTPUT_DIR_PREFIX=${3:-po2_len_oracle}
DATASET_NAME=${4:-sharegpt}
MODEL=${5:-meta-llama/Llama-2-7b-hf}
BATCH_CAP=${6:-48}
CHUNK_SIZE=${7:-512}

# Derive model type and dataset path
if [ "$MODEL" = "Qwen/Qwen2-7B" ]; then
  MODEL_TYPE="qwen"
else
  MODEL_TYPE="llama"
fi
DATASET_PATH="~/Block/data/trace_data/$DATASET_NAME/generate/$MODEL_TYPE"

# Translate len mode
if [ "$LEN_MODE" = "oracle" ]; then
  USE_LENGTH_ESTIMATION="false"
else
  USE_LENGTH_ESTIMATION="true"
fi

# Common config
START_INDEX=0
PREDICTOR_WORKERS=16
GLOBAL_SCHEDULER_WORKERS=1
BACKEND_WORKERS=1
MAX_MODEL_LENGTH=4096
TIMEOUT_IN_SECONDS=1800
PREDICTOR_TIMEOUT_IN_SECONDS=1000
BATCH_SIZE_THRESHOLD_FOR_TIME_ESTIMATION="0"
BRANCH_NAME="main"
USE_PROCESS_FOR_FRONTEND=true
UPDATE_BLOCK_CODE=false
UPDATE_VLLM_CODE=false
RUN_EXP=true
RESTART_VLLM=false  # Set to true for first run only
ENABLE_CHUNKED_PREFILL="true"
N_SELECTED="2"
SCHEDULER_NAME="min_new_request_latency"
PROFILING_SAMPLE_RATE=0.000
USE_FOR_PROFILING_ONLY=false
NUM_REQUEST=10000
KEEP_ALL_METRICS=false
AVAILABLE_INSTANCE="12"
ENABLE_PREEMPTIVE_AUTO_PROVISIONING="false"
MAX_SLO="0"

echo "=== Single Run: QPS=$QPS, mode=$LEN_MODE, N=$N_SELECTED, output=$OUTPUT_DIR_PREFIX ==="
echo "    Model: $MODEL, Dataset: $DATASET_NAME, Batch: $BATCH_CAP, Chunk: $CHUNK_SIZE"

sh block/exp/experiment.sh \
  $SCHEDULER_NAME \
  $NUM_REQUEST \
  $RESTART_VLLM \
  $BATCH_CAP \
  $DATASET_NAME \
  $DATASET_PATH \
  $DATASET_NAME \
  true \
  $KEEP_ALL_METRICS \
  $START_INDEX \
  $MODEL \
  $MODEL_TYPE \
  $MAX_MODEL_LENGTH \
  $ENABLE_CHUNKED_PREFILL \
  $PREDICTOR_WORKERS \
  $GLOBAL_SCHEDULER_WORKERS \
  $BACKEND_WORKERS \
  $CHUNK_SIZE \
  $QPS \
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

echo ""
echo "=== Done: QPS=$QPS ==="
echo "Results: experiment_output/${OUTPUT_DIR_PREFIX}/${DATASET_NAME}/${SCHEDULER_NAME}/qps_${QPS}_*"
