#!/bin/bash
# Po2 Ablation Experiment
# Purpose: Test power-of-two-choices with N=2 (random sample 2 instances, pick best)
# Compare with main_experiment baselines (Block N=12 and Llumnix--)
#
# This addresses R12C's concern about scalability - shows that even N=2
# achieves near-optimal scheduling with minimal overhead.

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
RESTART_VLLM=false  # Set to true for fresh deployment
WARMUP=false

# Config matching main_experiment
ENABLE_CHUNKED_PREFILL="true"
MODEL="meta-llama/Llama-2-7b-hf"
DATASET_NAMES="sharegpt"

# Only Block scheduler with N=2
SCHEDULER_NAME="min_new_request_latency"
N_SELECTED="2"  # Power-of-two: randomly sample 2 instances

# Same QPS range as main_experiment
QPS="20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36"

PROFILING_SAMPLE_RATE=0.000
USE_FOR_PROFILING_ONLY=false
NUM_REQUEST=10000  # Match main_experiment
KEEP_ALL_METRICS=false
OUTPUT_DIR_PREFIX="po2"

AVAILABLE_INSTANCE="12"
ENABLE_PREEMPTIVE_AUTO_PROVISIONING="false"
MAX_SLO="0"

# No error injection
LENGTH_ERROR_PCT=0
LATENCY_ERROR_PCT=0

# Count total runs
num_qps=$(echo $QPS | wc -w)
echo "=== Po2 Ablation: N=2, ${num_qps} QPS levels ==="
echo "Results will be compared with main_experiment baselines"

for model in $MODEL; do
  if [ "$WARMUP" = "true" ] && [ "$RESTART_VLLM" = "true" ]; then
    echo "Running warmup script for ${model} model"
    sh block/exp/end_to_end_exp_scripts/a30_main/warmup.sh ${model} > /dev/null 2>&1
  fi
  if [ "$model" = "meta-llama/Llama-2-7b-hf" ]; then
    MODEL_TYPE="llama"
  elif [ "$model" = "Qwen/Qwen2-7B" ]; then
    MODEL_TYPE="qwen"
  fi

  for dataset_name in $DATASET_NAMES; do
    dataset_path="~/Block/data/trace_data/$dataset_name/generate/$MODEL_TYPE"

    for qps in $QPS; do
      echo "=== Po2 (N=2): QPS=$qps ==="

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
        true \
        $OUTPUT_DIR_PREFIX \
        $AVAILABLE_INSTANCE \
        $MAX_SLO \
        $ENABLE_PREEMPTIVE_AUTO_PROVISIONING \
        $LENGTH_ERROR_PCT \
        $LATENCY_ERROR_PCT

      # Only restart vLLM for the first run
      RESTART_VLLM=false
    done
  done
done

echo ""
echo "==========================================="
echo "Po2 ablation experiment completed!"
echo "==========================================="
echo "Results in: experiment_output/${OUTPUT_DIR_PREFIX}/"
echo ""
echo "Compare with main_experiment baselines:"
echo "  - Block (N=12): experiment_output/data/main/sharegpt/min_new_request_latency/"
echo "  - Llumnix--:    experiment_output/data/main/sharegpt/min_lunmnix_load/"