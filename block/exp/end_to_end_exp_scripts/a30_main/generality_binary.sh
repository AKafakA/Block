#!/bin/bash
# Phase 4 Generality with Binary Search
# Mirror of config_search_experiment.sh structure but with reduced QPS for binary search.
# Schedulers: Block-Po2 (oracle+estimated) + Llumnix-- (oracle only)
#
# Usage: bash generality_binary.sh [batch24|chunk2048|qwen|burstgpt|all]

VARIATION="${1:-all}"

# Common config (mirror main_experiment.sh)
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
RESTART_VLLM=true  # CRITICAL: must be true so predictors get redeployed with correct metric_type per scheduler

ENABLE_CHUNKED_PREFILL="true"
PROFILING_SAMPLE_RATE=0.000
USE_FOR_PROFILING_ONLY=false
NUM_REQUEST=10000
KEEP_ALL_METRICS=false
N_SELECTED="2"  # Po2 (new default)

AVAILABLE_INSTANCE="12"
ENABLE_PREEMPTIVE_AUTO_PROVISIONING="false"
MAX_SLO="0"
WARMUP="false"

# ============================================================================
# Run a single experiment (mirror of main_experiment.sh inner call)
# ============================================================================
run_one() {
  local scheduler=$1
  local model=$2
  local model_type=$3
  local dataset_name=$4
  local dataset_path=$5
  local qps=$6
  local batch_cap=$7
  local chunk_size=$8
  local use_estimation_len=$9
  local output_dir_prefix=${10}

  echo "Running: scheduler=$scheduler model=$model dataset=$dataset_name qps=$qps batch=$batch_cap chunk=$chunk_size est_len=$use_estimation_len"

  sh block/exp/experiment.sh \
    $scheduler $NUM_REQUEST $RESTART_VLLM $batch_cap \
    $dataset_name $dataset_path $dataset_name true $KEEP_ALL_METRICS $START_INDEX \
    $model $model_type $MAX_MODEL_LENGTH $ENABLE_CHUNKED_PREFILL \
    $PREDICTOR_WORKERS $GLOBAL_SCHEDULER_WORKERS $BACKEND_WORKERS $chunk_size \
    $qps $BRANCH_NAME $BATCH_SIZE_THRESHOLD_FOR_TIME_ESTIMATION $N_SELECTED \
    $PROFILING_SAMPLE_RATE $TIMEOUT_IN_SECONDS $USE_FOR_PROFILING_ONLY \
    $PREDICTOR_TIMEOUT_IN_SECONDS $USE_PROCESS_FOR_FRONTEND \
    $UPDATE_BLOCK_CODE $UPDATE_VLLM_CODE $RUN_EXP \
    $use_estimation_len $output_dir_prefix \
    $AVAILABLE_INSTANCE $MAX_SLO $ENABLE_PREEMPTIVE_AUTO_PROVISIONING
}

# ============================================================================
# Run a variation: all schedulers x all QPS
# ============================================================================
run_variation() {
  local var_name=$1
  local model=$2
  local model_type=$3
  local dataset_name=$4
  local dataset_path=$5
  local batch_cap=$6
  local chunk_size=$7
  local qps_list=$8
  local has_estimated=${9:-true}  # BurstGPT sets this to false

  echo "######################################################################"
  echo "# VARIATION: $var_name (model=$model dataset=$dataset_name batch=$batch_cap chunk=$chunk_size)"
  echo "# QPS levels: $qps_list"
  echo "# has_estimated: $has_estimated"
  echo "######################################################################"

  # Build scheduler list with length estimation modes
  # Block-Po2 oracle, Block-Po2 estimated, Llumnix-- oracle
  for qps in $qps_list; do
    # Block oracle
    run_one "min_new_request_latency" "$model" "$model_type" "$dataset_name" "$dataset_path" \
      "$qps" "$batch_cap" "$chunk_size" "false" "generality_${var_name}"

    # Block estimated (skip for BurstGPT)
    if [ "$has_estimated" = "true" ]; then
      run_one "min_new_request_latency" "$model" "$model_type" "$dataset_name" "$dataset_path" \
        "$qps" "$batch_cap" "$chunk_size" "true" "generality_${var_name}"
    fi

    # Llumnix-- baseline (oracle only — no length estimation)
    run_one "min_lunmnix_load" "$model" "$model_type" "$dataset_name" "$dataset_path" \
      "$qps" "$batch_cap" "$chunk_size" "false" "generality_${var_name}"
  done
}

# ============================================================================
# Variations
# ============================================================================
# Binary search QPS levels (8 levels per variation)

run_batch24() {
  run_variation "batch24" \
    "meta-llama/Llama-2-7b-hf" "llama" "sharegpt" \
    "~/Block/data/trace_data/sharegpt/generate/llama" \
    "24" "512" "20 23 26 28 30 32 34 36" "true"
}

run_chunk2048() {
  run_variation "chunk2048" \
    "meta-llama/Llama-2-7b-hf" "llama" "sharegpt" \
    "~/Block/data/trace_data/sharegpt/generate/llama" \
    "48" "2048" "20 23 26 28 30 32 34 36" "true"
}

run_qwen() {
  run_variation "qwen" \
    "Qwen/Qwen2-7B" "qwen" "sharegpt" \
    "~/Block/data/trace_data/sharegpt/generate/qwen" \
    "48" "512" "55 57 59 61 63 65 67 70" "true"
}

run_burstgpt() {
  # BurstGPT has no real prompts (trace with dummy tokens) → no length estimation
  run_variation "burstgpt" \
    "meta-llama/Llama-2-7b-hf" "llama" "burstgpt" \
    "~/Block/data/trace_data/burstgpt/generate/llama" \
    "48" "512" "48 50 53 55 57 59 61 64" "false"
}

# ============================================================================
# Main
# ============================================================================
case $VARIATION in
  batch24) run_batch24 ;;
  chunk2048) run_chunk2048 ;;
  qwen) run_qwen ;;
  burstgpt) run_burstgpt ;;
  all)
    run_batch24
    run_chunk2048
    run_qwen
    run_burstgpt
    ;;
  *)
    echo "Usage: $0 [batch24|chunk2048|qwen|burstgpt|all]"
    exit 1
    ;;
esac

echo ""
echo "=========================================="
echo "Generality experiment ($VARIATION) complete"
echo "=========================================="
