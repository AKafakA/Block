#!/bin/bash
# Po2 Generality Experiment — Binary Search for Capacity
# Purpose: Find capacity (TTFT P99 < 3s SLO) for Block Po2 across 4 variations.
#
# Variations:
#   A: batch_cap=24 (needs vLLM restart)
#   B: chunk_size=2048 (needs vLLM restart)
#   C: Qwen2-7B / ShareGPT (needs vLLM restart + model download)
#   D: Llama-7B / BurstGPT (same vLLM, different dataset)
#
# For each variation, runs both oracle and estimated lengths.
# Binary search: Phase 1 (QPS 20,28,36) → Phase 2 (binary) → Phase 3 (fine)
# ~10 runs per scheduler per variation = ~80 total runs
#
# Usage: ./po2_generality_experiment.sh [a|b|c|d|all]

set -e

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
ENABLE_CHUNKED_PREFILL="true"

SCHEDULER_NAME="min_new_request_latency"
N_SELECTED="2"

PROFILING_SAMPLE_RATE=0.000
USE_FOR_PROFILING_ONLY=false
NUM_REQUEST=10000
KEEP_ALL_METRICS=false

AVAILABLE_INSTANCE="12"
ENABLE_PREEMPTIVE_AUTO_PROVISIONING="false"
MAX_SLO="0"

run_single() {
  local model=$1
  local model_type=$2
  local dataset_name=$3
  local dataset_path=$4
  local qps=$5
  local batch_cap=$6
  local chunk_size=$7
  local use_estimation_len=$8
  local output_dir_prefix=$9
  local restart_vllm=${10}

  echo "  Running: model=$model dataset=$dataset_name QPS=$qps batch=$batch_cap chunk=$chunk_size est_len=$use_estimation_len restart=$restart_vllm"

  sh block/exp/experiment.sh \
    $SCHEDULER_NAME \
    $NUM_REQUEST \
    $restart_vllm \
    $batch_cap \
    $dataset_name \
    $dataset_path \
    $dataset_name \
    true \
    $KEEP_ALL_METRICS \
    $START_INDEX \
    $model \
    $model_type \
    $MAX_MODEL_LENGTH \
    $ENABLE_CHUNKED_PREFILL \
    $PREDICTOR_WORKERS \
    $GLOBAL_SCHEDULER_WORKERS \
    $BACKEND_WORKERS \
    $chunk_size \
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
    $use_estimation_len \
    $output_dir_prefix \
    $AVAILABLE_INSTANCE \
    $MAX_SLO \
    $ENABLE_PREEMPTIVE_AUTO_PROVISIONING
}

binary_search_capacity() {
  local model=$1
  local model_type=$2
  local dataset_name=$3
  local dataset_path=$4
  local batch_cap=$5
  local chunk_size=$6
  local use_estimation_len=$7
  local output_dir_prefix=$8
  local restart_vllm=$9

  local len_label="estimated"
  if [ "$use_estimation_len" = "false" ]; then
    len_label="oracle"
  fi

  echo ""
  echo "=== Binary Search: ${output_dir_prefix} (${len_label}) ==="

  # Phase 1: Coarse (QPS 20, 28, 36)
  echo "--- Phase 1: Coarse sweep ---"
  for qps in 20 28 36; do
    run_single "$model" "$model_type" "$dataset_name" "$dataset_path" \
      "$qps" "$batch_cap" "$chunk_size" "$use_estimation_len" "$output_dir_prefix" "$restart_vllm"
    restart_vllm=false
  done

  # Phase 2: Binary search
  echo "--- Phase 2: Binary search ---"
  for qps in 24 32; do
    run_single "$model" "$model_type" "$dataset_name" "$dataset_path" \
      "$qps" "$batch_cap" "$chunk_size" "$use_estimation_len" "$output_dir_prefix" false
  done

  # Phase 3: Fine-grained
  echo "--- Phase 3: Fine-grained ---"
  for qps in 26 30 34; do
    run_single "$model" "$model_type" "$dataset_name" "$dataset_path" \
      "$qps" "$batch_cap" "$chunk_size" "$use_estimation_len" "$output_dir_prefix" false
  done

  echo "=== Binary search complete for ${output_dir_prefix} (${len_label}) ==="
}

# Variation A: batch_cap=24
run_variation_a() {
  echo "######################################################################"
  echo "# VARIATION A: batch_cap=24 (Llama-7B / ShareGPT)"
  echo "######################################################################"
  local model="meta-llama/Llama-2-7b-hf"
  local model_type="llama"
  local dataset_path="~/Block/data/trace_data/sharegpt/generate/llama"

  binary_search_capacity "$model" "$model_type" "sharegpt" "$dataset_path" \
    24 512 "false" "generality/batch24_oracle" "true"
  binary_search_capacity "$model" "$model_type" "sharegpt" "$dataset_path" \
    24 512 "true" "generality/batch24_estimated" "false"
}

# Variation B: chunk_size=2048
run_variation_b() {
  echo "######################################################################"
  echo "# VARIATION B: chunk_size=2048 (Llama-7B / ShareGPT)"
  echo "######################################################################"
  local model="meta-llama/Llama-2-7b-hf"
  local model_type="llama"
  local dataset_path="~/Block/data/trace_data/sharegpt/generate/llama"

  binary_search_capacity "$model" "$model_type" "sharegpt" "$dataset_path" \
    48 2048 "false" "generality/chunk2048_oracle" "true"
  binary_search_capacity "$model" "$model_type" "sharegpt" "$dataset_path" \
    48 2048 "true" "generality/chunk2048_estimated" "false"
}

# Variation C: Qwen2-7B
run_variation_c() {
  echo "######################################################################"
  echo "# VARIATION C: Qwen2-7B / ShareGPT"
  echo "######################################################################"
  local model="Qwen/Qwen2-7B"
  local model_type="qwen"
  local dataset_path="~/Block/data/trace_data/sharegpt/generate/qwen"

  binary_search_capacity "$model" "$model_type" "sharegpt" "$dataset_path" \
    48 512 "false" "generality/qwen7b_oracle" "true"
  binary_search_capacity "$model" "$model_type" "sharegpt" "$dataset_path" \
    48 512 "true" "generality/qwen7b_estimated" "false"
}

# Variation D: BurstGPT
run_variation_d() {
  echo "######################################################################"
  echo "# VARIATION D: Llama-7B / BurstGPT"
  echo "######################################################################"
  local model="meta-llama/Llama-2-7b-hf"
  local model_type="llama"
  local dataset_path="~/Block/data/trace_data/burstgpt/generate/llama"

  binary_search_capacity "$model" "$model_type" "burstgpt" "$dataset_path" \
    48 512 "false" "generality/burstgpt_oracle" "true"
  binary_search_capacity "$model" "$model_type" "burstgpt" "$dataset_path" \
    48 512 "true" "generality/burstgpt_estimated" "false"
}

main() {
  local variation=${1:-all}
  echo "========================================================"
  echo "Po2 Generality Experiment — Binary Search for Capacity"
  echo "========================================================"

  case $variation in
    a|batch24) run_variation_a ;;
    b|chunk2048) run_variation_b ;;
    c|qwen) run_variation_c ;;
    d|burstgpt) run_variation_d ;;
    all)
      run_variation_a
      run_variation_b
      run_variation_c
      run_variation_d
      ;;
    *)
      echo "Usage: $0 [a|b|c|d|batch24|chunk2048|qwen|burstgpt|all]"
      exit 1
      ;;
  esac

  echo ""
  echo "========================================================"
  echo "Po2 Generality experiment completed!"
  echo "========================================================"
  echo "Results in: experiment_output/generality/"
}

main "$@"
