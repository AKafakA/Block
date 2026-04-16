#!/bin/bash
# Error Sensitivity Heatmap Experiment
# Purpose: Test Block's robustness to prediction errors by injecting controlled noise.
#
# This addresses R12A's concern: "How sensitive is the system to prediction error?"
# Creates a grid: length_error × latency_error (0%, 25%, 50%, 100%)
#
# Uses experiment.sh (proven working) for consistency with main_experiment.sh
#
# Also runs baseline (Llumnix--) for comparison in the heatmap visualization.

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
RESTART_VLLM=false  # Set to true for fresh deployment; false to reuse existing
WARMUP=false  # Set to true to run warmup script (downloads model, clears cache)

# Config for error heatmap experiment
ENABLE_CHUNKED_PREFILL="true"
MODEL="meta-llama/Llama-2-7b-hf"
DATASET_NAMES="sharegpt"
SCHEDULER_NAME="min_new_request_latency"
# Use QPS=30 to match main experiments
QPS="30"
PROFILING_SAMPLE_RATE=0.000
USE_FOR_PROFILING_ONLY=false
NUM_REQUEST=10000
KEEP_ALL_METRICS=false
N_SELECTED="2"
OUTPUT_DIR_PREFIX="error_heatmap_po2"
USE_ESTIMATION_LEN="true"

# Baseline configuration (Llumnix-- for comparison)
RUN_BASELINE=true  # Set to true to also run baseline for comparison
BASELINE_SCHEDULER="min_lunmnix_load"  # Llumnix-- scheduler

# Error levels for heatmap (percentage)
# Full grid experiment: 4x4 = 16 runs
LENGTH_ERROR_LEVELS="0 25 50 100"
LATENCY_ERROR_LEVELS="0 25 50 100"

AVAILABLE_INSTANCE="12"
ENABLE_PREEMPTIVE_AUTO_PROVISIONING="false"
MAX_SLO="0"

run_count=0
# Full grid: 4x4 = 16 runs + 1 baseline = 17 runs (if baseline enabled)
total_runs=16
if [ "$RUN_BASELINE" = "true" ]; then
  total_runs=$((total_runs + 1))
fi

for model in $MODEL; do
  if [ "$WARMUP" = "true" ] && [ "$RESTART_VLLM" = "true" ]; then
    echo "Running warmup script for ${model} model to download the model weights and cache them"
    sh block/exp/end_to_end_exp_scripts/a30_main/warmup.sh ${model} > /dev/null 2>&1
  fi
  if [ "$model" = "meta-llama/Llama-2-7b-hf" ]; then
    MODEL_TYPE="llama"
  elif [ "$model" = "Qwen/Qwen2-7B" ]; then
    MODEL_TYPE="qwen"
  fi

  for dataset_name in $DATASET_NAMES; do
    dataset_path="~/Block/data/trace_data/$dataset_name/generate/$MODEL_TYPE"

    # Run baseline first (Llumnix-- with 0% error) for comparison
    if [ "$RUN_BASELINE" = "true" ]; then
      run_count=$((run_count + 1))
      echo "=== Baseline [$run_count/$total_runs]: ${BASELINE_SCHEDULER} (Llumnix--) ==="

      for qps in $QPS; do
        # Baseline goes in len_err_0_lat_err_0 directory alongside Block results
        OUTPUT_DIR_BASELINE="${OUTPUT_DIR_PREFIX}/len_err_0_lat_err_0"

        echo "Running baseline experiment with scheduler: $BASELINE_SCHEDULER, model: $model, dataset: $dataset_name, qps: $qps"

        # Call experiment.sh with baseline scheduler (no error injection)
        sh block/exp/experiment.sh \
          $BASELINE_SCHEDULER \
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
          $USE_ESTIMATION_LEN \
          $OUTPUT_DIR_BASELINE \
          $AVAILABLE_INSTANCE \
          $MAX_SLO \
          $ENABLE_PREEMPTIVE_AUTO_PROVISIONING \
          0 \
          0

        # Don't restart vLLM for subsequent runs
        RESTART_VLLM=false
      done
    fi

    # Run Block with various error levels
    for length_error in $LENGTH_ERROR_LEVELS; do
      for latency_error in $LATENCY_ERROR_LEVELS; do
        run_count=$((run_count + 1))
        echo "=== Error Heatmap [$run_count/$total_runs]: length_error=${length_error}%, latency_error=${latency_error}% ==="

        for qps in $QPS; do
          # Create output dir with error levels in the name
          OUTPUT_DIR_WITH_ERROR="${OUTPUT_DIR_PREFIX}/len_err_${length_error}_lat_err_${latency_error}"

          echo "Running experiment with scheduler: $SCHEDULER_NAME, model: $model, dataset: $dataset_name, qps: $qps"
          echo "  Error injection: length_error=${length_error}%, latency_error=${latency_error}%"

          # Call experiment.sh with error injection parameters (36 and 37)
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
            $USE_ESTIMATION_LEN \
            $OUTPUT_DIR_WITH_ERROR \
            $AVAILABLE_INSTANCE \
            $MAX_SLO \
            $ENABLE_PREEMPTIVE_AUTO_PROVISIONING \
            $length_error \
            $latency_error

          # Only restart vLLM once for the first run
          RESTART_VLLM=false
        done
      done
    done
  done
done

echo ""
echo "=========================================="
echo "Error heatmap experiment completed!"
echo "=========================================="
echo "Results in: experiment_output/${OUTPUT_DIR_PREFIX}/"
echo "Total runs: $run_count"
if [ "$RUN_BASELINE" = "true" ]; then
  echo "Baseline (${BASELINE_SCHEDULER}) results in: experiment_output/${OUTPUT_DIR_PREFIX}/len_err_0_lat_err_0/${DATASET_NAMES}/${BASELINE_SCHEDULER}/"
fi
echo ""
echo "To generate heatmap plots, run:"
echo "  cd SoCC_revision/prediction_error && python plot_heatmap.py"