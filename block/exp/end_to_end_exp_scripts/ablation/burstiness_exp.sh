#!/bin/bash
# Burstiness Experiment
# Purpose: Test Block's performance under bursty workloads with gamma distribution.
#
# This addresses R12C's concern: "There is no study with bursty workloads"
# Tests burstiness factors: 0.25, 0.5, 1.0, 2.0
#   - Lower values (0.25, 0.5) = more bursty arrivals
#   - 1.0 = Poisson arrivals (baseline)
#   - Higher values (2.0) = more regular arrivals
#
# Compares Block vs Llumnix-- (min_lunmnix_load)
# Uses experiment.sh for consistency with other experiments.

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

# Config for burstiness experiment
ENABLE_CHUNKED_PREFILL="true"
MODEL="meta-llama/Llama-2-7b-hf"
DATASET_NAMES="sharegpt"

# Schedulers to compare: Block and Llumnix--
SCHEDULER_NAMES="min_new_request_latency min_lunmnix_load"

# Use QPS=30 to match main experiments
QPS="30"
PROFILING_SAMPLE_RATE=0.000
USE_FOR_PROFILING_ONLY=false
NUM_REQUEST=10000
KEEP_ALL_METRICS=false
N_SELECTED="12"
OUTPUT_DIR_PREFIX="burstiness"

# Burstiness levels: lower = more bursty (gamma distribution shape parameter)
# 0.25, 0.5 = bursty; 1.0 = Poisson; 2.0 = regular
BURSTINESS_LEVELS="0.25 0.5 1.0 2.0"

AVAILABLE_INSTANCE="12"
ENABLE_PREEMPTIVE_AUTO_PROVISIONING="false"
MAX_SLO="0"

# No error injection for burstiness study
LENGTH_ERROR_PCT=0
LATENCY_ERROR_PCT=0

# Count total runs
num_schedulers=$(echo $SCHEDULER_NAMES | wc -w)
num_burstiness=$(echo $BURSTINESS_LEVELS | wc -w)
total_runs=$((num_schedulers * num_burstiness))
run_count=0

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

    for scheduler in $SCHEDULER_NAMES; do
      # Determine if we use length estimation (only matters for Block)
      if [ "$scheduler" = "min_new_request_latency" ]; then
        USE_ESTIMATION_LEN="true"
        SCHEDULER_DISPLAY="Block"
      else
        USE_ESTIMATION_LEN="false"  # Llumnix-- doesn't use length estimation for scheduling
        SCHEDULER_DISPLAY="Llumnix--"
      fi

      for burstiness in $BURSTINESS_LEVELS; do
        run_count=$((run_count + 1))
        echo "=== Burstiness [$run_count/$total_runs]: ${SCHEDULER_DISPLAY} (${scheduler}), burstiness=${burstiness} ==="

        for qps in $QPS; do
          # Create output dir with burstiness in the name
          OUTPUT_DIR_WITH_BURSTINESS="${OUTPUT_DIR_PREFIX}/burstiness_${burstiness}"

          echo "Running experiment with scheduler: $scheduler, model: $model, dataset: $dataset_name, qps: $qps"
          echo "  Burstiness: ${burstiness} (gamma distribution shape parameter)"

          # Call experiment.sh with burstiness parameter (position 38)
          sh block/exp/experiment.sh \
            $scheduler \
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
            $OUTPUT_DIR_WITH_BURSTINESS \
            $AVAILABLE_INSTANCE \
            $MAX_SLO \
            $ENABLE_PREEMPTIVE_AUTO_PROVISIONING \
            $LENGTH_ERROR_PCT \
            $LATENCY_ERROR_PCT \
            $burstiness

          # Only restart vLLM once for the first run
          RESTART_VLLM=false
        done
      done
    done
  done
done

echo ""
echo "=========================================="
echo "Burstiness experiment completed!"
echo "=========================================="
echo "Results in: experiment_output/${OUTPUT_DIR_PREFIX}/"
echo "Total runs: $run_count"
echo ""
echo "Results structure:"
echo "  burstiness_0.25/sharegpt/min_new_request_latency/  (Block, most bursty)"
echo "  burstiness_0.25/sharegpt/min_lunmnix_load/         (Llumnix--, most bursty)"
echo "  burstiness_0.5/sharegpt/...                        (moderately bursty)"
echo "  burstiness_1.0/sharegpt/...                        (Poisson baseline)"
echo "  burstiness_2.0/sharegpt/...                        (regular arrivals)"
