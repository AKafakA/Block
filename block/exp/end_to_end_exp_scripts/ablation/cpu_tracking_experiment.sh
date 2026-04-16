#!/bin/bash
# CPU Tracking Experiment Script
# Runs Block scheduler with CPU tracking enabled at QPS 20, 28, 36
# Collects results to SoCC_revision/cpu_tracker

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

# Config for CPU tracking experiment
ENABLE_CHUNKED_PREFILL="true"
MODEL="meta-llama/Llama-2-7b-hf"
DATASET_NAMES="sharegpt"
# Block (Po2, Po6, Fanout) + Llumnix-- + Random for CPU overhead comparison
SCHEDULER_NAME="min_new_request_latency min_lunmnix_load random"
# QPS to test
QPS="20 24 28 32"
PROFILING_SAMPLE_RATE=0.000
USE_FOR_PROFILING_ONLY=false
NUM_REQUEST=10000
KEEP_ALL_METRICS=false
# For Block scheduler: test N=2 (Po2), N=6 (medium), N=12 (Fanout)
# Baselines (Llumnix--, Random) auto-use N=12 per experiment.sh logic
N_SELECTED="2 6 12"
OUTPUT_DIR_PREFIX="cpu_tracking"

AVAILABLE_INSTANCE="12"
ENABLE_PREEMPTIVE_AUTO_PROVISIONING="false"
MAX_SLO="0"
WARMUP="false"

# Enable CPU tracking (39th parameter)
ENABLE_CPU_TRACKING="true"

# Target host for SCP
TARGET_HOST="asdwb@d7525-10s10309.wisc.cloudlab.us"
LOCAL_OUTPUT_DIR="SoCC_revision/cpu_tracker"

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
    for scheduler in $SCHEDULER_NAME; do
      if [ "$scheduler" = "min_new_request_latency" ]; then
        USE_LENGTH_ESTIMATION="true false"
      else
        USE_LENGTH_ESTIMATION="false"
      fi
      for enable_chunked_prefill in $ENABLE_CHUNKED_PREFILL; do
        for use_estimation_len in $USE_LENGTH_ESTIMATION; do
          for batch_size_cut in $BATCH_SIZE_THRESHOLD_FOR_TIME_ESTIMATION; do
            for n_selected in $N_SELECTED; do
              for qps in $QPS; do
                dataset_path="~/Block/data/trace_data/$dataset_name/generate/$MODEL_TYPE"
                echo "Running experiment with scheduler: $scheduler, qps: $qps, len_est: $use_estimation_len, cpu_tracking: $ENABLE_CPU_TRACKING"
                sh block/exp/experiment.sh $scheduler $NUM_REQUEST $RESTART_VLLM $BATCH_CAP $dataset_name $dataset_path $dataset_name true $KEEP_ALL_METRICS $START_INDEX $model $MODEL_TYPE $MAX_MODEL_LENGTH $enable_chunked_prefill $PREDICTOR_WORKERS $GLOBAL_SCHEDULER_WORKERS $BACKEND_WORKERS $CHUNK_SIZE $qps $BRANCH_NAME $batch_size_cut $n_selected $PROFILING_SAMPLE_RATE $TIMEOUT_IN_SECONDS $USE_FOR_PROFILING_ONLY $PREDICTOR_TIMEOUT_IN_SECONDS $USE_PROCESS_FOR_FRONTEND $UPDATE_BLOCK_CODE $UPDATE_VLLM_CODE $RUN_EXP $use_estimation_len $OUTPUT_DIR_PREFIX $AVAILABLE_INSTANCE $MAX_SLO $ENABLE_PREEMPTIVE_AUTO_PROVISIONING 0 0 1.0 $ENABLE_CPU_TRACKING
                # Only restart on first iteration
                RESTART_VLLM=false
              done
            done
          done
        done
      done
    done
  done
done

# Collect results
echo "=========================================="
echo "Collecting results to $LOCAL_OUTPUT_DIR..."
echo "=========================================="
mkdir -p $LOCAL_OUTPUT_DIR
scp -r $TARGET_HOST:~/Block/experiment_output/${OUTPUT_DIR_PREFIX}/* $LOCAL_OUTPUT_DIR/

echo "Done! Results saved to: $LOCAL_OUTPUT_DIR"