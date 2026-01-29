#!/bin/bash
# Po2 Ablation Experiment
# Purpose: Test the tradeoff between prediction overhead and scheduling quality
# by varying N (number of instances probed) from 2 to 12.
#
# This addresses R12C's concern about scalability and demonstrates that Po2-style
# sampling can reduce overhead while maintaining scheduling quality.

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
RESTART_VLLM=true

# Config for Po2 ablation experiment
ENABLE_CHUNKED_PREFILL="true"
MODEL="meta-llama/Llama-2-7b-hf"
DATASET_NAMES="sharegpt"
# Only test Block scheduler with different N values
SCHEDULER_NAME="min_new_request_latency"
# Test at representative QPS levels
QPS="24 28 32"
PROFILING_SAMPLE_RATE=0.000
USE_FOR_PROFILING_ONLY=false
NUM_REQUEST=5000
KEEP_ALL_METRICS=false
# Key variable: test different N values (Po2 style)
N_SELECTED="2 4 6 12"
OUTPUT_DIR_PREFIX="po2_ablation"

AVAILABLE_INSTANCE="12"
ENABLE_PREEMPTIVE_AUTO_PROVISIONING="false"
MAX_SLO="0"

for model in $MODEL; do
  echo "Running warmup script for ${model} model"
  sh block/exp/end_to_end_exp_scripts/warmup.sh ${model} > /dev/null 2>&1
  if [ "$model" = "meta-llama/Llama-2-7b-hf" ]; then
    MODEL_TYPE="llama"
  elif [ "$model" = "Qwen/Qwen2-7B" ]; then
    MODEL_TYPE="qwen"
  fi
  for dataset_name in $DATASET_NAMES; do
    for scheduler in $SCHEDULER_NAME; do
      USE_LENGTH_ESTIMATION="true"
      for enable_chunked_prefill in $ENABLE_CHUNKED_PREFILL; do
        for use_estimation_len in $USE_LENGTH_ESTIMATION; do
          for batch_size_cut in $BATCH_SIZE_THRESHOLD_FOR_TIME_ESTIMATION; do
            for n_selected in $N_SELECTED; do
              for qps in $QPS; do
                dataset_path="~/Block/data/trace_data/$dataset_name/generate/$MODEL_TYPE"
                echo "=== Po2 Ablation: N=$n_selected, QPS=$qps ==="
                sh block/exp/experiment.sh $scheduler $NUM_REQUEST $RESTART_VLLM $BATCH_CAP $dataset_name $dataset_path $dataset_name true $KEEP_ALL_METRICS $START_INDEX $model $MODEL_TYPE $MAX_MODEL_LENGTH $enable_chunked_prefill $PREDICTOR_WORKERS $GLOBAL_SCHEDULER_WORKERS $BACKEND_WORKERS $CHUNK_SIZE $qps $BRANCH_NAME $batch_size_cut $n_selected $PROFILING_SAMPLE_RATE $TIMEOUT_IN_SECONDS $USE_FOR_PROFILING_ONLY $PREDICTOR_TIMEOUT_IN_SECONDS $USE_PROCESS_FOR_FRONTEND $UPDATE_BLOCK_CODE $UPDATE_VLLM_CODE $RUN_EXP $use_estimation_len $OUTPUT_DIR_PREFIX $AVAILABLE_INSTANCE $MAX_SLO $ENABLE_PREEMPTIVE_AUTO_PROVISIONING
                # Only restart vLLM for the first run
                RESTART_VLLM=false
              done
            done
          done
        done
      done
    done
  done
done

echo "Po2 ablation experiment completed!"
echo "Results in: experiment_output/${OUTPUT_DIR_PREFIX}/"
