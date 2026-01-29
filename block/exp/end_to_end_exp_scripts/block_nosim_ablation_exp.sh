#!/bin/bash
# Block-NoSim Ablation Experiment
# Purpose: Isolate the value of length prediction vs full simulation.
#
# This addresses R12C's concern about needing a "stronger baseline that uses
# richer but cheap signals without full simulation."
#
# Compares:
# - Block (length prediction + simulation)
# - Block-NoSim (length prediction, no simulation - min total unprocessed tokens)
# - Llumnix- (no length prediction, no simulation)
# - INFaaS++ (no length prediction, no simulation)

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

# Config for Block-NoSim ablation experiment
ENABLE_CHUNKED_PREFILL="true"
MODEL="meta-llama/Llama-2-7b-hf"
DATASET_NAMES="sharegpt"
# Compare Block, Block-NoSim, Llumnix-, INFaaS++
SCHEDULER_NAME="min_new_request_latency min_total_unprocessed_tokens min_lunmnix_load min_infass_load"
# Test across QPS range
QPS="24 28 32"
PROFILING_SAMPLE_RATE=0.000
USE_FOR_PROFILING_ONLY=false
NUM_REQUEST=5000
KEEP_ALL_METRICS=false
N_SELECTED="12"
OUTPUT_DIR_PREFIX="block_nosim_ablation"

AVAILABLE_INSTANCE="12"
ENABLE_PREEMPTIVE_AUTO_PROVISIONING="false"
MAX_SLO="0"

TARGET_HOST=""  # Fill with your global scheduler host

for model in $MODEL; do
  echo "Running warmup script for ${model} model"
  sh block/exp/end_to_end_exp_scripts/warmup.sh ${model} > /dev/null 2>&1
  if [ "$model" = "meta-llama/Llama-2-7b-hf" ]; then
    MODEL_TYPE="llama"
  elif [ "$model" = "Qwen/Qwen2-7B" ]; then
    MODEL_TYPE="qwen"
  fi

  HOST_CONFIG_PATH='block/config/host_configs.json'
  PREDICTOR_CONFIG_PATH="block/config/${MODEL_TYPE}_config.json"

  for dataset_name in $DATASET_NAMES; do
    dataset_path="~/Block/data/trace_data/$dataset_name/generate/$MODEL_TYPE"

    for scheduler in $SCHEDULER_NAME; do
      echo "=== Testing scheduler: $scheduler ==="

      # Determine if this scheduler uses length estimation
      if [ "$scheduler" = "min_new_request_latency" ] || [ "$scheduler" = "min_total_unprocessed_tokens" ]; then
        USE_LENGTH_ESTIMATION="true"
      else
        USE_LENGTH_ESTIMATION="false"
      fi

      # Restart vLLM and predictors for each scheduler (different metric types)
      echo "Setting up vLLM and Predictors for $scheduler..."
      sh block/exp/reset.sh
      sleep 30
      nohup sh block/exp/run_exp_vllm.sh $BATCH_CAP $model false 0 $MAX_MODEL_LENGTH true $BACKEND_WORKERS $CHUNK_SIZE > /dev/null 2>&1 &
      sleep 60
      for suffix in $(seq 1 $PREDICTOR_WORKERS); do
        nohup sh block/exp/run_exp_predictor_${suffix}.sh $PREDICTOR_CONFIG_PATH $scheduler true $BATCH_CAP true $PREDICTOR_WORKERS $BRANCH_NAME $BATCH_SIZE_THRESHOLD_FOR_TIME_ESTIMATION $PREDICTOR_TIMEOUT_IN_SECONDS > /dev/null 2>&1 &
      done
      sleep 60

      for qps in $QPS; do
        echo "=== Ablation: scheduler=$scheduler, QPS=$qps ==="

        # Start global scheduler
        nohup sh block/exp/run_exp_global_scheduler.sh $TARGET_HOST $N_SELECTED $N_SELECTED $scheduler $HOST_CONFIG_PATH $GLOBAL_SCHEDULER_WORKERS $PREDICTOR_WORKERS $PROFILING_SAMPLE_RATE $TIMEOUT_IN_SECONDS $PREDICTOR_TIMEOUT_IN_SECONDS $AVAILABLE_INSTANCE $MAX_SLO false > /dev/null 2>&1 &
        sleep 10

        # Run benchmark
        OUTPUT_DIR="${OUTPUT_DIR_PREFIX}/${dataset_name}/${scheduler}/qps_${qps}"

        if [ "$USE_LENGTH_ESTIMATION" = "true" ]; then
          parallel-ssh -i -t 0 --host $TARGET_HOST "cd Block && export PYTHONPATH=. && python block/benchmark/benchmark_serving.py --ip_ports 127.0.0.1:8200 --tokenizer $model --num_sampled_requests $NUM_REQUEST --dataset_type $dataset_name --dataset_path $dataset_path --qps $qps --backend block --log_filename benchmark.log --output_dir $OUTPUT_DIR --data_start_index $START_INDEX --trust_remote_code --max_request_len $MAX_MODEL_LENGTH --timeout_in_seconds $TIMEOUT_IN_SECONDS --use_estimated_response_lens"
        else
          parallel-ssh -i -t 0 --host $TARGET_HOST "cd Block && export PYTHONPATH=. && python block/benchmark/benchmark_serving.py --ip_ports 127.0.0.1:8200 --tokenizer $model --num_sampled_requests $NUM_REQUEST --dataset_type $dataset_name --dataset_path $dataset_path --qps $qps --backend block --log_filename benchmark.log --output_dir $OUTPUT_DIR --data_start_index $START_INDEX --trust_remote_code --max_request_len $MAX_MODEL_LENGTH --timeout_in_seconds $TIMEOUT_IN_SECONDS"
        fi

        sleep 10
        parallel-ssh --host $TARGET_HOST "cd Block && mkdir -p experiment_output/$OUTPUT_DIR/running_logs"
        parallel-ssh --host $TARGET_HOST "cd Block && mv experiment_output/logs/* experiment_output/$OUTPUT_DIR/running_logs/."

        # Kill global scheduler before next iteration
        parallel-ssh -t 0 --host $TARGET_HOST "pkill -f global_scheduler"
        sleep 5
      done
    done
  done
done

echo "Block-NoSim ablation experiment completed!"
echo "Results in: experiment_output/${OUTPUT_DIR_PREFIX}/"
echo ""
echo "Expected ablation insights:"
echo "  - Block-NoSim vs Llumnix-: Value of length prediction alone"
echo "  - Block vs Block-NoSim: Value of simulation (beyond token counting)"
