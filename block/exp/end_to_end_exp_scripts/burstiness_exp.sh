#!/bin/bash
# Burstiness Experiment
# Purpose: Test Block's performance under bursty workloads with gamma distribution.
#
# This addresses R12C's concern: "There is no study with bursty workloads"
# Tests burstiness factors 0.25, 0.5, 0.75 (lower = more bursty)
# Compares Block vs Llumnix- (strongest baseline)

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

# Config for burstiness experiment
ENABLE_CHUNKED_PREFILL="true"
MODEL="meta-llama/Llama-2-7b-hf"
DATASET_NAMES="sharegpt"
# Compare Block vs Llumnix- under bursty conditions
SCHEDULER_NAME="min_new_request_latency min_lunmnix_load"
# Test at moderate QPS levels where burstiness matters
QPS="24 28 32"
PROFILING_SAMPLE_RATE=0.000
USE_FOR_PROFILING_ONLY=false
NUM_REQUEST=5000
KEEP_ALL_METRICS=false
N_SELECTED="12"
OUTPUT_DIR_PREFIX="burstiness"

# Burstiness levels: lower = more bursty (gamma distribution shape parameter)
BURSTINESS_LEVELS="0.25 0.5 0.75 1.0"

AVAILABLE_INSTANCE="12"
ENABLE_PREEMPTIVE_AUTO_PROVISIONING="false"
MAX_SLO="0"

# Note: Need to modify experiment.sh to pass burstiness to benchmark_serving.py
# For now, we'll call benchmark directly with burstiness parameter

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
      if [ "$scheduler" = "min_new_request_latency" ]; then
        USE_LENGTH_ESTIMATION="true"
      else
        USE_LENGTH_ESTIMATION="false"
      fi

      for burstiness in $BURSTINESS_LEVELS; do
        for qps in $QPS; do
          echo "=== Burstiness Experiment: scheduler=$scheduler, burstiness=$burstiness, QPS=$qps ==="

          # Restart vLLM and predictors for first run of each scheduler
          if [ "$RESTART_VLLM" = "true" ]; then
            sh block/exp/reset.sh
            sleep 30
            nohup sh block/exp/run_exp_vllm.sh $BATCH_CAP $model false 0 $MAX_MODEL_LENGTH true $BACKEND_WORKERS $CHUNK_SIZE > /dev/null 2>&1 &
            sleep 60
            for suffix in $(seq 1 $PREDICTOR_WORKERS); do
              nohup sh block/exp/run_exp_predictor_${suffix}.sh $PREDICTOR_CONFIG_PATH $scheduler true $BATCH_CAP true $PREDICTOR_WORKERS $BRANCH_NAME $BATCH_SIZE_THRESHOLD_FOR_TIME_ESTIMATION $PREDICTOR_TIMEOUT_IN_SECONDS > /dev/null 2>&1 &
            done
            sleep 60
            RESTART_VLLM=false
          fi

          # Start global scheduler
          nohup sh block/exp/run_exp_global_scheduler.sh $TARGET_HOST $N_SELECTED $N_SELECTED $scheduler $HOST_CONFIG_PATH $GLOBAL_SCHEDULER_WORKERS $PREDICTOR_WORKERS $PROFILING_SAMPLE_RATE $TIMEOUT_IN_SECONDS $PREDICTOR_TIMEOUT_IN_SECONDS $AVAILABLE_INSTANCE $MAX_SLO false > /dev/null 2>&1 &
          sleep 10

          # Run benchmark with burstiness parameter
          OUTPUT_DIR="${OUTPUT_DIR_PREFIX}/${dataset_name}/${scheduler}/qps_${qps}_burstiness_${burstiness}"

          if [ "$USE_LENGTH_ESTIMATION" = "true" ]; then
            parallel-ssh -i -t 0 --host $TARGET_HOST "cd Block && export PYTHONPATH=. && python block/benchmark/benchmark_serving.py --ip_ports 127.0.0.1:8200 --tokenizer $model --num_sampled_requests $NUM_REQUEST --dataset_type $dataset_name --dataset_path $dataset_path --qps $qps --backend block --log_filename benchmark.log --output_dir $OUTPUT_DIR --data_start_index $START_INDEX --trust_remote_code --max_request_len $MAX_MODEL_LENGTH --timeout_in_seconds $TIMEOUT_IN_SECONDS --use_estimated_response_lens --distribution gamma --burstiness $burstiness"
          else
            parallel-ssh -i -t 0 --host $TARGET_HOST "cd Block && export PYTHONPATH=. && python block/benchmark/benchmark_serving.py --ip_ports 127.0.0.1:8200 --tokenizer $model --num_sampled_requests $NUM_REQUEST --dataset_type $dataset_name --dataset_path $dataset_path --qps $qps --backend block --log_filename benchmark.log --output_dir $OUTPUT_DIR --data_start_index $START_INDEX --trust_remote_code --max_request_len $MAX_MODEL_LENGTH --timeout_in_seconds $TIMEOUT_IN_SECONDS --distribution gamma --burstiness $burstiness"
          fi

          sleep 10
          parallel-ssh --host $TARGET_HOST "cd Block && mkdir -p experiment_output/$OUTPUT_DIR/running_logs"
          parallel-ssh --host $TARGET_HOST "cd Block && mv experiment_output/logs/* experiment_output/$OUTPUT_DIR/running_logs/."
        done
      done
      # Reset for next scheduler
      RESTART_VLLM=true
    done
  done
done

echo "Burstiness experiment completed!"
echo "Results in: experiment_output/${OUTPUT_DIR_PREFIX}/"
