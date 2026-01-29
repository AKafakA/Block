#!/bin/bash
# Error Sensitivity Heatmap Experiment
# Purpose: Test Block's robustness to prediction errors by injecting controlled noise.
#
# This addresses R12A's concern: "How sensitive is the system to prediction error?"
# Creates a 6×6 grid: length_error × latency_error (0%, 10%, 20%, 30%, 40%, 50%)
# Plus one baseline run with no error injection.
#
# Total runs: 37 (6×6 + 1 baseline)

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

# Config for error heatmap experiment
ENABLE_CHUNKED_PREFILL="true"
MODEL="meta-llama/Llama-2-7b-hf"
DATASET_NAMES="sharegpt"
SCHEDULER_NAME="min_new_request_latency"
# Use moderate QPS where scheduling decisions matter
QPS="28"
PROFILING_SAMPLE_RATE=0.000
USE_FOR_PROFILING_ONLY=false
NUM_REQUEST=3000
KEEP_ALL_METRICS=false
N_SELECTED="12"
OUTPUT_DIR_PREFIX="error_heatmap"

# Error levels for heatmap (percentage)
LENGTH_ERROR_LEVELS="0 10 20 30 40 50"
LATENCY_ERROR_LEVELS="0 10 20 30 40 50"

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

    # Initial setup: start vLLM and predictors once
    echo "=== Setting up vLLM and Predictors ==="
    sh block/exp/reset.sh
    sleep 30
    nohup sh block/exp/run_exp_vllm.sh $BATCH_CAP $model false 0 $MAX_MODEL_LENGTH true $BACKEND_WORKERS $CHUNK_SIZE > /dev/null 2>&1 &
    sleep 60
    for suffix in $(seq 1 $PREDICTOR_WORKERS); do
      nohup sh block/exp/run_exp_predictor_${suffix}.sh $PREDICTOR_CONFIG_PATH $SCHEDULER_NAME true $BATCH_CAP true $PREDICTOR_WORKERS $BRANCH_NAME $BATCH_SIZE_THRESHOLD_FOR_TIME_ESTIMATION $PREDICTOR_TIMEOUT_IN_SECONDS > /dev/null 2>&1 &
    done
    sleep 60

    run_count=0
    total_runs=37

    for length_error in $LENGTH_ERROR_LEVELS; do
      for latency_error in $LATENCY_ERROR_LEVELS; do
        run_count=$((run_count + 1))
        echo "=== Error Heatmap [$run_count/$total_runs]: length_error=${length_error}%, latency_error=${latency_error}% ==="

        for qps in $QPS; do
          # Start global scheduler with error injection parameters
          # Parameters 14 and 15 are LENGTH_ERROR_PCT and LATENCY_ERROR_PCT
          nohup sh block/exp/run_exp_global_scheduler.sh $TARGET_HOST $N_SELECTED $N_SELECTED $SCHEDULER_NAME $HOST_CONFIG_PATH $GLOBAL_SCHEDULER_WORKERS $PREDICTOR_WORKERS $PROFILING_SAMPLE_RATE $TIMEOUT_IN_SECONDS $PREDICTOR_TIMEOUT_IN_SECONDS $AVAILABLE_INSTANCE $MAX_SLO false $length_error $latency_error > /dev/null 2>&1 &
          sleep 10

          # Run benchmark
          OUTPUT_DIR="${OUTPUT_DIR_PREFIX}/${dataset_name}/${SCHEDULER_NAME}/qps_${qps}_len_err_${length_error}_lat_err_${latency_error}"

          parallel-ssh -i -t 0 --host $TARGET_HOST "cd Block && export PYTHONPATH=. && python block/benchmark/benchmark_serving.py --ip_ports 127.0.0.1:8200 --tokenizer $model --num_sampled_requests $NUM_REQUEST --dataset_type $dataset_name --dataset_path $dataset_path --qps $qps --backend block --log_filename benchmark.log --output_dir $OUTPUT_DIR --data_start_index $START_INDEX --trust_remote_code --max_request_len $MAX_MODEL_LENGTH --timeout_in_seconds $TIMEOUT_IN_SECONDS --use_estimated_response_lens"

          sleep 5
          parallel-ssh --host $TARGET_HOST "cd Block && mkdir -p experiment_output/$OUTPUT_DIR/running_logs"
          parallel-ssh --host $TARGET_HOST "cd Block && mv experiment_output/logs/* experiment_output/$OUTPUT_DIR/running_logs/."

          # Kill global scheduler before next iteration
          parallel-ssh -t 0 --host $TARGET_HOST "pkill -f global_scheduler"
          sleep 5
        done
      done
    done
  done
done

echo "Error heatmap experiment completed!"
echo "Results in: experiment_output/${OUTPUT_DIR_PREFIX}/"
echo "Total runs: $run_count"
