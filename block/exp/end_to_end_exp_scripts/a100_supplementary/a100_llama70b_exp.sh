#!/bin/bash
# A100-40GB Llama-70B Latency Prediction Accuracy Experiment
# Purpose: Validate that Block's Vidur-based simulator prediction accuracy generalizes to larger models.
#
# This addresses R12C's concern: "All experiments evaluate only a 7B-parameter model"
#
# Configuration:
# - Model: Llama-2-70B (4-way tensor parallel on 4×A100-40GB)
# - Cluster: CloudLab d8545 nodes (4×A100 SXM4 40GB with NVLink)
# - Metric: Latency prediction accuracy (predicted vs actual serving latency)
#
# PREREQUISITES (must complete before running this script):
#   1. Provision CloudLab d8545 nodes
#   2. Run profiling: sh block/exp/end_to_end_exp_scripts/a100_supplementary/a100_40gb_profiling.sh
#   3. Verify profiling data exists in data/profiling/compute/a100_40gb/
#   4. Download Llama-2-70B model weights
#   5. Create block/config/a100_host_configs.json with node IPs
#
# Workflow:
#   Phase 1: Profiling (a100_40gb_profiling.sh) - generates execution time profiles
#   Phase 2: This script - runs benchmark with high profiling rate
#   Phase 3: Analysis - compare predicted vs actual latency

START_INDEX=0
BATCH_CAP=16
# Fewer predictor workers for 70B (more compute per prediction)
PREDICTOR_WORKERS=4
GLOBAL_SCHEDULER_WORKERS=1
BACKEND_WORKERS=1
MAX_MODEL_LENGTH=4096
CHUNK_SIZE=512
TIMEOUT_IN_SECONDS=3600
PREDICTOR_TIMEOUT_IN_SECONDS=2000
BATCH_SIZE_THRESHOLD_FOR_TIME_ESTIMATION="0"
BRANCH_NAME="main"
USE_PROCESS_FOR_FRONTEND=true
UPDATE_BLOCK_CODE=false
UPDATE_VLLM_CODE=false
RUN_EXP=true
RESTART_VLLM=true

# Config for Llama-70B experiment on A100-40GB
ENABLE_CHUNKED_PREFILL="true"
MODEL="meta-llama/Llama-2-70b-hf"
MODEL_TYPE="llama70b"
DATASET_NAMES="sharegpt"
SCHEDULER_NAME="min_new_request_latency"
# Lower QPS for 70B model (slower inference)
QPS="2 4 6"
# HIGH profiling rate to collect prediction accuracy data
PROFILING_SAMPLE_RATE=0.5
USE_FOR_PROFILING_ONLY=false
NUM_REQUEST=500
KEEP_ALL_METRICS=true
N_SELECTED="1"  # Single instance for 70B (one node with 4 GPUs)
OUTPUT_DIR_PREFIX="a100_40gb_llama70b"

# A100-40GB cluster config
AVAILABLE_INSTANCE="1"
ENABLE_PREEMPTIVE_AUTO_PROVISIONING="false"
MAX_SLO="0"

TARGET_HOST=""  # Fill with your CloudLab d8545 global scheduler host

# A100-40GB specific config paths
HOST_CONFIG_PATH='block/config/a100_host_configs.json'
PREDICTOR_CONFIG_PATH="block/config/llama70b_a100_40gb_config.json"

echo "=============================================="
echo "A100-40GB Llama-70B Prediction Accuracy Experiment"
echo "=============================================="
echo ""
echo "Prerequisites checklist:"
echo "  [ ] CloudLab d8545 nodes provisioned"
echo "  [ ] Profiling completed (a100_40gb_profiling.sh)"
echo "  [ ] Profiling data exists: data/profiling/compute/a100_40gb/meta-llama/Llama-2-70b-hf/"
echo "  [ ] Model weights downloaded: meta-llama/Llama-2-70b-hf"
echo "  [ ] Host config created: block/config/a100_host_configs.json"
echo ""

# Verify profiling data exists
PROFILE_PATH="data/profiling/compute/a100_40gb/meta-llama/Llama-2-70b-hf"
if [ ! -f "${PROFILE_PATH}/mlp.csv" ] || [ ! -f "${PROFILE_PATH}/attention.csv" ]; then
    echo "ERROR: Profiling data not found at ${PROFILE_PATH}"
    echo "Please run: sh block/exp/end_to_end_exp_scripts/a100_supplementary/a100_40gb_profiling.sh"
    exit 1
fi
echo "Profiling data found. Proceeding..."
echo ""

for model in $MODEL; do
  echo "Running warmup script for ${model} model (this may take a while)..."
  sh block/exp/end_to_end_exp_scripts/a30_main/warmup.sh ${model} > /dev/null 2>&1

  for dataset_name in $DATASET_NAMES; do
    dataset_path="~/Block/data/trace_data/$dataset_name/generate/llama"  # Reuse llama dataset

    echo "=== Setting up vLLM for Llama-70B (4-way tensor parallel) ==="
    sh block/exp/reset.sh
    sleep 30

    # Start vLLM with tensor parallelism for 70B
    # Note: vLLM command needs --tensor-parallel-size 4
    nohup sh block/exp/run_exp_vllm.sh $BATCH_CAP $model false 0 $MAX_MODEL_LENGTH true $BACKEND_WORKERS $CHUNK_SIZE > /dev/null 2>&1 &
    sleep 180  # Longer wait for 70B model loading

    # Start predictors
    for suffix in $(seq 1 $PREDICTOR_WORKERS); do
      nohup sh block/exp/run_exp_predictor_${suffix}.sh $PREDICTOR_CONFIG_PATH $SCHEDULER_NAME true $BATCH_CAP true $PREDICTOR_WORKERS $BRANCH_NAME $BATCH_SIZE_THRESHOLD_FOR_TIME_ESTIMATION $PREDICTOR_TIMEOUT_IN_SECONDS > /dev/null 2>&1 &
    done
    sleep 60

    for qps in $QPS; do
      echo "=== Llama-70B Experiment: QPS=$qps ==="

      # Start global scheduler with high profiling rate
      nohup sh block/exp/run_exp_global_scheduler.sh $TARGET_HOST $N_SELECTED $N_SELECTED $SCHEDULER_NAME $HOST_CONFIG_PATH $GLOBAL_SCHEDULER_WORKERS $PREDICTOR_WORKERS $PROFILING_SAMPLE_RATE $TIMEOUT_IN_SECONDS $PREDICTOR_TIMEOUT_IN_SECONDS $AVAILABLE_INSTANCE $MAX_SLO false > /dev/null 2>&1 &
      sleep 10

      # Run benchmark with high profiling rate to measure prediction accuracy
      OUTPUT_DIR="${OUTPUT_DIR_PREFIX}/${dataset_name}/${SCHEDULER_NAME}/qps_${qps}"

      parallel-ssh -i -t 0 --host $TARGET_HOST "cd Block && export PYTHONPATH=. && python block/benchmark/benchmark_serving.py --ip_ports 127.0.0.1:8200 --tokenizer $model --num_sampled_requests $NUM_REQUEST --dataset_type $dataset_name --dataset_path $dataset_path --qps $qps --backend block --log_filename benchmark.log --output_dir $OUTPUT_DIR --data_start_index $START_INDEX --trust_remote_code --max_request_len $MAX_MODEL_LENGTH --timeout_in_seconds $TIMEOUT_IN_SECONDS --use_estimated_response_lens"

      sleep 10
      parallel-ssh --host $TARGET_HOST "cd Block && mkdir -p experiment_output/$OUTPUT_DIR/running_logs"
      parallel-ssh --host $TARGET_HOST "cd Block && mv experiment_output/logs/* experiment_output/$OUTPUT_DIR/running_logs/."

      # Kill global scheduler before next iteration
      parallel-ssh -t 0 --host $TARGET_HOST "pkill -f global_scheduler"
      sleep 5
    done
  done
done

echo ""
echo "=============================================="
echo "Llama-70B experiment completed!"
echo "=============================================="
echo "Results in: experiment_output/${OUTPUT_DIR_PREFIX}/"
echo ""
echo "Key metrics to extract from profiling samples:"
echo "  - sampled_mean_error_ratios: Mean prediction error ratio"
echo "  - sampled_predict_accuracies: Prediction vs actual comparison"
echo "  - sampled_predict_latency: Predicted latency values"
echo "  - sampled_serving_latencies: Actual serving latencies"
echo ""
echo "Analysis script: experiments_analysis/prediction_plot.py"
