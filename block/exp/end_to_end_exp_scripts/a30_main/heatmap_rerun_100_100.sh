#!/bin/bash
# Verify (100%, 100%) heatmap cell — rerun with same script logic but ONE cell only.
# Output goes to qps_32_len_err_100_lat_err_100_RERUN to avoid overwriting.

set -u

TARGET_HOST="$(head -1 block/config/hosts)"
BATCH_CAP=48
CHUNK_SIZE=512
PREDICTOR_WORKERS=16
GLOBAL_SCHEDULER_WORKERS=1
BACKEND_WORKERS=1
MAX_MODEL_LENGTH=4096
TIMEOUT_IN_SECONDS=1800
PREDICTOR_TIMEOUT_IN_SECONDS=1000
BATCH_SIZE_THRESHOLD_FOR_TIME_ESTIMATION="0"
BRANCH_NAME="main"
MODEL="meta-llama/Llama-2-7b-hf"
DATASET_PATH="~/Block/data/trace_data/sharegpt/generate/llama"
SCHEDULER_NAME="min_new_request_latency"
N_SELECTED=2
QPS=32
NUM_REQUEST=10000
PROFILING_SAMPLE_RATE=0.000
AVAILABLE_INSTANCE="12"
MAX_SLO="0"
HOST_CONFIG_PATH='block/config/host_configs.json'
PRED_CONFIG_PATH='block/config/llama_config.json'
OUTPUT_DIR="error_heatmap_po2_rerun/sharegpt/${SCHEDULER_NAME}/qps_${QPS}_len_err_100_lat_err_100_RERUN"
RESULTS_FILE=/tmp/a30_phase4_2_simple_results.txt

echo "=== (100%,100%) rerun starting $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$RESULTS_FILE"

echo "[rerun] reset + vllm + predictors"
sh block/exp/reset.sh
sleep 30
nohup sh block/exp/run_exp_vllm.sh $BATCH_CAP $MODEL false 0 $MAX_MODEL_LENGTH true $BACKEND_WORKERS $CHUNK_SIZE > /dev/null 2>&1 &
sleep 90
for s in $(seq 1 7); do
    nohup sh block/exp/run_exp_predictor_${s}.sh $PRED_CONFIG_PATH $SCHEDULER_NAME true $BATCH_CAP true $PREDICTOR_WORKERS $BRANCH_NAME 0 $PREDICTOR_TIMEOUT_IN_SECONDS > /dev/null 2>&1 &
done
sleep 10
for s in $(seq 8 $PREDICTOR_WORKERS); do
    nohup sh block/exp/run_exp_predictor_${s}.sh $PRED_CONFIG_PATH $SCHEDULER_NAME true $BATCH_CAP true $PREDICTOR_WORKERS $BRANCH_NAME 0 $PREDICTOR_TIMEOUT_IN_SECONDS > /dev/null 2>&1 &
done
sleep 60
echo "[rerun] deploy done"

# verify_predictors gate
bash block/exp/end_to_end_exp_scripts/a30_main/verify_predictors.sh "$PRED_CONFIG_PATH" "$SCHEDULER_NAME" "$BATCH_CAP" "$PREDICTOR_WORKERS" "$PREDICTOR_TIMEOUT_IN_SECONDS"

# Launch scheduler with --length_error_pct 100 --latency_error_pct 100
ssh -n -o ConnectTimeout=5 "$TARGET_HOST" "pkill -f global_scheduler/api_server" 2>&1 || true
sleep 5
nohup sh block/exp/run_exp_global_scheduler.sh "$TARGET_HOST" $N_SELECTED $N_SELECTED $SCHEDULER_NAME $HOST_CONFIG_PATH $GLOBAL_SCHEDULER_WORKERS $PREDICTOR_WORKERS $PROFILING_SAMPLE_RATE $TIMEOUT_IN_SECONDS $PREDICTOR_TIMEOUT_IN_SECONDS $AVAILABLE_INSTANCE $MAX_SLO false 100 100 > /dev/null 2>&1 &
sleep 20

echo "[rerun] benchmark QPS=32 with err=100/100"
parallel-ssh -i -t 0 --host "$TARGET_HOST" "cd Block && export PYTHONPATH=. && export HF_TOKEN= && python block/benchmark/benchmark_serving.py --ip_ports 127.0.0.1:8200 --tokenizer $MODEL --num_sampled_requests $NUM_REQUEST --dataset_type sharegpt --dataset_path $DATASET_PATH --qps $QPS --backend block --log_filename benchmark.log --output_dir $OUTPUT_DIR --data_start_index 0 --trust_remote_code --max_request_len $MAX_MODEL_LENGTH --timeout_in_seconds $TIMEOUT_IN_SECONDS --use_estimated_response_lens" > /dev/null 2>&1
sleep 5

# Compute and report
npz=/tmp/p42_heatmap_rerun_100_100.npz
rsync -az "asdwb@d7525-10s10327.wisc.cloudlab.us:~/Block/experiment_output/$OUTPUT_DIR/benchmark_all_metrics.npz" "$npz" > /dev/null 2>&1
python3 <<EOF | tee -a "$RESULTS_FILE"
import numpy as np
d = np.load("$npz", allow_pickle=True)
req = d['request_latencies']; ttft = d['prefill_token_latencies']
print(f"[(100,100) RERUN] e2e_mean={float(np.mean(req)):.0f}, e2e_P99={float(np.percentile(req,99)):.0f}, TTFT_P99={float(np.percentile(ttft,99)):.0f}, aQPS={float(d['actual_qps']):.2f}")
print(f"  Reference (Phase 1.2 Po2-est QPS=32): e2e_mean=14737, e2e_P99=41570, TTFT_P99=12152, aQPS=28.38")
print(f"  Original heatmap cell:                  e2e_mean=14264 (-3.2%), e2e_P99=40341 (-3.0%), TTFT_P99=12970 (+6.7%)")
EOF

echo "=== (100%,100%) rerun done at $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$RESULTS_FILE"
