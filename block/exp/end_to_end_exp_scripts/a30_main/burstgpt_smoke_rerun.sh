#!/bin/bash
# Smoke test: rerun burstgpt/po2_oracle at QPS=64.5 with verified 192/192 predictors.
# Purpose:
#   1. Verify predictor gate (verify_predictors.sh) works end-to-end
#   2. Compare TTFT P99 to prior 15/16 run (10734ms) — tests hypothesis that missing predictor hurts Po2 more than Llumnix
#
# Usage: nohup bash block/exp/end_to_end_exp_scripts/a30_main/burstgpt_smoke_rerun.sh > /tmp/a30_burstgpt_smoke.log 2>&1 &

set -u

TARGET_HOST="$(head -1 block/config/hosts)"
BATCH_CAP=48
CHUNK_SIZE=512
PREDICTOR_WORKERS=16
MAX_MODEL_LENGTH=4096
TIMEOUT_IN_SECONDS=1800
PREDICTOR_TIMEOUT_IN_SECONDS=1000
BRANCH_NAME="main"
MODEL="meta-llama/Llama-2-7b-hf"
PROFILING_SAMPLE_RATE=0.000
NUM_REQUEST=10000
AVAILABLE_INSTANCE="12"
MAX_SLO="0"
HOST_CONFIG_PATH='block/config/host_configs.json'
PRED_CONFIG_PATH='block/config/llama_config.json'
DATASET_PATH="~/Block/data/trace_data/burstgpt/generate/llama"
METRIC_TYPE=min_new_request_latency
N_SELECTED=2
QPS=64.5
RESULTS_FILE="/tmp/a30_phase4_2_simple_results.txt"

echo "=== Smoke rerun burstgpt/po2_oracle QPS=${QPS} starting $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$RESULTS_FILE"

echo "[smoke] reset + vllm + predictors"
sh block/exp/reset.sh
sleep 30
nohup sh block/exp/run_exp_vllm.sh $BATCH_CAP $MODEL false 0 $MAX_MODEL_LENGTH true 1 $CHUNK_SIZE > /dev/null 2>&1 &
sleep 90
for suffix in $(seq 1 7); do
    nohup sh block/exp/run_exp_predictor_${suffix}.sh $PRED_CONFIG_PATH $METRIC_TYPE true $BATCH_CAP true $PREDICTOR_WORKERS $BRANCH_NAME 0 $PREDICTOR_TIMEOUT_IN_SECONDS > /dev/null 2>&1 &
done
sleep 10
for suffix in $(seq 8 $PREDICTOR_WORKERS); do
    nohup sh block/exp/run_exp_predictor_${suffix}.sh $PRED_CONFIG_PATH $METRIC_TYPE true $BATCH_CAP true $PREDICTOR_WORKERS $BRANCH_NAME 0 $PREDICTOR_TIMEOUT_IN_SECONDS > /dev/null 2>&1 &
done
sleep 60
echo "[smoke] initial deploy done at $(date -u +%H:%M:%SZ)"

# HARD GATE: verify all 192 predictors before benchmark
if ! bash block/exp/end_to_end_exp_scripts/a30_main/verify_predictors.sh \
    "$PRED_CONFIG_PATH" "$METRIC_TYPE" "$BATCH_CAP" "$PREDICTOR_WORKERS" "$PREDICTOR_TIMEOUT_IN_SECONDS"; then
    echo "[smoke] ABORT: predictor verification failed, refusing to benchmark" | tee -a "$RESULTS_FILE"
    exit 1
fi

echo "[smoke] launching scheduler po2 N=$N_SELECTED metric=$METRIC_TYPE"
ssh -o ConnectTimeout=5 "$TARGET_HOST" "pkill -f global_scheduler/api_server" 2>&1 || true
sleep 5
nohup sh block/exp/run_exp_global_scheduler.sh "$TARGET_HOST" $N_SELECTED $N_SELECTED $METRIC_TYPE $HOST_CONFIG_PATH 1 $PREDICTOR_WORKERS $PROFILING_SAMPLE_RATE $TIMEOUT_IN_SECONDS $PREDICTOR_TIMEOUT_IN_SECONDS $AVAILABLE_INSTANCE $MAX_SLO false > /dev/null 2>&1 &
sleep 20

output_dir="generality_float/burstgpt/po2_oracle_smoke/qps_${QPS}"
echo "[smoke] running benchmark QPS=$QPS"
parallel-ssh -i -t 0 --host "$TARGET_HOST" "cd Block && export PYTHONPATH=. && export HF_TOKEN= && python block/benchmark/benchmark_serving.py --ip_ports 127.0.0.1:8200 --tokenizer $MODEL --num_sampled_requests $NUM_REQUEST --dataset_type burstgpt --dataset_path $DATASET_PATH --qps $QPS --backend block --log_filename benchmark.log --output_dir $output_dir --data_start_index 0 --trust_remote_code --max_request_len $MAX_MODEL_LENGTH --timeout_in_seconds $TIMEOUT_IN_SECONDS" > /dev/null 2>&1

sleep 5
npz="/tmp/p42_$(echo $output_dir | tr '/' '_').npz"
rsync -az "asdwb@d7525-10s10327.wisc.cloudlab.us:~/Block/experiment_output/$output_dir/benchmark_all_metrics.npz" "$npz" > /dev/null 2>&1
p99=$(python3 -c "import numpy as np; d=np.load('$npz', allow_pickle=True); v=d['prefill_token_latencies']; print(f'{np.percentile(v,99):.0f}')")

echo "[smoke] QPS=$QPS TTFT P99=${p99}ms (prior 15/16 run: 10734ms)" | tee -a "$RESULTS_FILE"

# Compare
delta=$(python3 -c "print(f'{(10734 - $p99) / 10734 * 100:+.1f}')")
echo "[smoke] delta from prior: ${delta}% (negative=worse with 16/16, positive=better with 16/16)" | tee -a "$RESULTS_FILE"

echo "=== Smoke rerun complete at $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$RESULTS_FILE"
