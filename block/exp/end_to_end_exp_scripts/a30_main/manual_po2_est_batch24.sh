#!/bin/bash
# NOTE (Apr 21 lesson): for absolute-metric experiments (heatmap/burstiness), every cell needs
# FRESH vLLM+predictor deploy. This script uses deploy-per-scheduler (shared across QPS probes)
# — acceptable for CAPACITY SEARCH but NOT for absolute-comparison runs. v2 fix: deploy_fresh
# inside QPS loop. Existing data accepted as-is.

# Manual probe for batch24/po2_est at QPS=28.4 (and 28.3 if needed) to refine capacity.
# Prior batch24/po2_est ran with buggy float_search that undershot to 28.2.
# Probes:
#   28.4: if 9.X band → capacity=28.5
#   28.4: if 10.X band → capacity=28.4
#   28.4: if >11s → probe 28.3
#   28.4: if <9s → we already know 28.5=11036ms, so capacity≈28.4 (unusual case)

set -u

TARGET_HOST="$(head -1 block/config/hosts)"
BATCH_CAP=24
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
DATASET_PATH="~/Block/data/trace_data/sharegpt/generate/llama"
METRIC_TYPE=min_new_request_latency
N_SELECTED=2
SLO_MS=10000
RESULTS_FILE="/tmp/a30_phase4_2_simple_results.txt"

echo "=== Manual batch24/po2_est refinement starting $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$RESULTS_FILE"

echo "[deploy] reset + vllm + predictors (batch=24, chunk=512)"
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
echo "[deploy] done"

echo "[scheduler] launching po2 (N=2, metric=$METRIC_TYPE)"
ssh -o ConnectTimeout=5 "$TARGET_HOST" "pkill -f global_scheduler/api_server" 2>&1 || true
sleep 5
nohup sh block/exp/run_exp_global_scheduler.sh "$TARGET_HOST" $N_SELECTED $N_SELECTED $METRIC_TYPE $HOST_CONFIG_PATH 1 $PREDICTOR_WORKERS $PROFILING_SAMPLE_RATE $TIMEOUT_IN_SECONDS $PREDICTOR_TIMEOUT_IN_SECONDS $AVAILABLE_INSTANCE $MAX_SLO false > /dev/null 2>&1 &
sleep 20

run_bench() {
    local qps=$1
    local output_dir="generality_float/batch24/po2_est/qps_${qps}"
    parallel-ssh -i -t 0 --host "$TARGET_HOST" "cd Block && export PYTHONPATH=. && export HF_TOKEN= && python block/benchmark/benchmark_serving.py --ip_ports 127.0.0.1:8200 --tokenizer $MODEL --num_sampled_requests $NUM_REQUEST --dataset_type sharegpt --dataset_path $DATASET_PATH --qps $qps --backend block --log_filename benchmark.log --output_dir $output_dir --data_start_index 0 --trust_remote_code --max_request_len $MAX_MODEL_LENGTH --timeout_in_seconds $TIMEOUT_IN_SECONDS --use_estimated_response_lens" > /dev/null 2>&1
    sleep 5
    local npz="/tmp/p42_$(echo $output_dir | tr '/' '_').npz"
    rsync -az "asdwb@d7525-10s10327.wisc.cloudlab.us:~/Block/experiment_output/$output_dir/benchmark_all_metrics.npz" "$npz" > /dev/null 2>&1
    python3 -c "import numpy as np; d=np.load('$npz', allow_pickle=True); v=d['prefill_token_latencies']; print(f'{np.percentile(v,99):.0f}')"
}

echo "[probe] QPS=28.4"
p99_28_4=$(run_bench 28.4)
echo "[batch24/po2_est MANUAL] QPS=28.4 TTFT P99=${p99_28_4}ms" | tee -a "$RESULTS_FILE"

capacity=""
if [ "$(awk "BEGIN{print (9000 <= $p99_28_4 && $p99_28_4 < 10000)}")" = "1" ]; then
    capacity=28.5
    echo "[batch24/po2_est MANUAL] 9.X at 28.4, capacity=28.5 (28.4+0.1)" | tee -a "$RESULTS_FILE"
elif [ "$(awk "BEGIN{print (10000 <= $p99_28_4 && $p99_28_4 < 11000)}")" = "1" ]; then
    capacity=28.4
    echo "[batch24/po2_est MANUAL] 10.X at 28.4, capacity=28.4" | tee -a "$RESULTS_FILE"
elif [ "$(awk "BEGIN{print ($p99_28_4 >= 11000)}")" = "1" ]; then
    echo "[batch24/po2_est MANUAL] 28.4 over (>=11s), probing 28.3"
    p99_28_3=$(run_bench 28.3)
    echo "[batch24/po2_est MANUAL] QPS=28.3 TTFT P99=${p99_28_3}ms" | tee -a "$RESULTS_FILE"
    if [ "$(awk "BEGIN{print (9000 <= $p99_28_3 && $p99_28_3 < 10000)}")" = "1" ]; then
        capacity=28.4
        echo "[batch24/po2_est MANUAL] 9.X at 28.3, capacity=28.4 (28.3+0.1)" | tee -a "$RESULTS_FILE"
    elif [ "$(awk "BEGIN{print (10000 <= $p99_28_3 && $p99_28_3 < 11000)}")" = "1" ]; then
        capacity=28.3
        echo "[batch24/po2_est MANUAL] 10.X at 28.3, capacity=28.3" | tee -a "$RESULTS_FILE"
    elif [ "$(awk "BEGIN{print ($p99_28_3 < 9000)}")" = "1" ]; then
        capacity="28.3-to-28.4"
        echo "[batch24/po2_est MANUAL] 28.3 under 9s, 28.4 over 11s — capacity between; conservative=28.3" | tee -a "$RESULTS_FILE"
    else
        capacity="<28.3"
        echo "[batch24/po2_est MANUAL] 28.3 over 11s — capacity <28.3, keep prior 28.2" | tee -a "$RESULTS_FILE"
    fi
else
    # p99_28_4 < 9000
    capacity=">=28.5 (but prior 28.5=11036 so anomaly)"
    echo "[batch24/po2_est MANUAL] 28.4 under 9s (anomaly: prior 28.5=11036), capacity=28.4 conservative" | tee -a "$RESULTS_FILE"
    capacity=28.4
fi

echo "=== batch24/po2_est FINAL capacity=${capacity} at $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$RESULTS_FILE"
