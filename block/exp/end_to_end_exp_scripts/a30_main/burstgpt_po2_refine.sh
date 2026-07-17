#!/bin/bash
# NOTE (Apr 21 lesson): for absolute-metric experiments (heatmap/burstiness), every cell needs
# FRESH vLLM+predictor deploy. This script uses deploy-per-scheduler (shared across QPS probes)
# — acceptable for CAPACITY SEARCH but NOT for absolute-comparison runs. v2 fix: deploy_fresh
# inside QPS loop. Existing data accepted as-is.

# BurstGPT Po2-oracle capacity refinement — probe QPS=64.5 and 65 with verified 192/192 predictors.
# Prior 15/16 result: capacity=64.5 (TTFT at 64.5 = 10734ms, 10.X band).
# Smoke test 16/16 result: at 64.5 TTFT = 7000ms (well under SLO) → true capacity higher.
# Probe 65 to see if it's still under 10s SLO or crosses.

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
RESULTS_FILE="/tmp/a30_phase4_2_simple_results.txt"

echo "=== BurstGPT Po2-oracle refinement starting $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$RESULTS_FILE"

echo "[refine] reset + vllm + predictors (batch=48, chunk=512)"
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
echo "[refine] deploy done"

# HARD GATE
if ! bash block/exp/end_to_end_exp_scripts/a30_main/verify_predictors.sh \
    "$PRED_CONFIG_PATH" "$METRIC_TYPE" "$BATCH_CAP" "$PREDICTOR_WORKERS" "$PREDICTOR_TIMEOUT_IN_SECONDS"; then
    echo "[refine] ABORT: predictor verification failed" | tee -a "$RESULTS_FILE"
    exit 1
fi

echo "[refine] launching scheduler po2 N=2"
ssh -n -o ConnectTimeout=5 "$TARGET_HOST" "pkill -f global_scheduler/api_server" 2>&1 || true
sleep 5
nohup sh block/exp/run_exp_global_scheduler.sh "$TARGET_HOST" $N_SELECTED $N_SELECTED $METRIC_TYPE $HOST_CONFIG_PATH 1 $PREDICTOR_WORKERS $PROFILING_SAMPLE_RATE $TIMEOUT_IN_SECONDS $PREDICTOR_TIMEOUT_IN_SECONDS $AVAILABLE_INSTANCE $MAX_SLO false > /dev/null 2>&1 &
sleep 20

probe() {
    local qps=$1
    local output_dir="generality_float/burstgpt/po2_oracle_refine/qps_${qps}"
    parallel-ssh -i -t 0 --host "$TARGET_HOST" "cd Block && export PYTHONPATH=. && export HF_TOKEN= && python block/benchmark/benchmark_serving.py --ip_ports 127.0.0.1:8200 --tokenizer $MODEL --num_sampled_requests $NUM_REQUEST --dataset_type burstgpt --dataset_path $DATASET_PATH --qps $qps --backend block --log_filename benchmark.log --output_dir $output_dir --data_start_index 0 --trust_remote_code --max_request_len $MAX_MODEL_LENGTH --timeout_in_seconds $TIMEOUT_IN_SECONDS" > /dev/null 2>&1
    sleep 5
    local npz="/tmp/p42_$(echo $output_dir | tr '/' '_').npz"
    rsync -az "asdwb@d7525-10s10327.wisc.cloudlab.us:~/Block/experiment_output/$output_dir/benchmark_all_metrics.npz" "$npz" > /dev/null 2>&1
    python3 -c "import numpy as np; d=np.load('$npz', allow_pickle=True); v=d['prefill_token_latencies']; print(f'{np.percentile(v,99):.0f}')"
}

SLO=10000
echo "[refine] probe QPS=64.5 (confirm smoke)"
p65_5=$(probe 64.5)
echo "[burstgpt/po2_oracle REFINE] QPS=64.5 TTFT P99=${p65_5}ms" | tee -a "$RESULTS_FILE"

echo "[refine] probe QPS=65"
p65=$(probe 65)
echo "[burstgpt/po2_oracle REFINE] QPS=65 TTFT P99=${p65}ms" | tee -a "$RESULTS_FILE"

# Determine capacity from these probes
if [ "$(awk "BEGIN{print (9000 <= $p65 && $p65 < 10000)}")" = "1" ]; then
    cap="65.1 (9.X band at 65)"
elif [ "$(awk "BEGIN{print (10000 <= $p65 && $p65 < 11000)}")" = "1" ]; then
    cap="65 (10.X band at 65)"
elif [ "$(awk "BEGIN{print ($p65 < 9000)}")" = "1" ]; then
    echo "[refine] 65 still well under SLO (${p65}ms), probing QPS=66"
    p66=$(probe 66)
    echo "[burstgpt/po2_oracle REFINE] QPS=66 TTFT P99=${p66}ms" | tee -a "$RESULTS_FILE"
    if [ "$(awk "BEGIN{print (9000 <= $p66 && $p66 < 10000)}")" = "1" ]; then
        cap="66.1 (9.X band at 66)"
    elif [ "$(awk "BEGIN{print (10000 <= $p66 && $p66 < 11000)}")" = "1" ]; then
        cap=66
    elif [ "$(awk "BEGIN{print ($p66 < 9000)}")" = "1" ]; then
        cap=">66 (need further probe)"
    else
        cap=">65 <66 (66 over SLO, 65 under)"
    fi
else
    cap="~64.5 (65 over 11s, reverse search)"
fi

echo "=== BurstGPT Po2-oracle REFINED capacity=${cap} at $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$RESULTS_FILE"
