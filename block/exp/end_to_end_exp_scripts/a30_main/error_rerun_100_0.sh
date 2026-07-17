#!/bin/bash
# (100,0) error injection REDO2 — single cell, NO CPU tracking.
# Assumes warmup already done (predictor 1 alive + cache populated).
# Launches predictors 2-16 (cache load fast), then scheduler with --length_error_pct 100.

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
DATASET_PATH="~/Block/data/trace_data/sharegpt/generate/llama"
METRIC_TYPE=min_new_request_latency
RESULTS_FILE=/tmp/a30_phase4_2_simple_results.txt
LOG=/tmp/a30_error_rerun_100_0.log

echo "=== (100,0) REDO2 + CPU tracker smoke test starting $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$LOG" "$RESULTS_FILE"

# Step A: sync patched scheduler (needed to extract cpu_percent from predictor responses)
echo "[sync] pushing patched api_server.py to all 12 nodes"
parallel-scp -h block/config/hosts block/global_scheduler/api_server.py Block/block/global_scheduler/api_server.py > /dev/null 2>&1
echo "[sync] done"

# Step B: kill warmup's predictor 1 so we can relaunch with CPU tracking enabled (else only 15 of 16 would have CPU tracking)
echo "[deploy] killing warmup's predictor 1 to relaunch all 16 uniformly with CPU tracking"
parallel-ssh -i -t 0 -h block/config/hosts "pkill -f 'predictor/api_server.py.*--predictor_index 1$'" > /dev/null 2>&1
sleep 3

# Launch all 16 predictors with --enable_cpu_tracking (10th arg = true)
echo "[deploy] launching 16 predictors with CPU tracking (cache hit, fast)"
for suffix in $(seq 1 7); do
    nohup sh block/exp/run_exp_predictor_${suffix}.sh $PRED_CONFIG_PATH $METRIC_TYPE true $BATCH_CAP true $PREDICTOR_WORKERS $BRANCH_NAME 0 $PREDICTOR_TIMEOUT_IN_SECONDS true > /dev/null 2>&1 &
done
sleep 10
for suffix in $(seq 8 $PREDICTOR_WORKERS); do
    nohup sh block/exp/run_exp_predictor_${suffix}.sh $PRED_CONFIG_PATH $METRIC_TYPE true $BATCH_CAP true $PREDICTOR_WORKERS $BRANCH_NAME 0 $PREDICTOR_TIMEOUT_IN_SECONDS true > /dev/null 2>&1 &
done
sleep 60
bash block/exp/end_to_end_exp_scripts/a30_main/verify_predictors.sh "$PRED_CONFIG_PATH" "$METRIC_TYPE" "$BATCH_CAP" "$PREDICTOR_WORKERS" "$PREDICTOR_TIMEOUT_IN_SECONDS" || true
echo "[deploy] done"

# Scheduler with --length_error_pct 100 --latency_error_pct 0
ssh -n -o ConnectTimeout=5 "$TARGET_HOST" "pkill -f global_scheduler/api_server" 2>&1 || true
sleep 5
nohup sh block/exp/run_exp_global_scheduler.sh "$TARGET_HOST" 2 2 $METRIC_TYPE $HOST_CONFIG_PATH 1 $PREDICTOR_WORKERS $PROFILING_SAMPLE_RATE $TIMEOUT_IN_SECONDS $PREDICTOR_TIMEOUT_IN_SECONDS $AVAILABLE_INSTANCE $MAX_SLO false 100 0 > /dev/null 2>&1 &
sleep 20

OUTPUT_DIR="error_heatmap_po2_rerun/sharegpt/${METRIC_TYPE}/qps_32_len_err_100_lat_err_0_REDO2"
parallel-ssh -i -t 0 --host "$TARGET_HOST" "cd Block && export PYTHONPATH=. && export HF_TOKEN= && python block/benchmark/benchmark_serving.py --ip_ports 127.0.0.1:8200 --tokenizer $MODEL --num_sampled_requests $NUM_REQUEST --dataset_type sharegpt --dataset_path $DATASET_PATH --qps 32 --backend block --log_filename benchmark.log --output_dir $OUTPUT_DIR --data_start_index 0 --trust_remote_code --max_request_len $MAX_MODEL_LENGTH --timeout_in_seconds $TIMEOUT_IN_SECONDS --use_estimated_response_lens" > /dev/null 2>&1
sleep 5
npz=/tmp/redo2_100_0.npz
rsync -az "asdwb@d7525-10s10327.wisc.cloudlab.us:~/Block/experiment_output/$OUTPUT_DIR/benchmark_all_metrics.npz" "$npz" > /dev/null 2>&1
python3 -c "
import numpy as np
d = np.load('$npz', allow_pickle=True)
req = d['request_latencies']; ttft = d['prefill_token_latencies']
keys = list(d.keys())
cpu_present = 'mean_cpu_percent' in keys
cpu_str = f', mean_cpu={float(d[\"mean_cpu_percent\"]):.2f}%, max_cpu={float(d[\"max_cpu_percent\"]):.2f}%, mem_rss={float(d[\"mean_memory_rss_mb\"]):.0f}MB' if cpu_present else ', CPU_FIELDS_MISSING'
print(f'[(100,0) REDO2] e2e_mean={float(np.mean(req)):.0f}, e2e_P99={float(np.percentile(req,99)):.0f}, TTFT_P99={float(np.percentile(ttft,99)):.0f} vs baseline 14737{cpu_str}')
print(f'[smoke test] NPZ keys with cpu/mem: {[k for k in keys if \"cpu\" in k or \"memory\" in k]}')
" | tee -a "$LOG" "$RESULTS_FILE"
echo "=== (100,0) REDO2 complete at $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$LOG" "$RESULTS_FILE"
