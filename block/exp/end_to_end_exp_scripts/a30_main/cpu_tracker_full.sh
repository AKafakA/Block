#!/bin/bash
# Full CPU tracker sweep with patched predictor + scheduler + benchmark.
# Syncs all 3 patched files, restarts predictors with --enable_cpu_tracking,
# runs Po2-est × QPS {20, 24, 28, 32, 36}.

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
QPS_LEVELS="20 24 28 32 36"
RESULTS_FILE=/tmp/a30_phase4_2_simple_results.txt
LOG=/tmp/a30_cpu_tracker_full.log

echo "=== CPU tracker FULL sweep starting $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$LOG" "$RESULTS_FILE"

# Step 1: sync all 3 patched files
echo "[sync] pushing patched predictor + scheduler + benchmark files to all 12 nodes"
parallel-scp -h block/config/hosts block/predictor/api_server.py Block/block/predictor/api_server.py > /dev/null 2>&1
parallel-scp -h block/config/hosts block/global_scheduler/api_server.py Block/block/global_scheduler/api_server.py > /dev/null 2>&1
parallel-scp -h block/config/hosts block/benchmark/benchmark_serving.py Block/block/benchmark/benchmark_serving.py > /dev/null 2>&1
echo "[sync] done"

# Step 2: kill all predictors + scheduler (vLLM stays — no model reload)
echo "[restart] killing all predictors + scheduler"
parallel-ssh -i -t 0 -h block/config/hosts "pkill -f predictor/api_server" > /dev/null 2>&1
ssh -n -o ConnectTimeout=5 "$TARGET_HOST" "pkill -f global_scheduler/api_server" 2>&1 || true
sleep 5

# Step 3: launch 16 predictors with patched code + --enable_cpu_tracking (cache hit)
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

# Step 4: launch scheduler N=2 Po2-est
nohup sh block/exp/run_exp_global_scheduler.sh "$TARGET_HOST" 2 2 $METRIC_TYPE $HOST_CONFIG_PATH 1 $PREDICTOR_WORKERS $PROFILING_SAMPLE_RATE $TIMEOUT_IN_SECONDS $PREDICTOR_TIMEOUT_IN_SECONDS $AVAILABLE_INSTANCE $MAX_SLO false > /dev/null 2>&1 &
sleep 20

# Step 5: CPU tracker Po2-est sweep
echo "=== [cpu_tracker/po2_est full] sweep starting ===" | tee -a "$LOG" "$RESULTS_FILE"
for qps in $QPS_LEVELS; do
    output_dir="cpu_tracker_po2_v2/sharegpt/${METRIC_TYPE}/qps_${qps}_n_2_len_estimated_true_cpu_tracking_true"
    parallel-ssh -i -t 0 --host "$TARGET_HOST" "cd Block && export PYTHONPATH=. && export HF_TOKEN= && python block/benchmark/benchmark_serving.py --ip_ports 127.0.0.1:8200 --tokenizer $MODEL --num_sampled_requests $NUM_REQUEST --dataset_type sharegpt --dataset_path $DATASET_PATH --qps $qps --backend block --log_filename benchmark.log --output_dir $output_dir --data_start_index 0 --trust_remote_code --max_request_len $MAX_MODEL_LENGTH --timeout_in_seconds $TIMEOUT_IN_SECONDS --use_estimated_response_lens" > /dev/null 2>&1
    sleep 5
    npz=/tmp/cpufull_$(echo $output_dir | tr '/' '_').npz
    rsync -az "asdwb@d7525-10s10327.wisc.cloudlab.us:~/Block/experiment_output/$output_dir/benchmark_all_metrics.npz" "$npz" > /dev/null 2>&1
    python3 -c "
import numpy as np
d = np.load('$npz', allow_pickle=True)
keys = list(d.keys())
ttft = d['prefill_token_latencies']
req = d['request_latencies']
cpu_present = 'cpu_percents' in keys and len(d['cpu_percents']) > 0
if cpu_present:
    cp = d['cpu_percents']
    rss = d['memory_rss_mb']
    cores = int(d['cpu_cores']) if 'cpu_cores' in keys else 0
    print(f'[cpu_tracker po2_est QPS=$qps] TTFT_P99={float(np.percentile(ttft,99)):.0f}ms, e2e_mean={float(np.mean(req)):.0f}, mean_cpu={float(np.mean(cp)):.1f}%, max_cpu={float(np.max(cp)):.1f}%, mean_rss={float(np.mean(rss)):.0f}MB, cores={cores}')
else:
    print(f'[cpu_tracker po2_est QPS=$qps] TTFT_P99={float(np.percentile(ttft,99)):.0f}ms NO_CPU_DATA keys={keys}')
" | tee -a "$LOG" "$RESULTS_FILE"
done

echo "=== CPU tracker FULL complete at $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$LOG" "$RESULTS_FILE"
