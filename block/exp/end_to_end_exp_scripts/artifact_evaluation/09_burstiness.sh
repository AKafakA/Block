#!/bin/bash
# 09_burstiness.sh — Sec 6.5: Burstiness sensitivity
#
# Runs BOTH Po2 (N=2) and Fanout (N=12) variants against Llumnix-N12 so the
# artifact reproduces the Block-Pow2 paper (Po2 emphasis) AND the original
# Block paper (Fanout baseline) simultaneously.
#
# Pass 1 (N=2):  min_new_request_latency + min_lunmnix_load × burst {0.25, 0.5, 1.0, 2.0}
# Pass 2 (N=12): min_new_request_latency (Fanout-est) × same 4 burst levels
#
# ~45-60 min total.

set -u
REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$REPO_ROOT"

echo "=== 09_burstiness: Sec 6.5 (Po2 + Fanout + Llumnix) ==="
date -u +%Y-%m-%dT%H:%M:%SZ
TARGET_HOST="$(head -1 block/config/hosts)"

# Verify Llumnix N=12 branching present in burstiness_exp.sh
if ! grep -q 'N_THIS_RUN="12"' block/exp/end_to_end_exp_scripts/burstiness_exp.sh; then
    echo "FAIL: burstiness_exp.sh missing Llumnix N=12 branch"
    exit 1
fi
echo "[ok] Llumnix N=12 branching present"

# Pass 1 — default burstiness_exp.sh (Po2 N=2 + Llumnix N=12)
echo "--- Pass 1: Po2-est + Llumnix-N12 ---"
N_SELECTED=2 nohup sh block/exp/end_to_end_exp_scripts/burstiness_exp.sh > /tmp/ae_09_burstiness_po2.log 2>&1

mkdir -p experiment_results_a30/phase3_1_burstiness/po2
rsync -az "$TARGET_HOST:~/Block/experiment_output/burstiness_po2/" experiment_results_a30/phase3_1_burstiness/po2/

# Pass 2 — Fanout-est (N=12, same scheduler, estimated lengths)
# Override N_SELECTED and dir by wrapping the script with SCHEDULER_NAME only covering Block-Fanout
echo "--- Pass 2: Fanout-est (N=12) ---"
# Fanout-est uses the same scheduler but N=12 (fanout) — run a targeted copy of the script.
# We use the fanout_est variant (already in repo for this purpose):
if [ -f block/exp/end_to_end_exp_scripts/ablation/burstiness_fanout_est_exp.sh ]; then
    echo "NOTE: ablation/burstiness_fanout_est_exp.sh is quarantined (known broken wiring in _fanout_est variants)."
    echo "      Running via manual loop instead."
fi

# Manual Fanout-est loop (fresh deploy Fanout N=12, then 4 burst probes at QPS=32)
MODEL="meta-llama/Llama-2-7b-hf"
DATASET_PATH="~/Block/data/trace_data/sharegpt/generate/llama"
for burst in 0.25 0.5 1.0 2.0; do
    echo "  [fanout-est] burst=$burst"
    sh block/exp/reset.sh
    sleep 30
    nohup sh block/exp/run_exp_vllm.sh 48 $MODEL false 0 4096 true 1 512 > /dev/null 2>&1 &
    sleep 90
    for s in $(seq 1 7); do
        nohup sh block/exp/run_exp_predictor_${s}.sh block/config/llama_config.json min_new_request_latency true 48 true 16 main 0 1000 false > /dev/null 2>&1 &
    done
    sleep 10
    for s in $(seq 8 16); do
        nohup sh block/exp/run_exp_predictor_${s}.sh block/config/llama_config.json min_new_request_latency true 48 true 16 main 0 1000 false > /dev/null 2>&1 &
    done
    sleep 60
    bash block/exp/end_to_end_exp_scripts/a30_main/verify_predictors.sh block/config/llama_config.json min_new_request_latency 48 16 1000 || true

    ssh -n "$TARGET_HOST" "pkill -f global_scheduler/api_server" 2>&1 || true
    sleep 5
    nohup sh block/exp/run_exp_global_scheduler.sh "$TARGET_HOST" 12 12 min_new_request_latency block/config/host_configs.json 1 16 0.000 1800 1000 12 0 false > /dev/null 2>&1 &
    sleep 20

    OUTPUT_DIR="burstiness_fanout/sharegpt/min_new_request_latency/qps_32_burst_${burst}"
    parallel-ssh -i -t 0 --host "$TARGET_HOST" "cd Block && export PYTHONPATH=. && export HF_TOKEN= && python block/benchmark/benchmark_serving.py --ip_ports 127.0.0.1:8200 --tokenizer $MODEL --num_sampled_requests 10000 --dataset_type sharegpt --dataset_path $DATASET_PATH --qps 32 --backend block --log_filename benchmark.log --output_dir $OUTPUT_DIR --data_start_index 0 --trust_remote_code --max_request_len 4096 --timeout_in_seconds 1800 --use_estimated_response_lens --distribution gamma --burstiness $burst" > /dev/null 2>&1
    sleep 5
done

mkdir -p experiment_results_a30/phase3_1_burstiness/fanout
rsync -az "$TARGET_HOST:~/Block/experiment_output/burstiness_fanout/" experiment_results_a30/phase3_1_burstiness/fanout/

total=$(find experiment_results_a30/phase3_1_burstiness -name benchmark_all_metrics.npz | wc -l)
echo "[sync] phase3_1_burstiness: $total NPZs total (expected ~12: 4 burst × 3 schedulers)"
echo "=== 09_burstiness COMPLETE ==="
