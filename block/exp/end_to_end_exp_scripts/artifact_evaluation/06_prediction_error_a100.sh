#!/bin/bash
# 06_prediction_error_a100.sh — Sec 6.2: Latency Prediction Metrics (A100 Llama-70B)
#
# A100 variant of 0a. Uses 2× A100 nodes (TP=4, Llama-2-70B) and the
# A100 orchestration harness (run_benchmark.sh). Otherwise same idea:
# PROFILING_SAMPLE_RATE=0.01, oracle lengths, measure predicted-vs-actual
# execution time error across a QPS sweep.
#
# Time: ~1.5h (5 QPS × ~15 min on 70B)

set -u
REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$REPO_ROOT"

echo "=== 06_prediction_error_a100: Sec 6.2 Latency Prediction Metrics (A100 Llama-70B) ==="
date -u +%Y-%m-%dT%H:%M:%SZ

A100_HOSTS="block/config/a100_hosts"
A100_HEAD=$(head -1 "$A100_HOSTS" 2>/dev/null)
if [ -z "$A100_HEAD" ]; then
    echo "FAIL: $A100_HOSTS missing — populate with 2 A100 hostnames"
    exit 1
fi

# Sanity: benchmark has --use_estimated_response_lens only when set explicitly;
# for prediction-error experiment we WANT oracle lengths so the predictor's
# execution-time prediction is isolated from length-predictor noise.
# (i.e. DO NOT pass --use_estimated_response_lens here)

# Deploy Fanout N=12 (oracle length) with profiling sampling on
# The A100 deploy_block.sh + run_benchmark.sh pair is configured via positional args;
# here we pass profiling_rate=0.01 through the scheduler env/arg if supported.
# If the existing A100 deploy doesn't accept a rate, manually start the scheduler with
#   --profiling_sampling_rate 0.01
# and then call run_benchmark.sh sweep for "20 24 28 32 36".

echo "[deploy] Fanout N=12 with profiling sampling on"
sh block/exp/end_to_end_exp_scripts/a100_supplementary/deploy_block.sh true 12
sleep 30

# Restart scheduler with profiling_sampling_rate=0.01
ssh -n "$A100_HEAD" "pkill -f global_scheduler/api_server" || true
sleep 5
ssh -n "$A100_HEAD" "cd Block && export PYTHONPATH=. && (nohup python block/global_scheduler/api_server.py \
    --config_path block/config/a100_host_configs.json \
    --metrics_type min_new_request_latency \
    --num_query_predictor 12 --num_required_predictor 12 \
    --workers 1 --num_predictor_ports 16 \
    --profiling_sampling_rate 0.01 \
    --predictor_timeout 1000 --backend_timeout 1800 \
    --initial_available_instance 2 --max_slo_in_seconds 0 \
    > experiment_output/logs/global_scheduler.log 2>&1 &)" || true
sleep 20

# Run sweep (oracle lengths — no --use_estimated_response_lens flag)
echo "[run] QPS sweep 20 24 28 32 36"
nohup sh block/exp/end_to_end_exp_scripts/a100_supplementary/run_benchmark.sh \
    sweep block 10000 "20 24 28 32 36" \
    > /tmp/ae_0b_prediction_a100.log 2>&1

mkdir -p experiment_results_a100/phase_prediction_a100
# CRITICAL: A100 run_benchmark.sh writes to fixed path; sync immediately
rsync -az "$A100_HEAD:~/Block/experiment_output/benchmark_output/block_sweep/" \
    experiment_results_a100/phase_prediction_a100/
n=$(find experiment_results_a100/phase_prediction_a100 -name benchmark_all_metrics.npz | wc -l)
echo "[sync] phase_prediction_a100: $n NPZs (expected 5)"

echo "--- Prediction accuracy summary (A100 Llama-2-70B) ---"
python3 <<'PY'
import numpy as np, glob
for f in sorted(glob.glob('experiment_results_a100/phase_prediction_a100/**/benchmark_all_metrics.npz', recursive=True)):
    d = np.load(f, allow_pickle=True)
    qps = f.split('qps')[1].split('_')[0].strip('/')
    acc = d.get('sampled_predict_accuracies')
    err = d.get('sampled_mean_error_ratios')
    if acc is None or err is None:
        print(f"  QPS={qps}: fields missing — profiling_sample_rate was 0?")
        continue
    acc = np.array(acc); err = np.array(err)
    if len(acc) == 0:
        print(f"  QPS={qps}: no samples"); continue
    print(f"  QPS={qps}: predict_accuracy={float(np.mean(acc)):.3f}, mean_error_ratio={float(np.mean(err))*100:.1f}%, n={len(acc)}")
PY

echo "=== 06_prediction_error_a100 COMPLETE ==="
