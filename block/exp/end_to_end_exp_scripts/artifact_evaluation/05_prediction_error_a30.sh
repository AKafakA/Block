#!/bin/bash
# 05_prediction_error_a30.sh — Sec 6.2: Latency Prediction Metrics (A30)
#
# Measures predictor accuracy: per-request sampled_predict_accuracy and
# sampled_mean_error_ratio by comparing predicted execution time vs actual
# serving time. Sample rate 1% of requests.
#
# Underlying script: block/exp/end_to_end_exp_scripts/prediction_experiment.sh
#   - scheduler min_new_request_latency, N=12 (Fanout, full broadcast)
#   - oracle lengths (USE_LENGTH_ESTIMATION=false) — isolates predictor
#     accuracy, removes RoBERTa length-predictor noise
#   - PROFILING_SAMPLE_RATE=0.01, USE_FOR_PROFILING_ONLY=true
#   - QPS {20, 24, 28, 32, 36} — varies load to check how prediction
#     accuracy changes with system load
#
# Time: ~1h (5 QPS × ~10 min)

set -u
REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$REPO_ROOT"

echo "=== 05_prediction_error_a30: Sec 6.2 Latency Prediction Metrics (A30) ==="
date -u +%Y-%m-%dT%H:%M:%SZ
TARGET_HOST="$(head -1 block/config/hosts)"

# Verify predictors deployed 16/16 per node (hard gate)
bash block/exp/end_to_end_exp_scripts/artifact_evaluation/util_predictor_health.sh || {
    echo "FAIL: predictor health check failed — deploy first with 02_main_sweep_a30.sh or warmup"
    exit 1
}

nohup sh block/exp/end_to_end_exp_scripts/prediction_experiment.sh > /tmp/ae_0a_prediction_a30.log 2>&1
mkdir -p experiment_results_a30/phase_prediction_a30
rsync -az "$TARGET_HOST:~/Block/experiment_output/prediction/" experiment_results_a30/phase_prediction_a30/
n=$(find experiment_results_a30/phase_prediction_a30 -name benchmark_all_metrics.npz | wc -l)
echo "[sync] phase_prediction_a30: $n NPZs (expected 5)"

# Extract predict-accuracy summary for paper
echo "--- Prediction accuracy summary (A30 Llama-2-7B) ---"
python3 <<'PY'
import numpy as np, glob
for f in sorted(glob.glob('experiment_results_a30/phase_prediction_a30/**/benchmark_all_metrics.npz', recursive=True)):
    d = np.load(f, allow_pickle=True)
    qps = f.split('qps_')[1].split('_')[0]
    acc = d.get('sampled_predict_accuracies')
    err = d.get('sampled_mean_error_ratios')
    if acc is None or err is None:
        print(f"  QPS={qps}: fields missing — profiling_sample_rate was 0?")
        continue
    acc = np.array(acc); err = np.array(err)
    if len(acc) == 0:
        print(f"  QPS={qps}: no samples collected"); continue
    print(f"  QPS={qps}: predict_accuracy mean={float(np.mean(acc)):.3f}, mean_error_ratio={float(np.mean(err))*100:.1f}%, n_samples={len(acc)}")
PY

echo "=== 05_prediction_error_a30 COMPLETE ==="
