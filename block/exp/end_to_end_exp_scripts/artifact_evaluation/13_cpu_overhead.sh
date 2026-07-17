#!/bin/bash
# 13_cpu_overhead.sh — Sec 6.7: CPU overhead measurement
#
# Runs BOTH Po2 (N=2) and Fanout (N=12) × QPS {20, 24, 28, 32, 36} with
# --enable_cpu_tracking so the artifact directly reproduces the CPU vs Fanout
# comparison figure. Without the Fanout pass, reviewer has to import
# revision priors which is less self-contained.
#
# Requires CPU patches across predictor + scheduler + benchmark (verified below).
# ~1.5h total (45 min × 2).
#
# Env: RUN_BOTH=false to skip Fanout pass (Po2-only, ~45 min)

set -u
REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$REPO_ROOT"

RUN_BOTH="${RUN_BOTH:-true}"

echo "=== 13_cpu_overhead: Sec 6.7 (Po2 + Fanout CPU profile) ==="
date -u +%Y-%m-%dT%H:%M:%SZ
TARGET_HOST="$(head -1 block/config/hosts)"

# Verify CPU patches
echo "[check] CPU pipeline patches"
grep -q "metric\[\"cpu_percent\"\]" block/predictor/api_server.py || { echo "FAIL: predictor patch missing"; exit 1; }
grep -q "single_metric\['mean_cpu_percent'\]" block/global_scheduler/api_server.py || { echo "FAIL: scheduler patch missing"; exit 1; }
grep -q "self._cpu_percents = \[\]" block/benchmark/benchmark_serving.py || { echo "FAIL: benchmark patch missing"; exit 1; }
echo "[ok] all patches present"

# Pass 1 — Po2 (N=2) × 5 QPS with --enable_cpu_tracking
echo "--- Pass 1: Po2-est CPU profile ---"
N_SELECTED=2 OUTPUT_PREFIX=cpu_tracker_po2_v2 nohup sh block/exp/end_to_end_exp_scripts/a30_main/cpu_tracker_full.sh > /tmp/ae_13_cpu_po2.log 2>&1

mkdir -p experiment_results_a30/phase7_cpu_tracker/po2
rsync -az "$TARGET_HOST:~/Block/experiment_output/cpu_tracker_po2_v2/" experiment_results_a30/phase7_cpu_tracker/po2/

# Pass 2 — Fanout (N=12) × 5 QPS with --enable_cpu_tracking
if [ "$RUN_BOTH" = "true" ]; then
    echo "--- Pass 2: Fanout-est CPU profile ---"
    N_SELECTED=12 OUTPUT_PREFIX=cpu_tracker_fanout nohup sh block/exp/end_to_end_exp_scripts/a30_main/cpu_tracker_full.sh > /tmp/ae_13_cpu_fanout.log 2>&1
    mkdir -p experiment_results_a30/phase7_cpu_tracker/fanout
    rsync -az "$TARGET_HOST:~/Block/experiment_output/cpu_tracker_fanout/" experiment_results_a30/phase7_cpu_tracker/fanout/
else
    echo "--- Skipping Fanout CPU pass (RUN_BOTH=false; will need to pull from revision priors) ---"
fi

n=$(find experiment_results_a30/phase7_cpu_tracker -name benchmark_all_metrics.npz | wc -l)
expected=5
[ "$RUN_BOTH" = "true" ] && expected=10
echo "[sync] phase7_cpu_tracker: $n NPZs (expected $expected)"

# Verify CPU fields in NPZ
echo "--- CPU verification (first cell) ---"
python3 -c "
import numpy as np, glob
files = sorted(glob.glob('experiment_results_a30/phase7_cpu_tracker/**/benchmark_all_metrics.npz', recursive=True))
if not files:
    print('FAIL: no NPZs found'); exit(1)
d = np.load(files[0], allow_pickle=True)
keys = list(d.keys())
if 'cpu_percents' not in keys or len(d['cpu_percents']) == 0:
    print(f'FAIL: cpu_percents missing or empty. NPZ keys: {keys}')
    exit(1)
cp = d['cpu_percents']
print(f'OK: cpu_percents present, n={len(cp)}, mean={float(np.mean(cp)):.1f}%, max={float(np.max(cp)):.1f}%, cores={int(d[\"cpu_cores\"])}')
" || { echo "FAIL: CPU data not in NPZ — patch did not propagate. Check sync."; exit 1; }

# Print the campaign log lines for paper reference
echo "--- All Po2 cells ---"
grep -E "cpu_tracker po2_est QPS=" /tmp/ae_13_cpu_po2.log | sort -u
if [ "$RUN_BOTH" = "true" ]; then
    echo "--- All Fanout cells ---"
    grep -E "cpu_tracker.*QPS=" /tmp/ae_13_cpu_fanout.log | sort -u
fi

echo "=== 13_cpu_overhead COMPLETE ==="
