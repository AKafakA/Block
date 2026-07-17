#!/bin/bash
# 08_n_ablation.sh — Sec 6.4 + 6.6: N-tunable ablation
#
# Phase 2 (Sec 6.4): N ∈ {4, 6, 8} × QPS=30 single-point ablation.
# Phase 7b (Sec 6.6 N-tunable): Po4-est + Po8-est capacity refinement.
#
# Capacity-search semantics for Po4/Po8:
#  1. Integer probes around seed=32 (Po2-est cap+0.5, rounded up).
#  2. When SLO-crossing bucket found (e.g. [31, 32]), refine within at 0.1 QPS.
#  3. Default: 9.X/10.X early-stop bands (what this campaign did for time).
#
# For full paper rigor, the IDEAL procedure is:
#  - Integer bracket → sweep ALL 0.1-step floats within the bucket
#    (lo.0, lo.1, ..., hi.0) → report capacity at highest QPS with TTFT P99 < SLO.
#  - Gives a dense TTFT-vs-QPS curve, not just a single capacity point.
#
# Total time: ~1h (early-stop) or ~2.5h (full_sweep).

set -u
REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$REPO_ROOT"

echo "=== 08_n_ablation: Sec 6.4 + 6.6 ==="
date -u +%Y-%m-%dT%H:%M:%SZ
TARGET_HOST="$(head -1 block/config/hosts)"

# Phase 2: N=4,6,8 at fixed QPS=30 (~30 min)
echo "--- Phase 2: N ∈ {4, 6, 8} × QPS=30 ---"
nohup sh block/exp/end_to_end_exp_scripts/ablation/n_ablation_exp.sh > /tmp/ae_08_phase2.log 2>&1
mkdir -p experiment_results_a30/phase2_n_ablation
rsync -az "$TARGET_HOST:~/Block/experiment_output/n_ablation/" experiment_results_a30/phase2_n_ablation/
echo "[sync] phase2_n_ablation: $(find experiment_results_a30/phase2_n_ablation -name benchmark_all_metrics.npz | wc -l) NPZs (expected 3)"

# Phase 7b: Po4 + Po8 capacity refinement at 0.1 QPS resolution
echo "--- Phase 7b: Po4 + Po8 capacity refinement ---"
nohup sh block/exp/end_to_end_exp_scripts/a30_main/po4_then_po8_capacity.sh > /tmp/ae_03_phase7b.log 2>&1

mkdir -p experiment_results_a30/phase7_po4po8
rsync -az "$TARGET_HOST:~/Block/experiment_output/po4_capacity/" experiment_results_a30/phase7_po4po8/po4/
rsync -az "$TARGET_HOST:~/Block/experiment_output/po8_capacity/" experiment_results_a30/phase7_po4po8/po8/
echo "[sync] phase7_po4po8: $(find experiment_results_a30/phase7_po4po8 -name benchmark_all_metrics.npz | wc -l) NPZs"

echo "=== 08_n_ablation COMPLETE ==="
