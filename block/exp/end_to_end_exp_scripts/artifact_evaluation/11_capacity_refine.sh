#!/bin/bash
# 11_capacity_refine.sh — Float-resolution capacity table
#
# Phase 4.1: 5 schedulers × capacity search (Po2-est/oracle, Fanout-est/oracle, Llumnix).
#
# Capacity-search semantics:
#  1. Integer sweep first (QPS 20..36 @ 1-QPS steps) — from Phase 1.1/1.2 main sweep.
#  2. Find integer bucket [lo, hi] where TTFT P99 crosses SLO=10s.
#  3. Refine within the bucket at 0.1 QPS resolution.
#
# Two modes for step 3 (env MODE=...):
#   early_stop (default, ~3h): binary probes until TTFT in 9.X (9-10s) or
#     10.X (10-11s) band, then stop. Saves ~50% of probes.
#   full_sweep (~6-8h): probe ALL 10 float points within [lo, hi]
#     (lo.0, lo.1, ..., hi.0). Gives the dense TTFT-vs-QPS curve.
#
# Paper uses early_stop for the capacity table values; use full_sweep
# if reviewer requests dense curves.

set -u
REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$REPO_ROOT"

echo "=== 11_capacity_refine: float capacity table ==="
date -u +%Y-%m-%dT%H:%M:%SZ
TARGET_HOST="$(head -1 block/config/hosts)"

# Use float_capacity_search.sh (created during campaign)
if [ ! -f block/exp/end_to_end_exp_scripts/a30_main/float_capacity_search.sh ]; then
    echo "FAIL: float_capacity_search.sh missing"
    exit 1
fi

nohup sh block/exp/end_to_end_exp_scripts/a30_main/float_capacity_search.sh > /tmp/ae_06_float.log 2>&1
mkdir -p experiment_results_a30/phase4_1_float
rsync -az "$TARGET_HOST:~/Block/experiment_output/main_float/" experiment_results_a30/phase4_1_float/
n=$(find experiment_results_a30/phase4_1_float -name benchmark_all_metrics.npz | wc -l)
echo "[sync] phase4_1_float: $n NPZs"

# Extract capacity values
echo "--- Capacity table ---"
grep -E "FINAL.*capacity|9\.X.*capacity|10\.X.*capacity" /tmp/ae_06_float.log | tail -20

echo "=== 11_capacity_refine COMPLETE ==="
