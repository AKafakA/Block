#!/bin/bash
# 07_main_sweep_a30.sh — Section 6.3: Main TTFT/throughput sweep
# Phase 1.1 (6 schedulers × QPS 20-36) + Phase 1.2 (Po2 oracle+est × QPS 20-36)
# Total ~26h wall clock.

set -u
REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$REPO_ROOT"

echo "=== 07_main_sweep_a30: Section 6.3 main sweep ==="
date -u +%Y-%m-%dT%H:%M:%SZ
TARGET_HOST="$(head -1 block/config/hosts)"

# Phase 1.2 first (Po2 — faster, ~6h)
echo "--- Phase 1.2: Po2 (oracle+est × QPS 20-36) ---"
nohup sh block/exp/end_to_end_exp_scripts/a30_main/po2_main_experiment.sh > /tmp/ae_02_phase12.log 2>&1
echo "[done] Phase 1.2 finished at $(date -u +%H:%M:%SZ)"

# Sync Phase 1.2
mkdir -p experiment_results_a30/phase12_po2
rsync -az "$TARGET_HOST:~/Block/experiment_output/main_po2/" experiment_results_a30/phase12_po2/
n=$(find experiment_results_a30/phase12_po2 -name benchmark_all_metrics.npz | wc -l)
echo "[sync] phase12_po2: $n NPZs (expected 34)"

# Phase 1.1 next (~20h)
echo "--- Phase 1.1: 6 schedulers × QPS 20-36 ---"
nohup sh block/exp/end_to_end_exp_scripts/a30_main/main_experiment.sh > /tmp/ae_02_phase11.log 2>&1
echo "[done] Phase 1.1 finished at $(date -u +%H:%M:%SZ)"

# Sync Phase 1.1
mkdir -p experiment_results_a30/phase11_main
rsync -az "$TARGET_HOST:~/Block/experiment_output/main/" experiment_results_a30/phase11_main/
n=$(find experiment_results_a30/phase11_main -name benchmark_all_metrics.npz | wc -l)
echo "[sync] phase11_main: $n NPZs (expected ~119)"

echo "=== 07_main_sweep COMPLETE ==="
