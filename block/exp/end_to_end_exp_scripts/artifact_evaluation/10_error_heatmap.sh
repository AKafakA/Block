#!/bin/bash
# 10_error_heatmap.sh — Sec 6.6: Error injection heatmap
#
# Runs BOTH Po2 (N=2) and Fanout (N=12) × 15 (length_err, latency_err)
# cells @ QPS=32 for the Block-Pow2 paper AND the original Block paper.
# Fresh-deploy-per-cell. ~5h total.
#
# Env:
#   RUN_BOTH=true   (default) — runs Po2 pass then Fanout pass
#   RUN_BOTH=false  — runs only Po2 pass (~2.5h)

set -u
REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$REPO_ROOT"

RUN_BOTH="${RUN_BOTH:-true}"

echo "=== 10_error_heatmap: Sec 6.6 (Po2 + Fanout) ==="
date -u +%Y-%m-%dT%H:%M:%SZ
TARGET_HOST="$(head -1 block/config/hosts)"

# Verify fresh-deploy-per-cell
if ! grep -A 5 "for length_err\|for cell" block/exp/end_to_end_exp_scripts/error_heatmap_exp.sh | grep -q "reset.sh"; then
    echo "WARN: error_heatmap_exp.sh may not do fresh deploy per cell. Single-deploy gives ±4% noise."
fi

# Pass 1 — Po2 (N=2)
echo "--- Pass 1: Po2-est (N=2) × 15 cells ---"
N_SELECTED=2 OUTPUT_DIR_PREFIX=error_heatmap_po2 nohup sh block/exp/end_to_end_exp_scripts/error_heatmap_exp.sh > /tmp/ae_10_heatmap_po2.log 2>&1
mkdir -p experiment_results_a30/phase3_2_error_heatmap/po2
rsync -az "$TARGET_HOST:~/Block/experiment_output/error_heatmap_po2/" experiment_results_a30/phase3_2_error_heatmap/po2/

# Pass 2 — Fanout-est (N=12)
if [ "$RUN_BOTH" = "true" ]; then
    echo "--- Pass 2: Fanout-est (N=12) × 15 cells ---"
    N_SELECTED=12 OUTPUT_DIR_PREFIX=error_heatmap_fanout nohup sh block/exp/end_to_end_exp_scripts/error_heatmap_exp.sh > /tmp/ae_10_heatmap_fanout.log 2>&1
    mkdir -p experiment_results_a30/phase3_2_error_heatmap/fanout
    rsync -az "$TARGET_HOST:~/Block/experiment_output/error_heatmap_fanout/" experiment_results_a30/phase3_2_error_heatmap/fanout/
else
    echo "--- Skipping Fanout pass (RUN_BOTH=false) ---"
fi

total=$(find experiment_results_a30/phase3_2_error_heatmap -name benchmark_all_metrics.npz | wc -l)
expected=15
[ "$RUN_BOTH" = "true" ] && expected=30
echo "[sync] phase3_2_error_heatmap: $total NPZs (expected $expected)"
echo "=== 10_error_heatmap COMPLETE ==="
