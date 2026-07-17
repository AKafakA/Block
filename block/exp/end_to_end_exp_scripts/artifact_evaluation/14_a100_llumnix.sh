#!/bin/bash
# 14_a100_llumnix.sh — Sec 6.8: A100 Llama-2-70B Llumnix sweep
# Uses 2× CloudLab d8545 A100 nodes (4× A100-40GB SXM4 each, TP=4)
# ~3h.

set -u
REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$REPO_ROOT"

echo "=== 14_a100_llumnix: Sec 6.8 A100 Llumnix baseline ==="
date -u +%Y-%m-%dT%H:%M:%SZ

# A100 host file (different from A30)
A100_HOSTS="block/config/a100_hosts"
if [ ! -f "$A100_HOSTS" ]; then
    echo "FAIL: $A100_HOSTS missing — populate with 2 A100 hostnames"
    exit 1
fi

# Deploy Llumnix on A100 nodes (uses setup_llumnix.sh + launching scheduler)
echo "[deploy] Llumnix on A100"
sh block/exp/setup_llumnix.sh

# Run sweep
echo "[run] Llumnix sweep across 21 QPS levels (16-36)"
nohup sh block/exp/end_to_end_exp_scripts/a100_supplementary/run_benchmark.sh \
    sweep llumnix 10000 "16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36" \
    > /tmp/ae_09_a100_llumnix.log 2>&1

A100_HEAD=$(head -1 "$A100_HOSTS")
mkdir -p experiment_results_a100/llumnix_sweep
# Sync ONLY the per-QPS summary .txt files the Fig 8 plotter parses
# (plot_llumnix_aggregate.py reads llumnix_qps*/llumnix_qps*_logs_logs.txt).
# Per-request NPZ arrays are excluded — not used by Fig 8.
rsync -az --include='*/' --include='*_logs_logs.txt' --exclude='*' \
    "$A100_HEAD:~/Block/experiment_output/benchmark_output/llumnix_sweep/" \
    experiment_results_a100/llumnix_sweep/
n=$(find experiment_results_a100/llumnix_sweep -name '*_logs_logs.txt' | wc -l)
echo "[sync] llumnix_sweep: $n summary txts (expected 21, QPS 16-36)"
if [ "$n" -lt 21 ]; then
    echo "WARN: expected 21 summary txts, got $n"
fi

echo "=== 14_a100_llumnix COMPLETE ==="
