#!/bin/bash
# 15_a100_block.sh — Sec 6.8: A100 Llama-2-70B Block sweeps
# 4 configs: Po2+CP, Fanout+CP, Po2-noCP, Fanout-noCP × 6 QPS = 24 runs
# CRITICAL: A100 run_benchmark.sh writes to FIXED PATH. Sync NPZ per-config IMMEDIATELY.
# ~9h total.

set -u
REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$REPO_ROOT"

echo "=== 15_a100_block: Sec 6.8 A100 Block sweeps ==="
date -u +%Y-%m-%dT%H:%M:%SZ

A100_HOSTS="block/config/a100_hosts"
A100_HEAD=$(head -1 "$A100_HOSTS")

# Verify run_benchmark.sh has --use_estimated_response_lens
if ! grep -q "use_estimated_response_lens" block/exp/end_to_end_exp_scripts/a100_supplementary/run_benchmark.sh; then
    echo "FAIL: A100 run_benchmark.sh missing --use_estimated_response_lens — would use oracle (cheating)"
    exit 1
fi
echo "[ok] --use_estimated_response_lens present in run_benchmark.sh"

run_config() {
    local config_name=$1
    local n=$2
    local cp_flag=$3  # "true" or "false"
    local log=$4

    echo "--- Config: $config_name (N=$n, chunked_prefill=$cp_flag) ---"
    # Deploy
    sh block/exp/end_to_end_exp_scripts/a100_supplementary/deploy_block.sh "$cp_flag" "$n"
    sleep 30

    # Sweep
    nohup sh block/exp/end_to_end_exp_scripts/a100_supplementary/run_benchmark.sh \
        sweep block 10000 "16 20 24 28 32 36" > "$log" 2>&1

    # CRITICAL: sync the per-QPS summary .txt files the Fig 8 plotter parses,
    # IMMEDIATELY before the next config overwrites the cluster-side output dir.
    # plot_llumnix_aggregate.py reads block_qps*/block_qps*_logs_logs.txt — that
    # is all Fig 8 needs. Per-request NPZ arrays are not collected (not used).
    mkdir -p experiment_results_a100/phase57_block/$config_name
    rsync -az --include='*/' --include='*_logs_logs.txt' --exclude='*' \
        "$A100_HEAD:~/Block/experiment_output/benchmark_output/block_sweep/" \
        experiment_results_a100/phase57_block/$config_name/

    n_txt=$(find experiment_results_a100/phase57_block/$config_name -name '*_logs_logs.txt' | wc -l)
    if [ "$n_txt" -lt "6" ]; then
        echo "FAIL: $config_name only has $n_txt summary txts (expected 6) — DO NOT advance to next config"
        exit 1
    fi
    echo "[sync ok] $config_name: $n_txt summary txts"
}

run_config "po2_cp"      2 true  /tmp/ae_10_a100_po2_cp.log
run_config "fanout_cp"   8 true  /tmp/ae_10_a100_fanout_cp.log
run_config "po2_nocp"    2 false /tmp/ae_10_a100_po2_nocp.log
run_config "fanout_nocp" 8 false /tmp/ae_10_a100_fanout_nocp.log

echo "=== 15_a100_block COMPLETE ==="
echo "All 4 configs synced (summary txts only) to:"
echo "  experiment_results_a100/phase57_block/{po2_cp, fanout_cp, po2_nocp, fanout_nocp}/block_qps*/*_logs_logs.txt"
