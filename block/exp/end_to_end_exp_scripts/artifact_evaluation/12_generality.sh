#!/bin/bash
# 12_generality.sh — Sec 6.7: Generality (chunk2048, batch24, BurstGPT, Qwen)
#
# Per variant, all three schedulers (Po2-est, Po2-oracle, Llumnix-N12)
# run a **binary-search capacity refinement** (integer bracket + 0.1-step
# float refine, 9.X/10.X early-stop). Binary search is chosen because
# many (variant × scheduler) pairs would make full-sweep exceed lease budget.
#
# Per scheduler search: 3-8 probes total (1 seed + 1-4 integer steps + ≤4 float).
# Sub-phases run sequentially; total ~5h.

set -u
REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$REPO_ROOT"

echo "=== 12_generality: Sec 6.7 ==="
date -u +%Y-%m-%dT%H:%M:%SZ
TARGET_HOST="$(head -1 block/config/hosts)"

# Sub-phase A: chunk2048 + batch24 (~3h)
echo "--- 07A: chunk2048 + batch24 capacity float ---"
nohup sh block/exp/end_to_end_exp_scripts/a30_main/generality_float_simple.sh > /tmp/ae_07a_generality.log 2>&1
mkdir -p experiment_results_a30/phase4_2_generality
rsync -az "$TARGET_HOST:~/Block/experiment_output/generality_float/" experiment_results_a30/phase4_2_generality/

# Sub-phase B: BurstGPT (~1.5h)
echo "--- 07B: BurstGPT capacity (Po2-oracle + Llumnix) ---"
nohup sh block/exp/end_to_end_exp_scripts/a30_main/burstgpt_float_search.sh > /tmp/ae_07b_burstgpt.log 2>&1
mkdir -p experiment_results_a30/phase4_2_burstgpt
rsync -az "$TARGET_HOST:~/Block/experiment_output/generality_float/burstgpt/" experiment_results_a30/phase4_2_burstgpt/

# Sub-phase C: Qwen (~2h, includes warmup)
echo "--- 07C: Qwen2-7B (warmup + capacity for Po2-est, Po2-oracle, Llumnix-N12) ---"
echo "[warmup] training Qwen predictor cache (one-time)"
sh block/exp/end_to_end_exp_scripts/warmup.sh "Qwen/Qwen2-7B" 2>&1 | tail -5

echo "[run] qwen capacity search"
nohup sh block/exp/end_to_end_exp_scripts/a30_main/qwen_capacity_search.sh > /tmp/ae_07c_qwen.log 2>&1
mkdir -p experiment_results_a30/phase4_2_qwen
rsync -az "$TARGET_HOST:~/Block/experiment_output/generality_float/qwen/" experiment_results_a30/phase4_2_qwen/

echo "--- Capacity values ---"
grep -E "FINAL.*capacity|9\.X.*capacity|10\.X.*capacity" /tmp/ae_07a_generality.log /tmp/ae_07b_burstgpt.log /tmp/ae_07c_qwen.log | tail -20

# Note for paper: Qwen sharegpt yields 9963/10000 valid samples (tokenizer filter); uniform across schedulers.
echo "NOTE: Qwen runs have 9963/10000 samples (tokenizer filtering). Comparison is fair (uniform)."

echo "=== 12_generality COMPLETE ==="
