#!/bin/bash
# 04_warmup_llama.sh — Train Llama predictor cache (one-time).
# Avoids OOM during 16-predictor concurrent first launch.

set -u
REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$REPO_ROOT"

echo "=== 04_warmup_llama: train Llama-2-7b predictor cache ==="
date -u +%Y-%m-%dT%H:%M:%SZ

LOG=/tmp/ae_01_warmup_llama.log

sh block/exp/end_to_end_exp_scripts/warmup.sh "meta-llama/Llama-2-7b-hf" 2>&1 | tee "$LOG"

# Verify cache populated
echo "[verify] cache pkl count per node"
all_ok=1
for host in $(cat block/config/hosts); do
    short=$(echo $host | grep -oP 'd7525-10s\K\d+')
    n=$(ssh -n -o ConnectTimeout=5 "$host" "ls ~/Block/cache/*.pkl 2>/dev/null | wc -l")
    if [ "$n" -lt "50" ]; then
        echo "  $short: $n pkls — INSUFFICIENT (expected >50)"
        all_ok=0
    else
        echo "  $short: $n pkls ✓"
    fi
done

if [ "$all_ok" = "1" ]; then
    echo "=== 04_warmup PASSED. Ready for 07_main_sweep_a30.sh ==="
else
    echo "FAIL: cache incomplete on some nodes. Re-run warmup or investigate."
    exit 1
fi
