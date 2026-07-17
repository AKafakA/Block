#!/bin/bash
# 02_profile_vidur_a100.sh — Vidur hardware profiling for A100-40GB (Llama-2-70B)
#
# Wraps the existing a100_40gb_profiling.sh which profiles MLP, attention,
# collectives, and CPU overhead for Llama-70B on A100-40GB across TP=1/2/4.
#
# ONE-TIME per device class. Run on ONE CloudLab d8545 node with 4 A100-40GB
# GPUs free. Time: ~5-6h.

set -eu
REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$REPO_ROOT"

echo "=== 02_profile_vidur_a100: Sec 6.2 prereq (Vidur hardware profile, A100 Llama-70B) ==="
date -u +%Y-%m-%dT%H:%M:%SZ

# Verify 4 A100s
ngpu=$(nvidia-smi -L | wc -l)
echo "[gpu] $ngpu GPUs detected"
if [ "$ngpu" != "4" ]; then
    echo "WARN: expected 4 A100-40GB GPUs; got $ngpu"
fi

# Delegate to existing profiling script
exec sh block/exp/end_to_end_exp_scripts/a100_40gb_profiling.sh
