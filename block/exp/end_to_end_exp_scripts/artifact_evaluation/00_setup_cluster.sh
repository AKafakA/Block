#!/bin/bash
# 00_setup_cluster.sh — Verify cluster bring-up and code patches.
# Run after CloudLab nodes provisioned and setup.sh applied to all 12 nodes.
# This script does NOT install anything — it VERIFIES the setup is complete.

set -u

REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$REPO_ROOT"

echo "=== 00_setup_cluster: pre-flight verification ==="
date -u +%Y-%m-%dT%H:%M:%SZ

# 1. host file present
if [ ! -f block/config/hosts ]; then
    echo "FAIL: block/config/hosts missing — populate with 12 A30 hostnames"
    exit 1
fi
N_HOSTS=$(wc -l < block/config/hosts)
echo "[ok] block/config/hosts has $N_HOSTS hosts"
if [ "$N_HOSTS" != "12" ]; then echo "WARN: expected 12 A30 hosts"; fi

# 2. SSH reachability all nodes
echo "[check] SSH reachability"
unreachable=""
for host in $(cat block/config/hosts); do
    if ! ssh -n -o ConnectTimeout=5 -o BatchMode=yes "$host" "echo ok" >/dev/null 2>&1; then
        unreachable="$unreachable $host"
    fi
done
if [ -n "$unreachable" ]; then
    echo "FAIL: unreachable nodes:$unreachable"
    exit 1
fi
echo "[ok] all nodes reachable"

# 3. Python deps
echo "[check] Python deps on nodes"
parallel-ssh -i -t 10 -h block/config/hosts "python -c 'import vllm, torch, transformers, psutil; print(vllm.__version__, transformers.__version__)'" 2>&1 | grep -E "FAILURE|Error" && { echo "FAIL: Python deps missing on some nodes"; exit 1; }
echo "[ok] Python deps present"

# 4. Required code patches
echo "[check] CPU tracking patches in code"
grep -q "metric\[\"cpu_percent\"\]" block/predictor/api_server.py || { echo "FAIL: predictor cpu patch missing"; exit 1; }
grep -q "single_metric\['mean_cpu_percent'\]" block/global_scheduler/api_server.py || { echo "FAIL: scheduler cpu patch missing"; exit 1; }
grep -q "self._cpu_percents = \[\]" block/benchmark/benchmark_serving.py || { echo "FAIL: benchmark cpu patch missing"; exit 1; }
grep -q "timeout 30 ssh" block/exp/end_to_end_exp_scripts/a30_main/verify_predictors.sh || { echo "FAIL: verify_predictors timeout patch missing"; exit 1; }
echo "[ok] all patches present locally"

# 5. Sync patches to all 12 nodes
echo "[sync] pushing patched code to all nodes"
parallel-scp -h block/config/hosts block/predictor/api_server.py Block/block/predictor/api_server.py > /dev/null 2>&1
parallel-scp -h block/config/hosts block/global_scheduler/api_server.py Block/block/global_scheduler/api_server.py > /dev/null 2>&1
parallel-scp -h block/config/hosts block/benchmark/benchmark_serving.py Block/block/benchmark/benchmark_serving.py > /dev/null 2>&1
parallel-scp -h block/config/hosts block/exp/experiment.sh Block/block/exp/experiment.sh > /dev/null 2>&1
parallel-scp -h block/config/hosts block/exp/end_to_end_exp_scripts/a30_main/verify_predictors.sh Block/block/exp/end_to_end_exp_scripts/a30_main/verify_predictors.sh > /dev/null 2>&1
echo "[sync] done"

# 6. HF_TOKEN check
echo "[check] HF_TOKEN on master"
TARGET_HOST="$(head -1 block/config/hosts)"
ssh -n "$TARGET_HOST" "test -n \"\$HF_TOKEN\" || test -f ~/.hf_token" && echo "[ok] HF_TOKEN configured on master" || { echo "FAIL: set HF_TOKEN on master"; exit 1; }

# 7. Models accessible
echo "[check] Llama-2-7b cached on master"
ssh -n "$TARGET_HOST" "ls ~/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf 2>/dev/null" >/dev/null && echo "[ok] Llama-2-7b cached" || echo "WARN: Llama-2-7b not cached — first run will download"

echo
echo "=== 00_setup PASSED. Ready for 04_warmup_llama.sh ==="
