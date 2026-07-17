#!/bin/bash
# 01_profile_vidur_a30.sh — Vidur hardware profiling for A30 (Llama-2-7B)
#
# Profiles execution time of attention, MLP, collectives, and CPU overhead
# for Llama-2-7B on an A30 GPU. Produces CSV tables under data/profiling/
# that the runtime predictor service loads to estimate per-request latency.
#
# ONE-TIME per device class. This is Vidur simulator infrastructure (not
# part of Block's contribution), but included here so the artifact is
# fully self-contained.
#
# Run on ONE A30 node with 1 GPU free. Single-GPU (no TP on A30 7B).
# Time: ~3-4h.

set -eu
REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$REPO_ROOT"

MODEL="meta-llama/Llama-2-7b-hf"
DEVICE_NAME="a30"
NETWORK_DEVICE="a30_single_gpu"
NUM_GPUS=1
TP_SIZES="1"
MAX_TOKENS=4096
OUTPUT_BASE="./data/profiling"

echo "=== 01_profile_vidur_a30: Sec 6.2 prereq (Vidur hardware profile, A30 Llama-7B) ==="
date -u +%Y-%m-%dT%H:%M:%SZ

# Verify GPU
nvidia-smi -L | head -1 | grep -qi "A30" || echo "WARN: not an A30 node; proceeding anyway"

export PYTHONPATH=.
mkdir -p "${OUTPUT_BASE}/compute/${DEVICE_NAME}/${MODEL}" "${OUTPUT_BASE}/network/${NETWORK_DEVICE}" "${OUTPUT_BASE}/cpu_overhead/${NETWORK_DEVICE}/${MODEL}"

# Phase 1: MLP
echo "--- Phase 1: MLP kernel profiling ---"
python vidur/profiling/mlp/main.py \
    --models "$MODEL" \
    --num_gpus $NUM_GPUS \
    --num_tensor_parallel_workers $TP_SIZES \
    --max_tokens $MAX_TOKENS \
    --output_dir "./profiling_outputs_tmp_a30/mlp" 2>&1 | tail -10
cp ./profiling_outputs_tmp_a30/mlp/*/${MODEL}/mlp.csv "${OUTPUT_BASE}/compute/${DEVICE_NAME}/${MODEL}/mlp.csv"

# Phase 2: Attention
echo "--- Phase 2: Attention kernel profiling ---"
python vidur/profiling/attention/main.py \
    --models "$MODEL" \
    --num_gpus $NUM_GPUS \
    --num_tensor_parallel_workers $TP_SIZES \
    --max_tokens $MAX_TOKENS \
    --output_dir "./profiling_outputs_tmp_a30/attention" 2>&1 | tail -10
cp ./profiling_outputs_tmp_a30/attention/*/${MODEL}/attention.csv "${OUTPUT_BASE}/compute/${DEVICE_NAME}/${MODEL}/attention.csv"

# Phase 3: Collectives (on A30 single-GPU, still need all_reduce/send_recv tables for multi-instance scheduling)
echo "--- Phase 3: Collectives profiling ---"
python vidur/profiling/collectives/main.py \
    --num_workers_list "1" \
    --devices_per_node $NUM_GPUS \
    --output_dir "./profiling_outputs_tmp_a30/collectives" 2>&1 | tail -10 || echo "WARN: collectives may be N/A on single-GPU node"
if [ -f ./profiling_outputs_tmp_a30/collectives/*/all_reduce.csv ]; then
    cp ./profiling_outputs_tmp_a30/collectives/*/all_reduce.csv "${OUTPUT_BASE}/network/${NETWORK_DEVICE}/all_reduce.csv"
    cp ./profiling_outputs_tmp_a30/collectives/*/send_recv.csv "${OUTPUT_BASE}/network/${NETWORK_DEVICE}/send_recv.csv"
fi

# Phase 4: CPU overhead
echo "--- Phase 4: CPU overhead profiling ---"
python vidur/profiling/cpu_overhead/main.py \
    --models "$MODEL" \
    --num_tensor_parallel_workers $TP_SIZES \
    --output_dir "./profiling_outputs_tmp_a30/cpu_overhead" 2>&1 | tail -10 || echo "WARN: cpu_overhead may require specific deps"
if [ -f ./profiling_outputs_tmp_a30/cpu_overhead/*/${MODEL}/cpu_overheads.csv ]; then
    cp ./profiling_outputs_tmp_a30/cpu_overhead/*/${MODEL}/cpu_overheads.csv "${OUTPUT_BASE}/cpu_overhead/${NETWORK_DEVICE}/${MODEL}/cpu_overheads.csv"
fi

echo "--- Output summary ---"
find "$OUTPUT_BASE" -name "*.csv" -path "*${DEVICE_NAME}*" -o -name "*.csv" -path "*${NETWORK_DEVICE}*" | head -10

echo "=== 01_profile_vidur_a30 COMPLETE ==="
echo "Sync these to all 12 A30 cluster nodes before running serving experiments:"
echo "  parallel-scp -h block/config/hosts -r data/profiling Block/data/"
