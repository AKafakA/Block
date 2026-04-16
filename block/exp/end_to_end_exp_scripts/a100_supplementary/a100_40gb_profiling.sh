#!/bin/bash
# A100-40GB (CloudLab d8545) Vidur Profiling Script
# Purpose: Profile Llama-70B execution times on A100-40GB for accurate simulator predictions
#
# This script must be run BEFORE the A100 Llama-70B experiment.
# Run on a CloudLab d8545 node with 4×A100-40GB GPUs.
#
# Prerequisites:
#   1. CloudLab d8545 node provisioned
#   2. Llama-2-70B model weights downloaded
#   3. Dependencies installed (ray, pandas, torch, etc.)

set -e

# Configuration
MODEL="meta-llama/Llama-2-70b-hf"
DEVICE_NAME="a100_40gb"
NETWORK_DEVICE="a100_40gb_pairwise_nvlink"
NUM_GPUS=4
TENSOR_PARALLEL_SIZES="1 2 4"
MAX_TOKENS=4096
OUTPUT_BASE="./data/profiling"

echo "=============================================="
echo "A100-40GB Vidur Profiling for Llama-70B"
echo "=============================================="
echo ""
echo "Device: $DEVICE_NAME"
echo "Model: $MODEL"
echo "GPUs: $NUM_GPUS"
echo ""

# Create output directories
mkdir -p ${OUTPUT_BASE}/compute/${DEVICE_NAME}/${MODEL}
mkdir -p ${OUTPUT_BASE}/network/${NETWORK_DEVICE}

# Phase 1: MLP Profiling
echo "=== Phase 1: MLP Kernel Profiling ==="
cd /auto/homes/wd312/Code/llm/Block
export PYTHONPATH=.

python vidur/profiling/mlp/main.py \
    --models "${MODEL}" \
    --num_gpus ${NUM_GPUS} \
    --num_tensor_parallel_workers ${TENSOR_PARALLEL_SIZES} \
    --max_tokens ${MAX_TOKENS} \
    --output_dir "./profiling_outputs_tmp/mlp"

# Copy results to standard location
cp ./profiling_outputs_tmp/mlp/*/${MODEL}/mlp.csv \
   ${OUTPUT_BASE}/compute/${DEVICE_NAME}/${MODEL}/mlp.csv
echo "MLP profiling complete: ${OUTPUT_BASE}/compute/${DEVICE_NAME}/${MODEL}/mlp.csv"

# Phase 2: Attention Profiling
echo ""
echo "=== Phase 2: Attention Kernel Profiling ==="
python vidur/profiling/attention/main.py \
    --models "${MODEL}" \
    --num_gpus ${NUM_GPUS} \
    --num_tensor_parallel_workers ${TENSOR_PARALLEL_SIZES} \
    --max_tokens ${MAX_TOKENS} \
    --output_dir "./profiling_outputs_tmp/attention"

# Copy results
cp ./profiling_outputs_tmp/attention/*/${MODEL}/attention.csv \
   ${OUTPUT_BASE}/compute/${DEVICE_NAME}/${MODEL}/attention.csv
echo "Attention profiling complete: ${OUTPUT_BASE}/compute/${DEVICE_NAME}/${MODEL}/attention.csv"

# Phase 3: Collectives Profiling (all_reduce, send_recv)
echo ""
echo "=== Phase 3: Collective Communication Profiling ==="
python vidur/profiling/collectives/main.py \
    --num_workers_list "2 4" \
    --devices_per_node ${NUM_GPUS} \
    --output_dir "./profiling_outputs_tmp/collectives"

# Copy results
cp ./profiling_outputs_tmp/collectives/*/all_reduce.csv \
   ${OUTPUT_BASE}/network/${NETWORK_DEVICE}/all_reduce.csv
cp ./profiling_outputs_tmp/collectives/*/send_recv.csv \
   ${OUTPUT_BASE}/network/${NETWORK_DEVICE}/send_recv.csv
echo "Collectives profiling complete"

# Phase 4: CPU Overhead Profiling (Optional but recommended)
echo ""
echo "=== Phase 4: CPU Overhead Profiling ==="
python vidur/profiling/cpu_overhead/main.py \
    --models "${MODEL}" \
    --num_gpus ${NUM_GPUS} \
    --tensor_parallel_degrees "4" \
    --output_dir "./profiling_outputs_tmp/cpu_overhead"

# Create directory and copy
mkdir -p ${OUTPUT_BASE}/cpu_overhead/${NETWORK_DEVICE}/${MODEL}
cp ./profiling_outputs_tmp/cpu_overhead/*/${MODEL}/cpu_overheads.csv \
   ${OUTPUT_BASE}/cpu_overhead/${NETWORK_DEVICE}/${MODEL}/cpu_overheads.csv 2>/dev/null || echo "CPU overhead profiling skipped or failed"

# Cleanup
rm -rf ./profiling_outputs_tmp

echo ""
echo "=============================================="
echo "Profiling Complete!"
echo "=============================================="
echo ""
echo "Output locations:"
echo "  Compute: ${OUTPUT_BASE}/compute/${DEVICE_NAME}/${MODEL}/"
echo "  Network: ${OUTPUT_BASE}/network/${NETWORK_DEVICE}/"
echo ""
echo "Next steps:"
echo "  1. Verify profiling data files exist"
echo "  2. Run: sh block/exp/end_to_end_exp_scripts/a100_supplementary/a100_llama70b_exp.sh"
