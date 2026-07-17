#!/bin/bash
# 03_train_length_model.sh — Sec 6.2 prereq: train RoBERTa length predictor.
#
# Trains a RoBERTa regression head to predict response token length from
# the prompt text. The resulting checkpoint feeds scheduler runtime via the
# length predictor service and controls --use_estimated_response_lens
# benchmarks (Phase 1.1 Fanout-est, Phase 1.2 Po2-est, Phase 4-7 Po2-est, etc).
#
# One-time per (model, tokenizer) pair. Llama-2-7B is trained by default;
# Qwen2-7B supported via --model qwen.
#
# Usage:
#   sh 0c_train_length_model.sh              # Llama-2-7B (default)
#   sh 0c_train_length_model.sh qwen         # Qwen2-7B
#
# Time: ~3h on a single GPU (3090 / A30 / RTX 8000 all fine).
# Must be run on a GPU node (NOT on CloudLab where GPU lease is precious —
# ideally on a dedicated dev-GPU VM so the cluster lease is freed for serving
# experiments).

set -u
REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$REPO_ROOT"

MODEL="${1:-llama}"
case "$MODEL" in
    llama)
        TOKENIZER="meta-llama/Llama-2-7b-hf"
        TRAIN="data/length_estimation/sharegpt-llama-7b-train-40k.json"
        VAL="data/length_estimation/sharegpt-llama-7b-val-10k.json"
        OUTDIR="./model/roberta-length-prediction/llama-2-7b"
        ;;
    qwen)
        TOKENIZER="Qwen/Qwen2-7B"
        TRAIN="data/length_estimation/sharegpt-qwen-train-40k.json"
        VAL="data/length_estimation/sharegpt-qwen-val-10k.json"
        OUTDIR="./model/roberta-length-prediction/qwen2-7b"
        ;;
    *)
        echo "Usage: $0 [llama|qwen]"; exit 1;;
esac

echo "=== 03_train_length_model: Sec 6.2 prereq (RoBERTa length predictor for $MODEL) ==="
date -u +%Y-%m-%dT%H:%M:%SZ

# Verify training data
[ -f "$TRAIN" ] || { echo "FAIL: train data missing at $TRAIN"; exit 1; }
[ -f "$VAL" ]   || { echo "FAIL: val data missing at $VAL"; exit 1; }
echo "[ok] train: $TRAIN, val: $VAL"

# Verify GPU
nvidia-smi -L | head -1 || { echo "FAIL: no GPU visible"; exit 1; }

mkdir -p "$OUTDIR"

python block/length_estimation/train_roberta.py \
    --regression-model-name roberta-base \
    --tokenizer "$TOKENIZER" \
    --train-data-path "$TRAIN" \
    --val-data-path "$VAL" \
    --output-dir "$OUTDIR" \
    --epochs 300 \
    --batch-size 8 \
    2>&1 | tee "/tmp/ae_0c_train_length_$MODEL.log"

# Evaluate on val set
echo "--- Evaluation on held-out validation set ---"
python block/length_estimation/eval_roberta.py \
    --model-path "$OUTDIR" \
    --tokenizer "$TOKENIZER" \
    --val-data-path "$VAL" \
    2>&1 | tee "/tmp/ae_0c_eval_length_$MODEL.log"

echo "=== 03_train_length_model COMPLETE for $MODEL ==="
echo "Output checkpoint: $OUTDIR"
echo "Copy to cluster nodes under ~/Block/model/ before running Block experiments with --use_estimated_response_lens"
