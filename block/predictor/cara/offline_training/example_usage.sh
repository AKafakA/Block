#!/bin/bash
# Example usage of CARA training data preparation

# Set paths
INPUT_DATA="data/cara/cara-best-route-training.json"
TOKENIZER="Qwen/Qwen2.5-72B"

echo "========================================"
echo "CARA Training Data Preparation Examples"
echo "========================================"

# Example 1: Basic similarity scoring (fast, recommended)
echo -e "\n[Example 1] Basic similarity scoring (CPU)"
python -m block.predictor.cara.offline_training.prepare_training_data \
  --input "$INPUT_DATA" \
  --tokenizer "$TOKENIZER" \
  --scoring-method similarity \
  --device cpu

# Example 2: Similarity with GPU and better embedding model
echo -e "\n[Example 2] Similarity scoring with GPU and better embeddings"
python -m block.predictor.cara.offline_training.prepare_training_data \
  --input "$INPUT_DATA" \
  --tokenizer "$TOKENIZER" \
  --scoring-method similarity \
  --embedding-model sentence-transformers/all-mpnet-base-v2 \
  --device cuda \
  --debug

# Example 3: LLM judge scoring
echo -e "\n[Example 3] LLM judge scoring"
python -m block.predictor.cara.offline_training.prepare_training_data \
  --input "$INPUT_DATA" \
  --tokenizer "$TOKENIZER" \
  --scoring-method llm_judge \
  --judge-model Unbabel/M-Prometheus-7B \
  --device cuda

# Example 4: Custom filtering parameters
echo -e "\n[Example 4] Custom filtering parameters"
python -m block.predictor.cara.offline_training.prepare_training_data \
  --input "$INPUT_DATA" \
  --tokenizer "$TOKENIZER" \
  --scoring-method similarity \
  --min-output-tokens 5 \
  --max-output-tokens 512 \
  --min-compression-ratio 0.25 \
  --device cpu

# Example 5: Specify reference model explicitly
echo -e "\n[Example 5] Explicit reference model"
python -m block.predictor.cara.offline_training.prepare_training_data \
  --input "$INPUT_DATA" \
  --tokenizer "$TOKENIZER" \
  --scoring-method similarity \
  --reference-model "Qwen/Qwen2.5-72B" \
  --device cpu

# Example 6: Exclude unavailable models (NEW)
echo -e "\n[Example 6] Exclude unavailable models"
python -m block.predictor.cara.offline_training.prepare_training_data \
  --input "$INPUT_DATA" \
  --tokenizer "$TOKENIZER" \
  --exclude-models "Qwen/Qwen2.5-3B" \
  --scoring-method similarity \
  --device cpu

# Example 7: Exclude models with LLM judge (RECOMMENDED for adversarial prompts)
echo -e "\n[Example 7] Exclude models + LLM judge"
python -m block.predictor.cara.offline_training.prepare_training_data \
  --input "$INPUT_DATA" \
  --tokenizer "$TOKENIZER" \
  --exclude-models "Qwen/Qwen2.5-3B" \
  --scoring-method llm_judge \
  --device cuda

echo -e "\n========================================"
echo "Examples complete!"
echo "========================================"