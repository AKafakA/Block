"""
Estimators for CARA multi-objective scheduling.

This package provides estimators for:
- Output length: Predict expected output tokens for (prompt, model)
- Quality: Predict expected quality score for (prompt, model)
- Cost: Calculate cost based on token counts and pricing

Current Implementation Status:
- Output Length: Placeholder (returns defaults)
- Quality: Placeholder (returns model-size-based estimates)
- Cost: Fully implemented (token-based pricing)

Future Work (KNN-based estimators):
See cara_paper/codex/cara-data-and-experiments.md for implementation details:
1. Embed prompts using sentence transformers (MPNet/MiniLM)
2. Build KNN index from training data
3. For new prompt: find k nearest neighbors, return weighted average
"""

from block.predictor.cara.estimators.base_estimator import (
    BaseOutputLengthEstimator,
    BaseQualityEstimator,
)
from block.predictor.cara.estimators.output_length_estimator import (
    PlaceholderOutputLengthEstimator,
)
from block.predictor.cara.estimators.quality_estimator import (
    PlaceholderQualityEstimator,
)
from block.predictor.cara.estimators.cost_estimator import CostEstimator

__all__ = [
    'BaseOutputLengthEstimator',
    'BaseQualityEstimator',
    'PlaceholderOutputLengthEstimator',
    'PlaceholderQualityEstimator',
    'CostEstimator',
]
