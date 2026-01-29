"""
Quality estimator implementations.

Currently provides a placeholder implementation based on model size.
KNN-based implementation is planned for future work.
"""
import logging
import re
from typing import Dict, Optional

from block.predictor.cara.estimators.base_estimator import BaseQualityEstimator

logger = logging.getLogger(__name__)


class PlaceholderQualityEstimator(BaseQualityEstimator):
    """Placeholder quality estimator based on model size heuristics.

    This is a temporary implementation until KNN-based estimation is available.
    It assumes larger models produce higher quality responses, which is a
    reasonable baseline but doesn't account for prompt-specific factors.

    Quality scores are normalized to [0, 1] where:
    - 1.0 = equivalent to best/largest model
    - Lower values for smaller models

    Usage:
        estimator = PlaceholderQualityEstimator(
            quality_by_size={
                "72b": 1.0,
                "32b": 0.95,
                "14b": 0.85,
                "7b": 0.75,
                "3b": 0.65,
            }
        )
        quality = estimator.predict(prompt, "Qwen/Qwen2.5-7B")  # Returns 0.75

    TODO: Implement KNN-based Quality Estimator
    ===========================================

    Implementation steps (see cara_paper/codex/cara-data-and-experiments.md):

    1. **Data Preparation**:
       - Use cara-best-route dataset with LLM-judge quality scores
       - Quality = similarity score to 72B model response
       - Store: (prompt_embedding, model_id, quality_score)

    2. **Embedding**:
       - Use same sentence transformer as output length estimator
       - Share embeddings between estimators for efficiency

    3. **Index Building**:
       - Build FAISS index (can share with output length estimator)
       - Store per-model quality scores for each prompt

    4. **Prediction**:
       ```python
       def predict(self, prompt: str, model_id: str) -> float:
           embedding = self.encoder.encode(prompt)
           distances, indices = self.index.search(embedding, k=10)

           # Get quality scores for this model from neighbors
           qualities = []
           weights = []
           for d, i in zip(distances, indices):
               if model_id in self.quality_scores[i]:
                   qualities.append(self.quality_scores[i][model_id])
                   weights.append(1.0 / (d + 1e-6))

           if not qualities:
               return self.default_quality.get(model_id, 0.7)

           # Distance-weighted average
           return np.average(qualities, weights=weights)
       ```

    5. **Evaluation**:
       - Best-model selection accuracy vs 72B
       - Monotonicity by model size
       - Per-source breakdown (different prompt types)

    6. **Alternative: Lightweight Encoder Head**:
       - Frozen embeddings + small MLP per model
       - Better generalization for out-of-distribution prompts
       - Requires training but more robust
    """

    # Default quality scores by model size
    DEFAULT_QUALITY_BY_SIZE: Dict[str, float] = {
        "72b": 1.0,
        "70b": 1.0,
        "65b": 0.98,
        "32b": 0.95,
        "34b": 0.95,
        "14b": 0.85,
        "13b": 0.85,
        "7b": 0.75,
        "8b": 0.75,
        "3b": 0.65,
        "1b": 0.55,
        "0.5b": 0.45,
    }

    def __init__(
        self,
        quality_by_size: Optional[Dict[str, float]] = None,
        default_quality: float = 0.7
    ):
        """Initialize placeholder quality estimator.

        Args:
            quality_by_size: Mapping from model size string to quality score
            default_quality: Default quality when model size cannot be determined
        """
        self.quality_by_size = quality_by_size or self.DEFAULT_QUALITY_BY_SIZE
        self.default_quality = default_quality

        logger.info(
            f"PlaceholderQualityEstimator initialized: "
            f"default={default_quality}, "
            f"sizes={list(self.quality_by_size.keys())}"
        )

    def predict(self, prompt: str, model_id: str) -> float:
        """Estimate quality based on model size.

        Args:
            prompt: Input prompt (ignored in placeholder)
            model_id: Model identifier (e.g., "Qwen/Qwen2.5-7B")

        Returns:
            Quality score in [0, 1]
        """
        model_lower = model_id.lower()

        # Extract size from model name (e.g., "7b" from "Qwen2.5-7B")
        # Pattern matches: 72b, 7b, 0.5b, 1.5b, etc.
        size_pattern = r'(\d+\.?\d*)b'
        match = re.search(size_pattern, model_lower)

        if match:
            size_str = match.group(1) + "b"
            # Try exact match first
            if size_str in self.quality_by_size:
                return self.quality_by_size[size_str]

            # Try numeric comparison for closest match
            try:
                size_num = float(match.group(1))
                # Find closest size
                closest_size = None
                closest_diff = float('inf')
                for key in self.quality_by_size:
                    try:
                        key_num = float(key.rstrip('b'))
                        diff = abs(key_num - size_num)
                        if diff < closest_diff:
                            closest_diff = diff
                            closest_size = key
                    except ValueError:
                        continue

                if closest_size:
                    return self.quality_by_size[closest_size]
            except ValueError:
                pass

        # Check for known model family patterns
        for size, quality in self.quality_by_size.items():
            if size in model_lower:
                return quality

        return self.default_quality
