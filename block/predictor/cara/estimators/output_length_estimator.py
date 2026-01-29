"""
Output length estimator implementations.

Currently provides a placeholder implementation that returns configurable defaults.
KNN-based implementation is planned for future work.
"""
import logging
from typing import Dict, Tuple, Optional

from block.predictor.cara.estimators.base_estimator import BaseOutputLengthEstimator

logger = logging.getLogger(__name__)


class PlaceholderOutputLengthEstimator(BaseOutputLengthEstimator):
    """Placeholder output length estimator that returns configurable defaults.

    This is a temporary implementation until KNN-based estimation is available.
    It returns fixed default values, optionally with per-model overrides.

    Usage:
        estimator = PlaceholderOutputLengthEstimator(
            default_mean=256,
            default_p95=512,
            model_defaults={
                "Qwen/Qwen2.5-72B": (300, 600),  # Larger models tend to generate more
                "Qwen/Qwen2.5-3B": (200, 400),
            }
        )
        mean, p95 = estimator.predict(prompt, model_id)

    TODO: Implement KNN-based Output Length Estimator
    ================================================

    Implementation steps (see cara_paper/codex/cara-data-and-experiments.md):

    1. **Data Preparation**:
       - Use cara-best-route dataset with actual output lengths
       - Filter to valid completions (non-empty, within token limits)
       - Store: (prompt_embedding, model_id, output_length)

    2. **Embedding**:
       - Use sentence transformer: 'all-mpnet-base-v2' or 'all-MiniLM-L6-v2'
       - Embed prompts offline during index building
       - Embed new prompts online during prediction

    3. **Index Building**:
       - Build FAISS index for fast nearest neighbor search
       - Separate index per model OR single index with model filtering
       - Store output lengths alongside embeddings

    4. **Prediction**:
       ```python
       def predict(self, prompt: str, model_id: str) -> Tuple[float, float]:
           embedding = self.encoder.encode(prompt)
           distances, indices = self.index.search(embedding, k=10)

           # Filter to same model
           neighbors = [(d, self.output_lengths[i])
                       for d, i in zip(distances, indices)
                       if self.model_ids[i] == model_id]

           # Distance-weighted average
           weights = 1.0 / (distances + 1e-6)
           mean = np.average([n[1] for n in neighbors], weights=weights)

           # P95 from neighbor distribution
           p95 = np.percentile([n[1] for n in neighbors], 95)

           return (mean, p95)
       ```

    5. **Guardrails**:
       - Clamp predictions to reasonable range [1, max_tokens]
       - Fallback to defaults if no neighbors found
       - Optional: rolling bias correction based on actual vs predicted
    """

    def __init__(
        self,
        default_mean: int = 256,
        default_p95: int = 512,
        model_defaults: Optional[Dict[str, Tuple[int, int]]] = None
    ):
        """Initialize placeholder estimator.

        Args:
            default_mean: Default mean output length
            default_p95: Default 95th percentile output length
            model_defaults: Optional per-model (mean, p95) overrides
        """
        self.default_mean = default_mean
        self.default_p95 = default_p95
        self.model_defaults = model_defaults or {}

        logger.info(
            f"PlaceholderOutputLengthEstimator initialized: "
            f"default=({default_mean}, {default_p95}), "
            f"model_overrides={list(self.model_defaults.keys())}"
        )

    def predict(self, prompt: str, model_id: str) -> Tuple[float, float]:
        """Return default output length estimates.

        Args:
            prompt: Input prompt (ignored in placeholder)
            model_id: Model identifier

        Returns:
            Tuple of (mean_output_tokens, p95_output_tokens)
        """
        # Check for model-specific defaults
        if model_id in self.model_defaults:
            return self.model_defaults[model_id]

        # Check for partial model name match (e.g., "72B" in model_id)
        model_lower = model_id.lower()
        for key, values in self.model_defaults.items():
            if key.lower() in model_lower or model_lower in key.lower():
                return values

        return (float(self.default_mean), float(self.default_p95))
