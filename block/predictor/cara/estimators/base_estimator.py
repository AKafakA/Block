"""
Abstract base classes for CARA estimators.

These define the interface for output length and quality estimation,
which are used in multi-objective scoring before latency prediction.
"""
from abc import ABC, abstractmethod
from typing import Tuple


class BaseOutputLengthEstimator(ABC):
    """Abstract base class for output length estimation.

    Estimates expected output length for a request on a given model.
    This is called BEFORE the latency predictor to fill the
    `num_predicted_output_tokens` field in PredictRequest.

    The output length is critical for:
    1. Latency prediction (output tokens affect decode time)
    2. Cost estimation (cost = f(input_tokens, output_tokens))
    3. Memory estimation (KV cache size depends on sequence length)

    Implementation Options:
    -----------------------

    1. **KNN-based** (Recommended for production):
       - Embed prompt using sentence transformer (MPNet/MiniLM)
       - Find k nearest neighbors in training data
       - Return weighted average of their output lengths
       - Training data: cara-best-route dataset with actual output lengths
       - Pros: Fast inference, easy to update, no training required
       - Cons: Requires embedding index, quality depends on training data coverage

    2. **Regression-based**:
       - Frozen embeddings + small MLP/GBDT regressor
       - Can use quantile regression for conservative bounds (Q90/Q95)
       - Pros: Better generalization
       - Cons: Requires training, harder to update

    3. **Heuristic-based** (Current placeholder):
       - Return fixed defaults based on task type or model
       - Pros: Simple, no dependencies
       - Cons: Inaccurate, doesn't adapt to prompt content

    See: cara_paper/codex/cara-data-and-experiments.md for detailed implementation guide.
    """

    @abstractmethod
    def predict(
        self,
        prompt: str,
        model_id: str
    ) -> Tuple[float, float]:
        """Predict output length for a prompt on a specific model.

        Args:
            prompt: Input prompt text (used for embedding in KNN approach)
            model_id: Model identifier (e.g., "Qwen/Qwen2.5-3B")

        Returns:
            Tuple of (mean_output_tokens, p95_output_tokens):
            - mean_output_tokens: Expected output length (for scoring)
            - p95_output_tokens: 95th percentile estimate (for conservative bounds/constraints)
        """
        pass


class BaseQualityEstimator(ABC):
    """Abstract base class for quality estimation.

    Estimates expected quality score for a request on a given model.
    Quality is defined relative to the best available model's response,
    typically measured via LLM-judge similarity scoring.

    Quality scores are used for:
    1. RSO constraint checking (user specifies minimum quality)
    2. Multi-objective scoring (trade off quality vs latency/cost)
    3. Model routing decisions (route to capable model)

    Implementation Options:
    -----------------------

    1. **KNN-based** (Recommended for production):
       - Embed prompt using sentence transformer
       - Find k nearest neighbors in training data
       - Return distance-weighted average of per-model quality scores
       - Training data: cara-best-route dataset with LLM-judge quality scores
       - Quality score = similarity to 72B model response

    2. **Lightweight encoder head**:
       - Frozen embeddings + small MLP per model (or multi-head)
       - Pros: Better generalization, handles distribution shift
       - Cons: Requires training per model

    3. **Model-size heuristic** (Current placeholder):
       - Larger models assumed to have higher quality
       - Quality proportional to model parameter count
       - Pros: Simple, no training data needed
       - Cons: Doesn't account for prompt-specific difficulty

    See: cara_paper/codex/cara-data-and-experiments.md for detailed implementation guide.
    """

    @abstractmethod
    def predict(
        self,
        prompt: str,
        model_id: str
    ) -> float:
        """Predict quality score for a prompt on a specific model.

        Args:
            prompt: Input prompt text (used for embedding in KNN approach)
            model_id: Model identifier (e.g., "Qwen/Qwen2.5-3B")

        Returns:
            Quality score in [0, 1] where:
            - 1.0 = equivalent to best model (e.g., 72B)
            - 0.0 = lowest quality
        """
        pass
