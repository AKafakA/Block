"""
Learned predictor for CARA scheduler.

Wraps multiple trained models:
- Bucket classifier (ModernBERT): output length distribution per model
- XGBoost TTFT: time-to-first-token per instance type
- TPOT lookup table: per-instance-type average TPOT
- KNN quality: quality score per model

Used by the multi-objective scheduler to make routing decisions.
"""
import logging
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from block.predictor.cara.base_predictor import CARABasePredictor
from block.predictor.cara.data_structures import PredictRequest

logger = logging.getLogger(__name__)


class CARALearnedPredictor(CARABasePredictor):
    """Learned predictor combining bucket classifier, XGBoost TTFT,
    TPOT lookup, and KNN quality models.

    Provides predictions needed by the multi-objective scheduler:
    - Output length distribution (bucket probabilities)
    - Expected output length
    - P(tokens <= budget) for budget compliance
    - TTFT prediction
    - TPOT estimate
    - Quality score
    """

    def __init__(self, config, port: int, hostname: str = "localhost",
                 instance_type: str = "unknown"):
        super().__init__(config, port)
        self._hostname = hostname
        self._instance_type = instance_type

        # Sub-models (loaded lazily)
        self._bucket_model = None
        self._bucket_tokenizer = None
        self._xgboost_predictor = None
        self._knn_estimator = None

        # Config sections
        self._bucket_config = config.bucket_config
        self._xgboost_config = config.xgboost_config
        self._tpot_lookup = config.tpot_lookup
        self._quality_config = config.quality_config
        self._instance_metadata = config.instance_metadata
        self._scoring_weights = config.scoring_weights
        self._slo_defaults = config.slo_defaults

        # Bucket parameters
        self._bucket_size = self._bucket_config.get("bucket_size", 64)
        self._max_buckets = self._bucket_config.get("max_buckets", 16)
        self._max_length = self._bucket_config.get("max_length", 1024)

        self._load_models()

    def _load_models(self):
        """Load all sub-models from checkpoints."""
        t0 = time.time()

        # 1. Load bucket classifier
        self._load_bucket_classifier()

        # 2. Load XGBoost TTFT
        self._load_xgboost()

        # 3. Load KNN quality
        self._load_knn_quality()

        elapsed = time.time() - t0
        logger.info(f"LearnedPredictor loaded all models in {elapsed:.1f}s")

    def _load_bucket_classifier(self):
        """Load ModernBERT bucket classifier for the instance's model."""
        bucket_dir = self._bucket_config.get("model_dir", "")
        model_map = self._bucket_config.get("model_map", {})

        # Find the right model subdirectory for this instance's target LLM
        meta = self._instance_metadata.get(self._instance_type, {})
        model_name = meta.get("model_name", "")

        subdir = model_map.get(model_name)
        if not subdir or not bucket_dir:
            logger.warning(
                f"No bucket classifier configured for instance_type={self._instance_type}, "
                f"model={model_name}"
            )
            return

        model_path = Path(bucket_dir) / subdir
        if not model_path.exists():
            logger.warning(f"Bucket classifier not found at {model_path}")
            return

        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch

            self._bucket_tokenizer = AutoTokenizer.from_pretrained(
                str(model_path), trust_remote_code=True
            )
            self._bucket_model = AutoModelForSequenceClassification.from_pretrained(
                str(model_path), trust_remote_code=True
            )
            self._bucket_model.eval()

            # Move to GPU if available
            if torch.cuda.is_available():
                self._bucket_model = self._bucket_model.cuda()

            logger.info(f"Bucket classifier loaded from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load bucket classifier: {e}")

    def _load_xgboost(self):
        """Load XGBoost TTFT predictor."""
        xgb_dir = self._xgboost_config.get("model_dir", "")
        if not xgb_dir or not Path(xgb_dir).exists():
            logger.warning(f"XGBoost model dir not found: {xgb_dir}")
            return

        try:
            from block.predictor.cara.estimators.xgboost_predictor import (
                XGBoostLatencyPredictor,
            )
            self._xgboost_predictor = XGBoostLatencyPredictor.load(xgb_dir)
            logger.info(f"XGBoost predictor loaded from {xgb_dir}")
        except Exception as e:
            logger.error(f"Failed to load XGBoost predictor: {e}")

    def _load_knn_quality(self):
        """Load KNN quality estimator."""
        knn_dir = self._quality_config.get("model_dir", "")
        if not knn_dir or not Path(knn_dir).exists():
            logger.warning(f"KNN model dir not found: {knn_dir}")
            return

        try:
            from block.predictor.cara.estimators.knn_estimator import KNNEstimator
            device = self._quality_config.get("device", "cpu")
            self._knn_estimator = KNNEstimator.load(knn_dir, device=device)
            logger.info(f"KNN quality estimator loaded from {knn_dir}")
        except Exception as e:
            logger.error(f"Failed to load KNN quality estimator: {e}")

    def predict_bucket_distribution(self, prompt: str) -> np.ndarray:
        """Predict output length bucket probabilities.

        Returns:
            Array of shape (num_buckets,) with probabilities.
        """
        if self._bucket_model is None or self._bucket_tokenizer is None:
            # Uniform fallback
            return np.ones(self._max_buckets) / self._max_buckets

        import torch

        inputs = self._bucket_tokenizer(
            prompt,
            truncation=True,
            max_length=self._bucket_config.get("max_length", 1024),
            return_tensors="pt",
        )

        device = next(self._bucket_model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self._bucket_model(**inputs).logits.squeeze()
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

        return probs

    def predict_expected_length(self, bucket_probs: np.ndarray) -> float:
        """Compute expected output length from bucket probabilities."""
        midpoints = np.array([
            (i * self._bucket_size + self._bucket_size / 2)
            for i in range(len(bucket_probs))
        ])
        return float(np.sum(bucket_probs * midpoints))

    def predict_budget_compliance(
        self, bucket_probs: np.ndarray, budget_tokens: int
    ) -> float:
        """Compute P(output_tokens <= budget) from bucket distribution."""
        budget_bucket = min(budget_tokens // self._bucket_size, len(bucket_probs) - 1)
        return float(np.sum(bucket_probs[: budget_bucket + 1]))

    def predict_ttft(
        self, schedule_state: Dict, num_prompt_tokens: int,
        num_predicted_output_tokens: int
    ) -> float:
        """Predict TTFT using XGBoost model.

        Returns TTFT in seconds.
        """
        if self._xgboost_predictor is None:
            return 1.0  # fallback: 1 second

        try:
            result = self._xgboost_predictor.predict(
                instance_type=self._instance_type,
                schedule_state=schedule_state,
                num_prompt_tokens=num_prompt_tokens,
                num_predicted_output_tokens=num_predicted_output_tokens,
            )
            return result.get("e2e_latency", 1.0)  # XGBoost trained on TTFT target
        except (ValueError, KeyError):
            return 1.0

    def predict_tpot(self) -> float:
        """Get TPOT estimate from lookup table.

        Returns TPOT in seconds.
        """
        return self._tpot_lookup.get(self._instance_type, 0.05)

    def predict_quality(self, prompt: str, model_name: str) -> float:
        """Predict quality score using KNN estimator.

        Returns quality score in [0, 1].
        """
        if self._knn_estimator is None:
            return 0.5  # fallback: neutral quality

        try:
            return self._knn_estimator.predict_quality(prompt, model_name)
        except (ValueError, KeyError):
            return 0.5

    async def predict(self, target_request: PredictRequest) -> Dict:
        """Full prediction for scheduling decision.

        Returns dict with all metrics needed by multi-objective scheduler.
        """
        # For the predictor API, we don't have the prompt text.
        # The scheduler will call predict_full() with the prompt directly.
        # This method provides a simple metric for backward compatibility.
        return {
            "target_metric": 0.0,
            "gpu_blocks": 0,
            "num_requests": 0,
            "num_preempted": 0,
            "predictor_type": "learned",
            "tpot": self.predict_tpot(),
        }

    def predict_full(
        self, prompt: str, num_prompt_tokens: int,
        num_predicted_output_tokens: int,
        schedule_state: Optional[Dict] = None,
        budget_tokens: int = 256,
    ) -> Dict:
        """Full prediction with all metrics for multi-objective scheduling.

        Args:
            prompt: Request prompt text
            num_prompt_tokens: Number of prompt tokens
            num_predicted_output_tokens: Predicted output tokens
            schedule_state: Current instance state (for TTFT prediction)
            budget_tokens: Token budget for compliance check

        Returns:
            Dict with:
                - bucket_probs: array of bucket probabilities
                - expected_length: expected output tokens
                - p_under_budget: P(output <= budget)
                - ttft: predicted TTFT in seconds
                - tpot: predicted TPOT in seconds
                - quality: predicted quality score [0, 1]
                - model_name: target LLM model name
                - instance_type: instance type string
                - cost_per_token: cost per output token
        """
        meta = self._instance_metadata.get(self._instance_type, {})
        model_name = meta.get("model_name", "unknown")
        cost_per_token = meta.get("cost_per_token", 0.01)

        # 1. Bucket distribution
        bucket_probs = self.predict_bucket_distribution(prompt)
        expected_length = self.predict_expected_length(bucket_probs)
        p_under_budget = self.predict_budget_compliance(bucket_probs, budget_tokens)

        # 2. TTFT
        state = schedule_state or {}
        ttft = self.predict_ttft(state, num_prompt_tokens, num_predicted_output_tokens)

        # 3. TPOT
        tpot = self.predict_tpot()

        # 4. Quality
        quality = self.predict_quality(prompt, model_name)

        return {
            "bucket_probs": bucket_probs.tolist(),
            "expected_length": expected_length,
            "p_under_budget": p_under_budget,
            "ttft": ttft,
            "tpot": tpot,
            "quality": quality,
            "model_name": model_name,
            "instance_type": self._instance_type,
            "cost_per_token": cost_per_token,
        }
