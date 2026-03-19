#!/usr/bin/env python3
"""
XGBoost-based latency predictor for CARA.

Predicts E2E latency, TTFT, and TPOT given instance state + request features.
One model per (model_size, GPU_type) combination.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Feature order (must match training)
INSTANCE_STATE_FEATURES = [
    "ema_decode_tok_per_s",
    "ema_prefill_tok_per_s",
    "ema_decode_iter_ms",
    "decode_ctx_p50",
    "decode_ctx_p95",
    "decode_ctx_max",
    "num_running",
    "num_active_decode_seqs",
    "num_waiting",
    "pending_prefill_tokens",
    "pending_decode_tokens",
    "token_budget_per_iter",
    "prefill_chunk_size",
    "max_num_seqs",
    "kv_cache_utilization",
    "kv_free_blocks",
    "kv_evictions_per_s",
]

REQUEST_FEATURES = [
    "num_prompt_tokens",
    "num_predicted_output_tokens",
]

DERIVED_FEATURES = [
    "req_decode_ctx_avg",
    "req_decode_ctx_max",
    "backlog_seconds",
    "request_prefill_time_est",
    "request_decode_time_est",
    "ctx_relative_to_p95",
]

ALL_FEATURES = INSTANCE_STATE_FEATURES + REQUEST_FEATURES + DERIVED_FEATURES


def compute_derived_features(
    schedule_state: Dict, num_prompt_tokens: int, num_predicted_output_tokens: int
) -> Dict[str, float]:
    """Compute derived features from instance state and request."""
    ema_prefill = schedule_state.get("ema_prefill_tok_per_s", 1.0) or 1.0
    ema_decode = schedule_state.get("ema_decode_tok_per_s", 1.0) or 1.0
    decode_ctx_p95 = schedule_state.get("decode_ctx_p95", 1.0) or 1.0

    req_decode_ctx_avg = num_prompt_tokens + (num_predicted_output_tokens - 1) / 2
    req_decode_ctx_max = num_prompt_tokens + num_predicted_output_tokens - 1

    pending_prefill = schedule_state.get("pending_prefill_tokens", 0)
    pending_decode = schedule_state.get("pending_decode_tokens", 0)

    backlog_seconds = pending_prefill / ema_prefill + pending_decode / ema_decode
    request_prefill_time_est = num_prompt_tokens / ema_prefill
    request_decode_time_est = num_predicted_output_tokens / ema_decode
    ctx_relative_to_p95 = req_decode_ctx_max / decode_ctx_p95 if decode_ctx_p95 > 0 else 1.0

    return {
        "req_decode_ctx_avg": req_decode_ctx_avg,
        "req_decode_ctx_max": req_decode_ctx_max,
        "backlog_seconds": backlog_seconds,
        "request_prefill_time_est": request_prefill_time_est,
        "request_decode_time_est": request_decode_time_est,
        "ctx_relative_to_p95": ctx_relative_to_p95,
    }


def build_feature_vector(
    schedule_state: Dict, num_prompt_tokens: int, num_predicted_output_tokens: int
) -> np.ndarray:
    """Build a single feature vector from instance state + request features."""
    derived = compute_derived_features(
        schedule_state, num_prompt_tokens, num_predicted_output_tokens
    )

    features = []
    for f in INSTANCE_STATE_FEATURES:
        features.append(float(schedule_state.get(f, 0.0) or 0.0))
    features.append(float(num_prompt_tokens))
    features.append(float(num_predicted_output_tokens))
    for f in DERIVED_FEATURES:
        features.append(float(derived.get(f, 0.0)))

    return np.array(features, dtype=np.float32)


class XGBoostLatencyPredictor:
    """XGBoost-based latency predictor.

    Maintains one model per (model_size, GPU_type) combination.
    Predicts E2E latency given instance state and request features.
    """

    def __init__(self):
        self.models: Dict[str, object] = {}  # instance_type -> xgb model
        self.feature_names: List[str] = ALL_FEATURES

    def predict(
        self,
        instance_type: str,
        schedule_state: Dict,
        num_prompt_tokens: int,
        num_predicted_output_tokens: int,
    ) -> Dict[str, float]:
        """Predict latency for a request on a specific instance type.

        Args:
            instance_type: E.g. "qwen2.5-3b_p100"
            schedule_state: Instance state dict from /instance_stats API.
            num_prompt_tokens: Input prompt tokens.
            num_predicted_output_tokens: Predicted output tokens.

        Returns:
            Dict with predicted e2e_latency (seconds).
        """
        if instance_type not in self.models:
            raise ValueError(
                f"No model for instance_type={instance_type}. "
                f"Available: {list(self.models.keys())}"
            )

        features = build_feature_vector(
            schedule_state, num_prompt_tokens, num_predicted_output_tokens
        )
        import xgboost as xgb

        dmatrix = xgb.DMatrix(features.reshape(1, -1), feature_names=self.feature_names)
        model = self.models[instance_type]
        pred = model.predict(dmatrix)

        return {"e2e_latency": float(pred[0])}

    def predict_batch(
        self,
        instance_type: str,
        schedule_state: Dict,
        requests: List[Dict],
    ) -> List[Dict[str, float]]:
        """Predict latency for multiple requests sharing the same instance state.

        Args:
            requests: List of dicts with num_prompt_tokens, num_predicted_output_tokens.
        """
        if instance_type not in self.models:
            raise ValueError(f"No model for instance_type={instance_type}")

        import xgboost as xgb

        feature_rows = []
        for req in requests:
            fv = build_feature_vector(
                schedule_state,
                req["num_prompt_tokens"],
                req["num_predicted_output_tokens"],
            )
            feature_rows.append(fv)

        X = np.stack(feature_rows)
        dmatrix = xgb.DMatrix(X, feature_names=self.feature_names)
        preds = self.models[instance_type].predict(dmatrix)

        return [{"e2e_latency": float(p)} for p in preds]

    def save(self, output_dir: str):
        """Save all models to disk."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for instance_type, model in self.models.items():
            model_file = output_path / f"{instance_type}.xgb"
            model.save_model(str(model_file))

        metadata = {
            "instance_types": list(self.models.keys()),
            "feature_names": self.feature_names,
            "num_features": len(self.feature_names),
        }
        with open(output_path / "xgboost_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"XGBoost predictor saved to {output_path}: {len(self.models)} models")

    @classmethod
    def load(cls, model_dir: str) -> "XGBoostLatencyPredictor":
        """Load predictor from disk."""
        import xgboost as xgb

        model_path = Path(model_dir)
        with open(model_path / "xgboost_metadata.json") as f:
            metadata = json.load(f)

        predictor = cls()
        predictor.feature_names = metadata["feature_names"]

        for instance_type in metadata["instance_types"]:
            model_file = model_path / f"{instance_type}.xgb"
            model = xgb.Booster()
            model.load_model(str(model_file))
            predictor.models[instance_type] = model

        logger.info(f"XGBoost predictor loaded: {len(predictor.models)} models")
        return predictor
