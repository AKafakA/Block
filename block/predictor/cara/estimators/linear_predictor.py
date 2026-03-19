#!/usr/bin/env python3
"""
Linear latency predictor for CARA (ablation baseline).

Simple 4-coefficient OLS regression:
    latency = a*prompt_tokens + b*predicted_output_tokens + c*num_waiting + d

One model per (model_size, GPU_type) combination.
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

LINEAR_FEATURES = [
    "num_prompt_tokens",
    "num_predicted_output_tokens",
    "num_waiting",
]


class LinearLatencyPredictor:
    """Linear regression latency predictor (baseline).

    Uses sklearn OLS with 3 features + intercept per instance type.
    """

    def __init__(self):
        self.models: Dict[str, object] = {}  # instance_type -> LinearRegression

    def predict(
        self,
        instance_type: str,
        schedule_state: Dict,
        num_prompt_tokens: int,
        num_predicted_output_tokens: int,
    ) -> Dict[str, float]:
        """Predict latency for a request."""
        if instance_type not in self.models:
            raise ValueError(f"No model for {instance_type}")

        features = np.array([[
            float(num_prompt_tokens),
            float(num_predicted_output_tokens),
            float(schedule_state.get("num_waiting", 0)),
        ]])

        pred = self.models[instance_type].predict(features)
        return {"e2e_latency": max(float(pred[0]), 0.0)}

    def save(self, output_dir: str):
        """Save all models to disk."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for inst_type, model in self.models.items():
            with open(output_path / f"{inst_type}_linear.pkl", "wb") as f:
                pickle.dump(model, f)

        metadata = {
            "instance_types": list(self.models.keys()),
            "feature_names": LINEAR_FEATURES,
            "coefficients": {},
        }
        for inst_type, model in self.models.items():
            metadata["coefficients"][inst_type] = {
                "coef": model.coef_.tolist(),
                "intercept": float(model.intercept_),
            }

        with open(output_path / "linear_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Linear predictor saved to {output_path}")

    @classmethod
    def load(cls, model_dir: str) -> "LinearLatencyPredictor":
        """Load predictor from disk."""
        model_path = Path(model_dir)
        with open(model_path / "linear_metadata.json") as f:
            metadata = json.load(f)

        predictor = cls()
        for inst_type in metadata["instance_types"]:
            with open(model_path / f"{inst_type}_linear.pkl", "rb") as f:
                predictor.models[inst_type] = pickle.load(f)

        logger.info(f"Linear predictor loaded: {len(predictor.models)} models")
        return predictor
