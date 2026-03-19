#!/usr/bin/env python3
"""
LSTM-based predictor for CARA latency prediction.

Sequence model on sliding window of (schedule_state, request_features).
Evaluates whether temporal/sequential information improves over point-in-time features.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class LSTMLatencyModel(nn.Module):
    """LSTM model for latency prediction from time-series state snapshots."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            (batch,) predicted latency
        """
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden)
        last_hidden = lstm_out[:, -1, :]  # Use last timestep
        return self.head(last_hidden).squeeze(-1)


class LSTMLatencyPredictor:
    """LSTM-based latency predictor.

    Uses a sliding window of recent schedule states to predict latency.
    One model per (model_size, GPU_type).
    """

    def __init__(
        self,
        window_size: int = 10,
        hidden_dim: int = 128,
        num_layers: int = 2,
        device: str = "cpu",
    ):
        self.window_size = window_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.device = device

        self.models: Dict[str, LSTMLatencyModel] = {}
        self.input_dim: int = 0

    def predict(
        self,
        instance_type: str,
        state_window: List[Dict],
        num_prompt_tokens: int,
        num_predicted_output_tokens: int,
    ) -> Dict[str, float]:
        """Predict latency given a window of recent states.

        Args:
            instance_type: E.g. "qwen2.5-3b_p100"
            state_window: List of recent schedule_state dicts (most recent last).
            num_prompt_tokens: Input tokens.
            num_predicted_output_tokens: Predicted output tokens.
        """
        if instance_type not in self.models:
            raise ValueError(f"No model for {instance_type}")

        from block.predictor.cara.estimators.xgboost_predictor import build_feature_vector

        # Build feature vectors for each timestep
        features = []
        for state in state_window[-self.window_size :]:
            fv = build_feature_vector(state, num_prompt_tokens, num_predicted_output_tokens)
            features.append(fv)

        # Pad if shorter than window
        while len(features) < self.window_size:
            features.insert(0, features[0].copy())

        x = torch.tensor(np.stack(features), dtype=torch.float32).unsqueeze(0).to(self.device)

        model = self.models[instance_type]
        model.eval()
        with torch.no_grad():
            pred = model(x)

        return {"e2e_latency": max(float(pred.item()), 0.0)}

    def save(self, output_dir: str):
        """Save all models to disk."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        state = {
            "window_size": self.window_size,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "input_dim": self.input_dim,
            "instance_types": list(self.models.keys()),
            "model_states": {
                inst: model.state_dict() for inst, model in self.models.items()
            },
        }
        torch.save(state, output_path / "lstm_predictor.pt")

        metadata = {
            "window_size": self.window_size,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "input_dim": self.input_dim,
            "instance_types": list(self.models.keys()),
        }
        with open(output_path / "lstm_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"LSTM predictor saved to {output_path}")

    @classmethod
    def load(cls, model_dir: str, device: str = "cpu") -> "LSTMLatencyPredictor":
        """Load predictor from disk."""
        model_path = Path(model_dir) / "lstm_predictor.pt"
        state = torch.load(model_path, map_location=device, weights_only=False)

        predictor = cls(
            window_size=state["window_size"],
            hidden_dim=state["hidden_dim"],
            num_layers=state["num_layers"],
            device=device,
        )
        predictor.input_dim = state["input_dim"]

        for inst_type in state["instance_types"]:
            model = LSTMLatencyModel(
                input_dim=predictor.input_dim,
                hidden_dim=predictor.hidden_dim,
                num_layers=predictor.num_layers,
            ).to(device)
            model.load_state_dict(state["model_states"][inst_type])
            predictor.models[inst_type] = model

        logger.info(f"LSTM predictor loaded: {len(predictor.models)} models")
        return predictor
