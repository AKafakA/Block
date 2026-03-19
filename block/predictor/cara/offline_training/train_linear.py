#!/usr/bin/env python3
"""
Train linear latency predictor for CARA (ablation baseline).

Usage:
    python -m block.predictor.cara.offline_training.train_linear \
        --data-dir data/cara/latency_training/ \
        --output-dir models/cara/linear/
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

from block.predictor.cara.estimators.linear_predictor import LinearLatencyPredictor
from block.predictor.cara.offline_training.train_xgboost import load_latency_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def prepare_linear_features(records: list, target: str = "e2el"):
    """Extract simple 3-feature matrix from latency records."""
    X_list = []
    y_list = []
    for rec in records:
        target_val = rec.get(target) or rec.get("actual_e2e_latency")
        if target_val is None or target_val <= 0:
            continue

        schedule_state = rec.get("schedule_state", {})
        num_prompt = rec.get("num_prompt_tokens") or rec.get("input_len", 0)
        num_output = rec.get("num_predicted_output_tokens") or rec.get("max_tokens") or rec.get("output_len", 0)
        num_waiting = schedule_state.get("num_waiting", rec.get("num_waiting", 0))

        X_list.append([float(num_prompt), float(num_output), float(num_waiting)])
        y_list.append(float(target_val))

    return np.array(X_list), np.array(y_list, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Train linear latency predictor baseline for CARA"
    )
    parser.add_argument("--data-dir", required=True, help="Latency data directory")
    parser.add_argument("--instance-types", nargs="+", default=None)
    parser.add_argument("--target", default="e2el")
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--output-dir", default="models/cara/linear")
    args = parser.parse_args()

    records_by_type = load_latency_data(args.data_dir, args.instance_types)
    if not records_by_type:
        logger.error("No latency data found!")
        return

    predictor = LinearLatencyPredictor()
    all_metrics = {}

    for inst_type, records in sorted(records_by_type.items()):
        logger.info(f"Training linear model for {inst_type} ({len(records)} records)")

        X, y = prepare_linear_features(records, args.target)
        if len(X) < 10:
            logger.warning(f"Too few records for {inst_type}, skipping")
            continue

        n_val = int(len(X) * args.val_split)
        indices = np.random.RandomState(42).permutation(len(X))
        train_idx, val_idx = indices[: len(X) - n_val], indices[len(X) - n_val :]

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        model = LinearRegression()
        model.fit(X_train, y_train)

        preds = model.predict(X_val)
        errors = np.abs(y_val - preds)

        metrics = {
            "mae": float(mean_absolute_error(y_val, preds)),
            "rmse": float(np.sqrt(mean_squared_error(y_val, preds))),
            "mape": float(np.mean(errors / np.maximum(y_val, 1e-6)) * 100),
            "coef": model.coef_.tolist(),
            "intercept": float(model.intercept_),
            "n_train": len(X_train),
            "n_val": len(X_val),
        }

        predictor.models[inst_type] = model
        all_metrics[inst_type] = metrics

        logger.info(
            f"  {inst_type}: MAE={metrics['mae']:.4f}s, MAPE={metrics['mape']:.1f}%"
        )
        logger.info(
            f"  Coefficients: prompt={model.coef_[0]:.6f}, "
            f"output={model.coef_[1]:.6f}, waiting={model.coef_[2]:.6f}, "
            f"intercept={model.intercept_:.4f}"
        )

    predictor.save(args.output_dir)

    output_path = Path(args.output_dir)
    with open(output_path / "training_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    print("\n" + "=" * 70)
    print("LINEAR LATENCY PREDICTOR RESULTS")
    print("=" * 70)
    for inst_type, m in sorted(all_metrics.items()):
        print(f"  {inst_type}: MAE={m['mae']:.4f}s, MAPE={m['mape']:.1f}%")


if __name__ == "__main__":
    main()
