#!/usr/bin/env python3
"""
Train XGBoost TTFT (Time-To-First-Token) predictor for CARA.

Convenience wrapper around train_xgboost.py with TTFT-specific defaults.
Trains one model per (model_size, GPU_type) combination.

Usage:
    python -m block.predictor.cara.offline_training.train_xgboost_ttft \
        --data-dir data/cara/latency_training/ \
        --output-dir models/cara/xgboost_ttft/

    # For specific instance types only
    python -m block.predictor.cara.offline_training.train_xgboost_ttft \
        --data-dir data/cara/latency_training/ \
        --instance-types qwen2.5-3b_p100 qwen2.5-7b_a30 \
        --output-dir models/cara/xgboost_ttft/
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from block.predictor.cara.offline_training.train_xgboost import (
    load_latency_data,
    prepare_features_and_targets,
    train_xgboost_model,
)
from block.predictor.cara.estimators.xgboost_predictor import (
    ALL_FEATURES,
    XGBoostLatencyPredictor,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Train XGBoost TTFT predictor for CARA"
    )
    parser.add_argument("--data-dir", required=True, help="Latency data directory (JSONL)")
    parser.add_argument(
        "--instance-types",
        nargs="+",
        default=None,
        help="Instance types to train for (default: all found in data)",
    )
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split")
    parser.add_argument("--output-dir", default="models/cara/xgboost_ttft")
    args = parser.parse_args()

    # Load data
    records_by_type = load_latency_data(args.data_dir, args.instance_types)

    if not records_by_type:
        logger.error("No latency data found!")
        return

    predictor = XGBoostLatencyPredictor()
    all_metrics = {}

    for inst_type, records in sorted(records_by_type.items()):
        logger.info(f"\n{'='*60}")
        logger.info(f"Training TTFT model for {inst_type} ({len(records)} records)")
        logger.info(f"{'='*60}")

        if len(records) < 50:
            logger.warning(f"Too few records for {inst_type} ({len(records)}), skipping")
            continue

        # Prepare features with TTFT as target
        X, y = prepare_features_and_targets(records, target="ttft")
        if len(X) < 50:
            logger.warning(f"Too few valid TTFT records for {inst_type} ({len(X)}), skipping")
            continue

        # Train/val split
        n_val = int(len(X) * args.val_split)
        n_train = len(X) - n_val

        indices = np.random.RandomState(42).permutation(len(X))
        train_idx = indices[:n_train]
        val_idx = indices[n_train:]

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        t0 = time.time()

        # TTFT-optimized XGBoost params (lower depth, more regularization)
        ttft_params = {
            "objective": "reg:squarederror",
            "eval_metric": ["mae", "rmse"],
            "max_depth": 5,
            "learning_rate": 0.08,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 10,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "seed": 42,
        }

        model, metrics = train_xgboost_model(
            X_train, y_train, X_val, y_val,
            feature_names=ALL_FEATURES,
            params=ttft_params,
        )
        train_time = time.time() - t0

        predictor.models[inst_type] = model
        metrics["train_time_s"] = train_time
        metrics["target"] = "ttft"
        all_metrics[inst_type] = metrics

        logger.info(
            f"  {inst_type}: MAE={metrics['mae']:.4f}s, "
            f"MAPE={metrics['mape']:.1f}%, "
            f"RMSE={metrics['rmse']:.4f}s, "
            f"trained in {train_time:.1f}s"
        )

    # Save
    predictor.save(args.output_dir)

    # Save metrics
    output_path = Path(args.output_dir)
    with open(output_path / "training_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    # Print summary
    print("\n" + "=" * 70)
    print("XGBOOST TTFT PREDICTOR TRAINING RESULTS")
    print("=" * 70)
    for inst_type, metrics in sorted(all_metrics.items()):
        print(
            f"  {inst_type}: MAE={metrics['mae']:.4f}s, "
            f"MAPE={metrics['mape']:.1f}%, "
            f"RMSE={metrics['rmse']:.4f}s, "
            f"n_train={metrics['n_train']}, n_val={metrics['n_val']}"
        )


if __name__ == "__main__":
    main()
