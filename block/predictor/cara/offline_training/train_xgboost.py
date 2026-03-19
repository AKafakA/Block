#!/usr/bin/env python3
"""
Train XGBoost latency predictor for CARA.

Trains one model per (model_size, GPU_type) combination using latency
benchmark data collected via generate_latency_benchmark.py.

Usage:
    # Train on all data
    python -m block.predictor.cara.offline_training.train_xgboost \
        --data-dir data/cara/latency_training/ \
        --output-dir models/cara/xgboost/

    # Train for specific instance type
    python -m block.predictor.cara.offline_training.train_xgboost \
        --data-dir data/cara/latency_training/ \
        --instance-types qwen2.5-3b_p100 qwen2.5-7b_a30 \
        --output-dir models/cara/xgboost/
"""

import argparse
import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_latency_data(data_dir: str, instance_types: Optional[List[str]] = None) -> Dict[str, list]:
    """Load latency benchmark JSONL files, grouped by instance_type.

    Args:
        data_dir: Directory containing JSONL files.
        instance_types: If specified, only load these instance types.

    Returns:
        Dict mapping instance_type to list of record dicts.
    """
    data_path = Path(data_dir)
    records_by_type = defaultdict(list)

    # Read all JSONL files
    jsonl_files = list(data_path.glob("**/*.jsonl"))
    if not jsonl_files:
        # Also try .json files
        jsonl_files = list(data_path.glob("**/*.json"))

    logger.info(f"Found {len(jsonl_files)} data files in {data_dir}")

    for fpath in jsonl_files:
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Determine instance type
                inst_type = record.get("instance_type")
                if not inst_type:
                    # Try to infer from model + host
                    model = record.get("model", "unknown")
                    host = record.get("host", "unknown")
                    inst_type = f"{model}_{host}"

                if instance_types and inst_type not in instance_types:
                    continue

                # Skip failed requests
                if not record.get("success", True):
                    continue

                records_by_type[inst_type].append(record)

    for inst_type, records in records_by_type.items():
        logger.info(f"  {inst_type}: {len(records)} records")

    return dict(records_by_type)


def prepare_features_and_targets(
    records: list, target: str = "e2el"
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert records to feature matrix and target vector.

    Args:
        records: List of latency record dicts.
        target: Target variable name (e2el, ttft, server_latency).

    Returns:
        (X, y) where X is (N, num_features) and y is (N,).
    """
    from block.predictor.cara.estimators.xgboost_predictor import (
        ALL_FEATURES,
        build_feature_vector,
    )

    X_list = []
    y_list = []

    for rec in records:
        # Get target value
        target_val = rec.get(target) or rec.get("actual_e2e_latency")
        if target_val is None or target_val <= 0:
            continue

        # Get schedule state
        schedule_state = rec.get("schedule_state", {})
        # Fallback: if state is stored flat in record
        if not schedule_state:
            schedule_state = {k: rec.get(k, 0) for k in [
                "ema_decode_tok_per_s", "ema_prefill_tok_per_s", "ema_decode_iter_ms",
                "decode_ctx_p50", "decode_ctx_p95", "decode_ctx_max",
                "num_running", "num_active_decode_seqs", "num_waiting",
                "pending_prefill_tokens", "pending_decode_tokens",
                "token_budget_per_iter", "prefill_chunk_size", "max_num_seqs",
                "kv_cache_utilization", "kv_free_blocks", "kv_evictions_per_s",
            ]}

        num_prompt = rec.get("num_prompt_tokens") or rec.get("input_len", 0)
        num_output = rec.get("num_predicted_output_tokens") or rec.get("max_tokens") or rec.get("output_len", 0)

        fv = build_feature_vector(schedule_state, int(num_prompt), int(num_output))
        X_list.append(fv)
        y_list.append(float(target_val))

    X = np.stack(X_list) if X_list else np.empty((0, len(ALL_FEATURES)))
    y = np.array(y_list, dtype=np.float32)

    return X, y


def train_xgboost_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: List[str],
    params: Optional[Dict] = None,
) -> Tuple[object, Dict]:
    """Train a single XGBoost model.

    Returns (model, metrics_dict).
    """
    import xgboost as xgb

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)

    if params is None:
        params = {
            "objective": "reg:squarederror",
            "eval_metric": ["mae", "rmse"],
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
            "seed": 42,
        }

    evals_result = {}
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=[(dtrain, "train"), (dval, "val")],
        evals_result=evals_result,
        early_stopping_rounds=30,
        verbose_eval=50,
    )

    # Evaluate on validation set
    preds = model.predict(dval)
    errors = np.abs(y_val - preds)
    rel_errors = errors / np.maximum(y_val, 1e-6)

    metrics = {
        "mae": float(np.mean(errors)),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "mape": float(np.mean(rel_errors) * 100),
        "p50_error": float(np.percentile(errors, 50)),
        "p95_error": float(np.percentile(errors, 95)),
        "p99_error": float(np.percentile(errors, 99)),
        "under_prediction_rate": float(np.mean(preds < y_val)),
        "n_train": len(X_train),
        "n_val": len(X_val),
        "best_iteration": model.best_iteration,
    }

    # Feature importance
    importance = model.get_score(importance_type="gain")
    top_features = sorted(importance.items(), key=lambda x: -x[1])[:10]
    metrics["top_features"] = {k: float(v) for k, v in top_features}

    return model, metrics


def main():
    parser = argparse.ArgumentParser(
        description="Train XGBoost latency predictor for CARA"
    )
    parser.add_argument("--data-dir", required=True, help="Latency data directory (JSONL)")
    parser.add_argument(
        "--instance-types",
        nargs="+",
        default=None,
        help="Instance types to train for (default: all found in data)",
    )
    parser.add_argument(
        "--target",
        default="e2el",
        help="Target variable (e2el, ttft, server_latency)",
    )
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split")
    parser.add_argument("--output-dir", default="models/cara/xgboost")
    args = parser.parse_args()

    from block.predictor.cara.estimators.xgboost_predictor import (
        ALL_FEATURES,
        XGBoostLatencyPredictor,
    )

    # Load data
    records_by_type = load_latency_data(args.data_dir, args.instance_types)

    if not records_by_type:
        logger.error("No latency data found!")
        return

    predictor = XGBoostLatencyPredictor()
    all_metrics = {}

    for inst_type, records in sorted(records_by_type.items()):
        logger.info(f"\n{'='*60}")
        logger.info(f"Training model for {inst_type} ({len(records)} records)")
        logger.info(f"{'='*60}")

        if len(records) < 50:
            logger.warning(f"Too few records for {inst_type} ({len(records)}), skipping")
            continue

        # Prepare features
        X, y = prepare_features_and_targets(records, target=args.target)
        if len(X) < 50:
            logger.warning(f"Too few valid records for {inst_type} ({len(X)}), skipping")
            continue

        # Train/val split (time-based if timestamps available)
        n_val = int(len(X) * args.val_split)
        n_train = len(X) - n_val

        # Shuffle and split
        indices = np.random.RandomState(42).permutation(len(X))
        train_idx = indices[:n_train]
        val_idx = indices[n_train:]

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        t0 = time.time()
        model, metrics = train_xgboost_model(
            X_train, y_train, X_val, y_val, feature_names=ALL_FEATURES
        )
        train_time = time.time() - t0

        predictor.models[inst_type] = model
        metrics["train_time_s"] = train_time
        all_metrics[inst_type] = metrics

        logger.info(
            f"  {inst_type}: MAE={metrics['mae']:.4f}s, "
            f"MAPE={metrics['mape']:.1f}%, "
            f"RMSE={metrics['rmse']:.4f}s, "
            f"trained in {train_time:.1f}s"
        )
        logger.info(f"  Top features: {list(metrics['top_features'].keys())[:5]}")

    # Save
    predictor.save(args.output_dir)

    # Save metrics
    output_path = Path(args.output_dir)
    with open(output_path / "training_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    # Print summary
    print("\n" + "=" * 70)
    print("XGBOOST LATENCY PREDICTOR TRAINING RESULTS")
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
