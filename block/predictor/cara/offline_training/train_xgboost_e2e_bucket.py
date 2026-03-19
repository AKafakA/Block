#!/usr/bin/env python3
"""
Train XGBoost E2E latency predictor using bucket classifier's E[length] predictions.

This implements the full inference pipeline:
  prompt → bucket classifier → E[length] → XGBoost → E2E latency

This is the realistic serving-time setup where we don't know the actual output
length at scheduling time, and instead use the bucket classifier's expected
output length as the `num_predicted_output_tokens` feature for XGBoost.

Usage:
    python -m block.predictor.cara.offline_training.train_xgboost_e2e_bucket \\
        --data-dir /mydata/latency_training_v2 \\
        --bucket-model-dir /mydata/models/cara \\
        --output-dir /mydata/models/cara/xgboost_e2e_bucket \\
        --batch-size 128 --device cuda

Compares three setups:
  1. actual_tokens: Oracle (uses ground-truth output length)
  2. bucket_predicted: Bucket classifier E[length] (realistic)
  3. max_tokens: Fixed 1024 (naive baseline)

The bucket_predicted MAPE is the number that goes in the paper as the
pipeline E2E latency prediction accuracy.
"""

import argparse
import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Bucket hyperparameters (must match training)
BUCKET_SIZE = 64
NUM_BUCKETS = 16
MAX_LENGTH = 1024  # NUM_BUCKETS * BUCKET_SIZE
BUCKET_MIDPOINTS = np.array([i * BUCKET_SIZE + BUCKET_SIZE // 2 for i in range(NUM_BUCKETS)], dtype=float)
# [32, 96, 160, 224, 288, 352, 416, 480, 544, 608, 672, 736, 800, 864, 928, 992]

# Model size key in checkpoint directory name
MODEL_SIZE_PATTERNS = {
    "3b": ["3b", "qwen2.5-3b", "qwen2-5-3b"],
    "7b": ["7b", "qwen2.5-7b", "qwen2-5-7b"],
    "14b": ["14b", "qwen2.5-14b", "qwen2-5-14b"],
    "72b": ["72b", "qwen2.5-72b", "qwen2-5-72b"],
}


def find_bucket_checkpoint(bucket_model_dir: str, model_size: str) -> Optional[str]:
    """Find bucket classifier checkpoint for a given model size.

    Looks for directories like:
      bucket_3b/  bucket_qwen2.5-3b/  bucket_3b_final/  etc.
    """
    base = Path(bucket_model_dir)
    patterns = MODEL_SIZE_PATTERNS.get(model_size, [model_size])

    candidates = []
    for p in patterns:
        candidates += list(base.glob(f"*bucket*{p}*"))
        candidates += list(base.glob(f"*{p}*bucket*"))

    # Among candidates, prefer final/best checkpoint
    for candidate in sorted(candidates, key=lambda x: x.name):
        # Check for checkpoint subdirectory
        for sub in ["checkpoint-final", "best", ""]:
            ckpt_path = candidate / sub if sub else candidate
            if (ckpt_path / "config.json").exists():
                return str(ckpt_path)

    logger.warning(f"No bucket checkpoint found for model_size={model_size} in {bucket_model_dir}")
    logger.warning(f"  Searched: {[str(p) for p in candidates]}")
    return None


def load_bucket_classifier(checkpoint_path: str, device: str = "cuda"):
    """Load a trained bucket classifier."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    logger.info(f"Loading bucket classifier from {checkpoint_path}")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint_path, num_labels=NUM_BUCKETS
    )
    model.eval()
    model = model.to(device)
    return model, tokenizer


@torch.no_grad()
def predict_expected_length_batch(
    prompts: List[str],
    model,
    tokenizer,
    device: str = "cuda",
    batch_size: int = 64,
) -> np.ndarray:
    """Run bucket classifier on prompts, return E[length] for each.

    E[length] = sum_k(softmax(logits)[k] * midpoint_k)

    Returns:
        Array of shape (N,) with expected output length in tokens.
    """
    all_expected = []
    midpoints_t = torch.tensor(BUCKET_MIDPOINTS, dtype=torch.float32, device=device)

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        enc = tokenizer(
            batch,
            truncation=True,
            max_length=1024,
            padding=True,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits  # (B, NUM_BUCKETS)
        probs = torch.softmax(logits, dim=-1)  # (B, NUM_BUCKETS)
        expected = (probs * midpoints_t).sum(dim=-1)  # (B,)
        all_expected.append(expected.cpu().numpy())

    return np.concatenate(all_expected)


def extract_model_size(instance_type: str) -> Optional[str]:
    """Extract model size (3b/7b/14b/72b) from instance_type like 'qwen2.5-7b_a30'."""
    it = instance_type.lower()
    for size in ["72b", "14b", "7b", "3b"]:  # longest first to avoid 7b matching 72b
        if size in it:
            return size
    return None


def load_latency_records(data_dir: str) -> Dict[str, List[dict]]:
    """Load latency records from JSONL files, grouped by instance_type.

    Expects records with fields:
        prompt, num_prompt_tokens, actual_output_tokens,
        schedule_state, instance_type, e2el/actual_e2e_latency
    """
    data_path = Path(data_dir)
    records_by_type = defaultdict(list)

    jsonl_files = sorted(data_path.glob("**/*.jsonl"))
    if not jsonl_files:
        jsonl_files = sorted(data_path.glob("**/*.json"))

    logger.info(f"Found {len(jsonl_files)} files in {data_dir}")

    for fpath in jsonl_files:
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if not rec.get("success", True):
                    continue

                inst_type = rec.get("instance_type")
                if not inst_type:
                    continue

                # Must have e2e latency
                e2el = rec.get("e2el") or rec.get("actual_e2e_latency")
                if not e2el or e2el <= 0:
                    continue

                # Must have prompt (for bucket inference) and actual output tokens
                if not rec.get("prompt") or not rec.get("actual_output_tokens"):
                    continue

                records_by_type[inst_type].append(rec)

    for inst_type, recs in records_by_type.items():
        logger.info(f"  {inst_type}: {len(recs)} records with prompt+actual_tokens")

    return dict(records_by_type)


def build_feature_matrix(
    records: List[dict],
    output_tokens_override: Optional[np.ndarray] = None,
    use_max_tokens: bool = False,
    max_tokens: int = 1024,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build XGBoost feature matrix from records.

    Args:
        output_tokens_override: If given, use these as num_predicted_output_tokens.
        use_max_tokens: If True, use fixed max_tokens (naive baseline).

    Returns:
        (X, y) arrays.
    """
    from block.predictor.cara.estimators.xgboost_predictor import (
        ALL_FEATURES,
        build_feature_vector,
    )

    X_list = []
    y_list = []
    valid_mask = []

    for i, rec in enumerate(records):
        e2el = rec.get("e2el") or rec.get("actual_e2e_latency")
        if not e2el or e2el <= 0:
            valid_mask.append(False)
            continue

        schedule_state = rec.get("schedule_state", {})
        if not schedule_state:
            valid_mask.append(False)
            continue

        num_prompt = rec.get("num_prompt_tokens") or rec.get("prompt_len", 0)

        if use_max_tokens:
            num_output = max_tokens
        elif output_tokens_override is not None:
            num_output = float(output_tokens_override[i])
        else:
            num_output = rec.get("actual_output_tokens") or rec.get("num_predicted_output_tokens", 0)

        num_output = max(1, int(round(num_output)))

        fv = build_feature_vector(schedule_state, int(num_prompt), num_output)
        X_list.append(fv)
        y_list.append(float(e2el))
        valid_mask.append(True)

    X = np.stack(X_list) if X_list else np.empty((0, len(ALL_FEATURES)))
    y = np.array(y_list, dtype=np.float32)
    return X, y


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: List[str],
) -> Tuple[object, dict]:
    """Train XGBoost regressor and return (model, metrics)."""
    import xgboost as xgb

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
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=30,
        verbose_eval=50,
    )

    preds = model.predict(dval)
    errors = np.abs(y_val - preds)
    rel_errors = errors / np.maximum(y_val, 1e-6)

    metrics = {
        "mae": float(np.mean(errors)),
        "rmse": float(np.sqrt(np.mean((y_val - preds) ** 2))),
        "mape": float(np.mean(rel_errors) * 100),
        "p50_ae": float(np.percentile(errors, 50)),
        "p95_ae": float(np.percentile(errors, 95)),
        "n_train": len(X_train),
        "n_val": len(X_val),
    }

    importance = model.get_score(importance_type="gain")
    top_features = sorted(importance.items(), key=lambda x: -x[1])[:5]
    metrics["top_features"] = [k for k, _ in top_features]

    return model, metrics


def main():
    parser = argparse.ArgumentParser(
        description="Train XGBoost E2E predictor using bucket classifier E[length]"
    )
    parser.add_argument("--data-dir", required=True, help="Latency training data dir (v2, with actual_output_tokens + prompt)")
    parser.add_argument("--bucket-model-dir", required=True, help="Directory containing bucket classifier checkpoints")
    parser.add_argument("--output-dir", default="/mydata/models/cara/xgboost_e2e_bucket", help="Output directory for models")
    parser.add_argument("--device", default="cuda", help="Device for bucket classifier inference")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for bucket inference")
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--instance-types", nargs="+", default=None, help="Filter to specific instance types")
    args = parser.parse_args()

    from block.predictor.cara.estimators.xgboost_predictor import (
        ALL_FEATURES,
        XGBoostLatencyPredictor,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load latency data ---
    records_by_type = load_latency_records(args.data_dir)

    if args.instance_types:
        records_by_type = {k: v for k, v in records_by_type.items() if k in args.instance_types}

    if not records_by_type:
        logger.error("No valid latency records found!")
        return

    # --- Load bucket classifiers (one per model size) ---
    bucket_models = {}  # model_size -> (model, tokenizer)
    model_sizes_needed = set()
    for inst_type in records_by_type:
        sz = extract_model_size(inst_type)
        if sz:
            model_sizes_needed.add(sz)

    logger.info(f"Model sizes needed: {sorted(model_sizes_needed)}")

    for size in sorted(model_sizes_needed):
        ckpt = find_bucket_checkpoint(args.bucket_model_dir, size)
        if ckpt:
            try:
                model, tokenizer = load_bucket_classifier(ckpt, args.device)
                bucket_models[size] = (model, tokenizer)
                logger.info(f"  Loaded {size} bucket classifier from {ckpt}")
            except Exception as e:
                logger.warning(f"  Failed to load {size} bucket classifier: {e}")
        else:
            logger.warning(f"  No checkpoint found for {size}, will skip bucket inference for {size} instances")

    # --- Process each instance type ---
    all_metrics = {}
    predictor_actual = XGBoostLatencyPredictor()
    predictor_bucket = XGBoostLatencyPredictor()
    predictor_maxtok = XGBoostLatencyPredictor()

    for inst_type, records in sorted(records_by_type.items()):
        if len(records) < 50:
            logger.warning(f"Skipping {inst_type}: only {len(records)} records")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {inst_type} ({len(records)} records)")
        logger.info(f"{'='*60}")

        model_size = extract_model_size(inst_type)

        # --- Run bucket classifier inference ---
        bucket_predicted = None
        if model_size and model_size in bucket_models:
            bm, bt = bucket_models[model_size]
            prompts = [rec["prompt"] for rec in records]
            logger.info(f"  Running bucket classifier inference on {len(prompts)} prompts...")
            t0 = time.time()
            bucket_predicted = predict_expected_length_batch(
                prompts, bm, bt, device=args.device, batch_size=args.batch_size
            )
            logger.info(f"  Inference done in {time.time()-t0:.1f}s. E[len] mean={bucket_predicted.mean():.1f}")

            # Log actual vs predicted stats
            actual_lens = np.array([rec.get("actual_output_tokens", 0) for rec in records], dtype=float)
            bucket_mape = np.mean(np.abs(bucket_predicted - actual_lens) / np.maximum(actual_lens, 1)) * 100
            logger.info(f"  Bucket E[len] MAPE vs actual: {bucket_mape:.1f}%")
        else:
            logger.warning(f"  No bucket classifier for {inst_type}, skipping bucket variant")

        # --- Train/val split ---
        n = len(records)
        idx = np.random.RandomState(42).permutation(n)
        n_val = int(n * args.val_split)
        train_idx, val_idx = idx[n_val:], idx[:n_val]

        train_recs = [records[i] for i in train_idx]
        val_recs = [records[i] for i in val_idx]

        metrics_inst = {}

        # --- Variant 1: actual output tokens (oracle) ---
        X_tr, y_tr = build_feature_matrix(train_recs)
        X_val, y_val = build_feature_matrix(val_recs)
        if len(X_tr) >= 50:
            t0 = time.time()
            model_a, m_a = train_xgboost(X_tr, y_tr, X_val, y_val, ALL_FEATURES)
            logger.info(f"  [oracle/actual] MAE={m_a['mae']:.4f}s, MAPE={m_a['mape']:.1f}%, trained in {time.time()-t0:.1f}s")
            logger.info(f"    top features: {m_a['top_features']}")
            predictor_actual.models[inst_type] = model_a
            metrics_inst["actual"] = m_a

        # --- Variant 2: bucket E[length] ---
        if bucket_predicted is not None:
            bucket_train = bucket_predicted[train_idx]
            bucket_val = bucket_predicted[val_idx]
            X_tr_b, y_tr_b = build_feature_matrix(train_recs, output_tokens_override=bucket_train)
            X_val_b, y_val_b = build_feature_matrix(val_recs, output_tokens_override=bucket_val)
            if len(X_tr_b) >= 50:
                t0 = time.time()
                model_b, m_b = train_xgboost(X_tr_b, y_tr_b, X_val_b, y_val_b, ALL_FEATURES)
                logger.info(f"  [bucket E[len]] MAE={m_b['mae']:.4f}s, MAPE={m_b['mape']:.1f}%, trained in {time.time()-t0:.1f}s")
                predictor_bucket.models[inst_type] = model_b
                metrics_inst["bucket_predicted"] = m_b

        # --- Variant 3: max_tokens=1024 (naive baseline) ---
        X_tr_m, y_tr_m = build_feature_matrix(train_recs, use_max_tokens=True)
        X_val_m, y_val_m = build_feature_matrix(val_recs, use_max_tokens=True)
        if len(X_tr_m) >= 50:
            t0 = time.time()
            model_m, m_m = train_xgboost(X_tr_m, y_tr_m, X_val_m, y_val_m, ALL_FEATURES)
            logger.info(f"  [max_tokens=1024] MAE={m_m['mae']:.4f}s, MAPE={m_m['mape']:.1f}%, trained in {time.time()-t0:.1f}s")
            predictor_maxtok.models[inst_type] = model_m
            metrics_inst["max_tokens"] = m_m

        all_metrics[inst_type] = metrics_inst

    # --- Save models ---
    if predictor_actual.models:
        predictor_actual.save(str(output_dir / "actual"))
    if predictor_bucket.models:
        predictor_bucket.save(str(output_dir / "bucket_predicted"))
    if predictor_maxtok.models:
        predictor_maxtok.save(str(output_dir / "max_tokens"))

    # --- Save metrics ---
    with open(output_dir / "comparison_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    # --- Print comparison table ---
    print("\n" + "=" * 80)
    print("E2E LATENCY PREDICTION: COMPARISON TABLE")
    print("(oracle=actual tokens, bucket=E[length] from classifier, naive=max_tokens=1024)")
    print("=" * 80)
    print(f"{'Instance':<22} {'Oracle MAPE':>12} {'Bucket MAPE':>12} {'Naive MAPE':>12}  {'Bucket MAE':>10}")
    print("-" * 80)
    for inst_type, m in sorted(all_metrics.items()):
        oracle_mape = m.get("actual", {}).get("mape", float("nan"))
        bucket_mape = m.get("bucket_predicted", {}).get("mape", float("nan"))
        naive_mape = m.get("max_tokens", {}).get("mape", float("nan"))
        bucket_mae = m.get("bucket_predicted", {}).get("mae", float("nan"))
        print(f"  {inst_type:<20} {oracle_mape:>11.1f}% {bucket_mape:>11.1f}% {naive_mape:>11.1f}%  {bucket_mae:>9.4f}s")

    print("\nKey result: 'Bucket MAPE' = pipeline E2E accuracy for paper")
    print(f"Saved models to: {output_dir}")


if __name__ == "__main__":
    main()
