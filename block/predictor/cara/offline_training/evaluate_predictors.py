#!/usr/bin/env python3
"""
Unified evaluation script for ALL CARA length and quality predictors.

Supports: KNN, MLP, encoder models (ModernBERT, DeBERTaV3, RoBERTa),
and LLM-based (Qwen-0.5B+LoRA) predictors.

Reports per-model and aggregate: MAE, MAPE, Acc@50, Acc@100,
Spearman rank correlation, best-model selection accuracy.

Usage:
    python -m block.predictor.cara.offline_training.evaluate_predictors \
        --test-input data/cara/training_data/cara_v3_all_training_test.json \
        --train-input data/cara/training_data/cara_v3_all_training_train.json \
        --predictors knn:models/cara/knn \
                     mlp:models/cara/mlp \
                     modernbert:models/cara/modernbert_length \
                     debertav3:models/cara/debertav3_length \
                     roberta:models/cara/roberta_length \
                     qwen05b:models/cara/qwen05b_length \
        --target length \
        --device cuda
"""

import argparse
import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _extract_quality(m_data: dict) -> float:
    """Extract combined quality score from model data (old or new schema)."""
    if "quality_score" in m_data:
        return float(m_data["quality_score"])
    sim = m_data.get("similarity_score")
    judge_scores = m_data.get("llm_judge_scores", {})
    valid_judges = [v for v in judge_scores.values() if v is not None]
    judge_mean = sum(valid_judges) / len(valid_judges) if valid_judges else None
    if sim is not None and judge_mean is not None:
        return 0.5 * float(sim) + 0.5 * float(judge_mean)
    elif sim is not None:
        return float(sim)
    elif judge_mean is not None:
        return float(judge_mean)
    return 0.0


# ---------------------------------------------------------------------------
# Predictor adapters — each returns dict: {model_name: predicted_value}
# ---------------------------------------------------------------------------

class KNNAdapter:
    def __init__(self, model_dir: str, train_data: list, target: str, device: str):
        from block.predictor.cara.estimators.knn_estimator import KNNEstimator
        self.est = KNNEstimator(device=device)
        self.est.build_index(train_data)
        self.target = target

    def predict(self, prompt: str, target_models: list) -> Dict[str, float]:
        result = self.est.predict_all_models(prompt)
        out = {}
        for m in target_models:
            if m in result:
                if self.target == "length":
                    out[m] = result[m].get("length_mean", 0)
                else:
                    out[m] = result[m].get("quality_score", 0)
        return out


class MLPAdapter:
    def __init__(self, model_dir: str, target: str, device: str):
        from block.predictor.cara.estimators.mlp_estimator import MLPEstimator
        self.est = MLPEstimator.load(model_dir)
        self.target = target

    def predict(self, prompt: str, target_models: list) -> Dict[str, float]:
        result = self.est.predict_all_models(prompt)
        out = {}
        for m in target_models:
            if m in result:
                if self.target == "length":
                    out[m] = result[m].get("length_mean", 0)
                else:
                    out[m] = result[m].get("quality_score", 0)
        return out


class EncoderAdapter:
    """Adapter for encoder-based regression (ModernBERT, DeBERTaV3, RoBERTa)."""

    def __init__(self, model_dir: str, target: str, device: str):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = device
        self.predictors = {}
        model_path = Path(model_dir)

        # Each subdirectory is a per-target-model predictor
        for subdir in sorted(model_path.iterdir()):
            if not subdir.is_dir() or subdir.name.startswith("."):
                continue
            # Reconstruct model name from directory name
            # e.g., "Qwen_Qwen2.5-3B_length" -> "Qwen/Qwen2.5-3B"
            name_parts = subdir.name.replace(f"_{target}", "")
            model_name = name_parts.replace("_", "/", 1)  # first _ -> /

            try:
                tokenizer = AutoTokenizer.from_pretrained(str(subdir), trust_remote_code=True)
                model = AutoModelForSequenceClassification.from_pretrained(
                    str(subdir), trust_remote_code=True
                ).to(device).eval()
                self.predictors[model_name] = (tokenizer, model)
                logger.info(f"  Loaded encoder for {model_name}")
            except Exception as e:
                logger.warning(f"  Failed to load {subdir.name}: {e}")

    def predict(self, prompt: str, target_models: list) -> Dict[str, float]:
        import torch
        out = {}
        for m in target_models:
            if m not in self.predictors:
                continue
            tokenizer, model = self.predictors[m]
            inputs = tokenizer(
                prompt, truncation=True, max_length=1024, return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                output = model(**inputs)
            out[m] = float(output.logits.squeeze().cpu())
        return out


class BucketEncoderAdapter:
    """Adapter for bucket classification encoder (ModernBERT etc.)."""

    def __init__(self, model_dir: str, target: str, device: str,
                 bucket_size: int = 64):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = device
        self.bucket_size = bucket_size
        self.predictors = {}
        model_path = Path(model_dir)

        for subdir in sorted(model_path.iterdir()):
            if not subdir.is_dir() or subdir.name.startswith("."):
                continue
            name_parts = subdir.name.replace(f"_{target}", "")
            model_name = name_parts.replace("_", "/", 1)

            try:
                tokenizer = AutoTokenizer.from_pretrained(str(subdir), trust_remote_code=True)
                model = AutoModelForSequenceClassification.from_pretrained(
                    str(subdir), trust_remote_code=True
                ).to(device).eval()
                self.predictors[model_name] = (tokenizer, model)
                logger.info(f"  Loaded bucket encoder for {model_name}")
            except Exception as e:
                logger.warning(f"  Failed to load {subdir.name}: {e}")

    def predict(self, prompt: str, target_models: list) -> Dict[str, float]:
        """Predict expected length from bucket probabilities."""
        import torch
        out = {}
        for m in target_models:
            if m not in self.predictors:
                continue
            tokenizer, model = self.predictors[m]
            inputs = tokenizer(
                prompt, truncation=True, max_length=1024, return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                output = model(**inputs)
            # Softmax to get probabilities
            probs = torch.softmax(output.logits.squeeze(), dim=-1).cpu().numpy()
            # Expected value from bucket midpoints
            midpoints = np.array([(i * self.bucket_size + self.bucket_size / 2)
                                  for i in range(len(probs))])
            expected_length = float(np.sum(probs * midpoints))
            out[m] = expected_length
        return out

    def predict_buckets(self, prompt: str, target_models: list) -> Dict[str, np.ndarray]:
        """Predict full bucket probability distribution."""
        import torch
        out = {}
        for m in target_models:
            if m not in self.predictors:
                continue
            tokenizer, model = self.predictors[m]
            inputs = tokenizer(
                prompt, truncation=True, max_length=1024, return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                output = model(**inputs)
            probs = torch.softmax(output.logits.squeeze(), dim=-1).cpu().numpy()
            out[m] = probs
        return out


def compute_bucket_metrics(
    actual_lengths: np.ndarray, bucket_probs_list: list,
    bucket_size: int = 64
) -> Dict[str, float]:
    """Compute bucket classification metrics including distribution quality.

    Args:
        actual_lengths: Array of actual output token counts.
        bucket_probs_list: List of probability arrays (one per sample).
        bucket_size: Size of each bucket.
    """
    num_buckets = len(bucket_probs_list[0])
    midpoints = np.array([(i * bucket_size + bucket_size / 2) for i in range(num_buckets)])

    actual_buckets = np.clip(actual_lengths.astype(int) // bucket_size, 0, num_buckets - 1)
    pred_buckets = np.array([np.argmax(p) for p in bucket_probs_list])
    expected_lengths = np.array([np.sum(p * midpoints) for p in bucket_probs_list])

    # --- Accuracy metrics ---
    exact_acc = float(np.mean(pred_buckets == actual_buckets))
    adjacent_acc = float(np.mean(np.abs(pred_buckets - actual_buckets) <= 1))

    # Top-K: actual bucket in top-K predicted buckets
    top1_hit = 0
    top3_hit = 0
    for actual_b, probs in zip(actual_buckets, bucket_probs_list):
        sorted_buckets = np.argsort(probs)[::-1]
        if actual_b == sorted_buckets[0]:
            top1_hit += 1
        if actual_b in sorted_buckets[:3]:
            top3_hit += 1
    top1_acc = float(top1_hit / len(actual_buckets))
    top3_acc = float(top3_hit / len(actual_buckets))

    # --- Distribution quality metrics ---
    # Probability assigned to actual bucket (higher = better calibrated)
    prob_at_actual = np.array([
        probs[int(ab)] for ab, probs in zip(actual_buckets, bucket_probs_list)
    ])
    mean_prob_actual = float(np.mean(prob_at_actual))

    # Negative log-likelihood (cross-entropy, lower = better)
    nll = -float(np.mean(np.log(np.clip(prob_at_actual, 1e-10, 1.0))))

    # Brier score (lower = better calibrated)
    brier = 0.0
    for actual_b, probs in zip(actual_buckets, bucket_probs_list):
        one_hot = np.zeros(num_buckets)
        one_hot[int(actual_b)] = 1.0
        brier += np.sum((probs - one_hot) ** 2)
    brier = float(brier / len(actual_buckets))

    # Confidence: mean max probability (higher = more confident)
    mean_confidence = float(np.mean([np.max(p) for p in bucket_probs_list]))

    # --- Token-level metrics from expected value ---
    errors = np.abs(actual_lengths - expected_lengths)
    non_zero = actual_lengths > 0
    mae = float(np.mean(errors))
    mape = float(np.mean(errors[non_zero] / actual_lengths[non_zero]) * 100) if non_zero.any() else 0.0

    return {
        "top1_accuracy": top1_acc,
        "top3_accuracy": top3_acc,
        "adjacent_accuracy": adjacent_acc,
        "mean_prob_at_actual": mean_prob_actual,
        "nll": nll,
        "brier_score": brier,
        "mean_confidence": mean_confidence,
        "expected_mae": mae,
        "expected_mape": mape,
        "n": int(len(actual_lengths)),
    }


class LLMAdapter:
    """Adapter for Qwen-0.5B + LoRA predictor."""

    def __init__(self, model_dir: str, target: str, device: str):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel

        model_path = Path(model_dir)
        metadata_file = model_path / "llm_predictor_metadata.json"
        self.metadata = {}
        if metadata_file.exists():
            with open(metadata_file) as f:
                self.metadata = json.load(f)

        base_model = self.metadata.get("base_model", "Qwen/Qwen2.5-0.5B")
        self.target = target
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        base = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype="auto", trust_remote_code=True
        ).to(device)
        self.model = PeftModel.from_pretrained(base, str(model_path)).eval()

    def predict(self, prompt: str, target_models: list) -> Dict[str, float]:
        import torch
        out = {}
        for m in target_models:
            model_short = m.split("/")[-1]
            input_text = f"Predict {self.target} for model {model_short}: {prompt}\nAnswer:"

            inputs = self.tokenizer(
                input_text, truncation=True, max_length=1024, return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, max_new_tokens=16, temperature=0.0,
                    do_sample=False, pad_token_id=self.tokenizer.eos_token_id
                )
            generated = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )
            try:
                value = float(generated.strip().split()[0].replace(",", ""))
                out[m] = value
            except (ValueError, IndexError):
                pass
        return out


# ---------------------------------------------------------------------------
# Evaluation logic
# ---------------------------------------------------------------------------

def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    """Compute all evaluation metrics."""
    errors = np.abs(actual - predicted)
    non_zero = actual > 0

    return {
        "mae": float(np.mean(errors)),
        "median_ae": float(np.median(errors)),
        "acc_50": float(np.mean(errors <= 50)),
        "acc_100": float(np.mean(errors <= 100)),
        "mape": float(np.mean(errors[non_zero] / actual[non_zero]) * 100)
        if non_zero.any() else 0.0,
        "spearman_r": float(stats.spearmanr(actual, predicted).correlation)
        if len(actual) > 2 else 0.0,
        "n": int(len(actual)),
    }


def evaluate_bucket_predictor(
    adapter: BucketEncoderAdapter, test_data: list, target_models: list, name: str,
    bucket_size: int = 64,
) -> Dict:
    """Evaluate a bucket classification predictor on all target models."""
    per_model = {}

    for tm in target_models:
        actual_lengths = []
        bucket_probs = []
        t0 = time.time()

        for req in test_data:
            m_data = req["models"].get(tm, {})
            if not m_data:
                continue
            actual = float(m_data.get("output_length", 0))
            probs = adapter.predict_buckets(req["prompt"], [tm])
            if tm in probs:
                actual_lengths.append(actual)
                bucket_probs.append(probs[tm])

        elapsed = time.time() - t0
        if not actual_lengths:
            continue

        metrics = compute_bucket_metrics(
            np.array(actual_lengths), bucket_probs, bucket_size
        )
        metrics["inference_time_s"] = elapsed
        metrics["samples_per_sec"] = len(actual_lengths) / elapsed if elapsed > 0 else 0

        ms = tm.split("/")[-1]
        per_model[tm] = metrics
        logger.info(
            f"  {ms}: top1={metrics['top1_accuracy']:.3f} "
            f"top3={metrics['top3_accuracy']:.3f} "
            f"adj={metrics['adjacent_accuracy']:.3f} "
            f"P(actual)={metrics['mean_prob_at_actual']:.3f} "
            f"NLL={metrics['nll']:.3f} "
            f"Brier={metrics['brier_score']:.3f} "
            f"exp_MAE={metrics['expected_mae']:.1f} "
            f"exp_MAPE={metrics['expected_mape']:.1f}% ({elapsed:.1f}s)"
        )

    if per_model:
        agg = {
            k: float(np.mean([m[k] for m in per_model.values()]))
            for k in ["top1_accuracy", "top3_accuracy", "adjacent_accuracy",
                       "mean_prob_at_actual", "nll", "brier_score",
                       "expected_mae", "expected_mape"]
        }
    else:
        agg = {}

    return {"name": name, "per_model": per_model, "aggregate": agg}


def evaluate_predictor(
    adapter, test_data: list, target_models: list, target: str, name: str
) -> Dict:
    """Evaluate a single predictor on all target models."""
    per_model = {}

    for tm in target_models:
        actual_vals = []
        pred_vals = []
        t0 = time.time()

        for req in test_data:
            m_data = req["models"].get(tm, {})
            if not m_data:
                continue

            if target == "length":
                actual = float(m_data.get("output_length", 0))
            else:
                actual = _extract_quality(m_data)

            preds = adapter.predict(req["prompt"], [tm])
            if tm in preds:
                actual_vals.append(actual)
                pred_vals.append(preds[tm])

        elapsed = time.time() - t0
        if not actual_vals:
            continue

        metrics = compute_metrics(np.array(actual_vals), np.array(pred_vals))
        metrics["inference_time_s"] = elapsed
        metrics["samples_per_sec"] = len(actual_vals) / elapsed if elapsed > 0 else 0
        per_model[tm] = metrics

        ms = tm.split("/")[-1]
        logger.info(
            f"  {ms}: MAE={metrics['mae']:.1f} MAPE={metrics['mape']:.1f}% "
            f"Acc@50={metrics['acc_50']:.3f} Acc@100={metrics['acc_100']:.3f} "
            f"ρ={metrics['spearman_r']:.3f} ({elapsed:.1f}s)"
        )

    # Aggregate
    if per_model:
        agg = {
            k: float(np.mean([m[k] for m in per_model.values()]))
            for k in ["mae", "mape", "acc_50", "acc_100", "spearman_r"]
        }
    else:
        agg = {}

    return {"name": name, "per_model": per_model, "aggregate": agg}


def print_comparison(all_results: list, target_models: list, target: str):
    """Print unified comparison table."""
    print(f"\n{'=' * 90}")
    print(f"  UNIFIED PREDICTOR COMPARISON — {target.upper()} PREDICTION")
    print(f"{'=' * 90}")

    # Aggregate table
    print(f"\n  {'Predictor':<15} {'MAE':>7} {'MAPE':>8} {'Acc@50':>8} {'Acc@100':>8} {'Spearman':>9}")
    print(f"  {'-' * 65}")
    for r in all_results:
        a = r["aggregate"]
        if not a:
            continue
        print(
            f"  {r['name']:<15} {a['mae']:>7.1f} {a['mape']:>7.1f}% "
            f"{a['acc_50']:>8.3f} {a['acc_100']:>8.3f} {a['spearman_r']:>9.3f}"
        )

    # Per-model detail
    for tm in target_models:
        ms = tm.split("/")[-1]
        print(f"\n  Target: {ms}")
        print(f"  {'Predictor':<15} {'MAE':>7} {'MAPE':>8} {'Acc@50':>8} {'Acc@100':>8} {'Spearman':>9} {'Speed':>10}")
        print(f"  {'-' * 75}")
        for r in all_results:
            m = r["per_model"].get(tm, {})
            if not m:
                continue
            speed = f"{m.get('samples_per_sec', 0):.0f}/s"
            print(
                f"  {r['name']:<15} {m['mae']:>7.1f} {m['mape']:>7.1f}% "
                f"{m['acc_50']:>8.3f} {m['acc_100']:>8.3f} "
                f"{m['spearman_r']:>9.3f} {speed:>10}"
            )


def main():
    parser = argparse.ArgumentParser(description="Unified predictor evaluation for CARA")
    parser.add_argument("--test-input", required=True, help="Test data JSON")
    parser.add_argument("--train-input", default=None, help="Training data (required for KNN)")
    parser.add_argument(
        "--predictors", nargs="+", required=True,
        help="type:path pairs (e.g., knn:models/cara/knn encoder:models/cara/modernbert_length)"
    )
    parser.add_argument("--target", choices=["length", "quality"], default="length")
    parser.add_argument("--target-models", nargs="+", default=None)
    parser.add_argument("--bucket-size", type=int, default=64,
                        help="Bucket size for bucket_encoder evaluation")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--output", default=None, help="Output JSON")
    args = parser.parse_args()

    # Load test data
    with open(args.test_input) as f:
        raw = json.load(f)
        test_data = raw["requests"] if "requests" in raw else raw
    if args.max_samples > 0:
        test_data = test_data[:args.max_samples]

    target_models = args.target_models or sorted(test_data[0]["models"].keys())
    logger.info(f"Test: {len(test_data)} requests, models: {[m.split('/')[-1] for m in target_models]}")

    # Load training data for KNN
    train_data = None
    if args.train_input:
        with open(args.train_input) as f:
            raw = json.load(f)
            train_data = raw["requests"] if "requests" in raw else raw

    # Build and evaluate each predictor
    all_results = []
    for spec in args.predictors:
        parts = spec.split(":", 1)
        if len(parts) != 2:
            logger.warning(f"Invalid spec: {spec}, expected type:path")
            continue
        pred_type, pred_path = parts

        logger.info(f"\nEvaluating: {pred_type} ({pred_path})")
        try:
            if pred_type == "knn":
                if not train_data:
                    logger.error("KNN needs --train-input")
                    continue
                adapter = KNNAdapter(pred_path, train_data, args.target, args.device)
            elif pred_type == "mlp":
                adapter = MLPAdapter(pred_path, args.target, args.device)
            elif pred_type in ("modernbert", "debertav3", "roberta", "encoder"):
                adapter = EncoderAdapter(pred_path, args.target, args.device)
            elif pred_type in ("qwen05b", "llm"):
                adapter = LLMAdapter(pred_path, args.target, args.device)
            elif pred_type in ("bucket", "bucket_encoder"):
                adapter = BucketEncoderAdapter(pred_path, args.target, args.device,
                                               bucket_size=args.bucket_size)
                result = evaluate_bucket_predictor(
                    adapter, test_data, target_models, pred_type,
                    bucket_size=args.bucket_size
                )
                all_results.append(result)
                continue  # skip the regression evaluate_predictor below
            else:
                logger.warning(f"Unknown type: {pred_type}")
                continue

            result = evaluate_predictor(adapter, test_data, target_models, args.target, pred_type)
            all_results.append(result)
        except Exception as e:
            logger.error(f"Failed {pred_type}: {e}")
            import traceback
            traceback.print_exc()

    if all_results:
        print_comparison(all_results, target_models, args.target)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
