#!/usr/bin/env python3
"""
Train MLP-based length and quality estimator for CARA.

Uses frozen sentence-transformer embeddings + per-model MLP heads.
Supports quantile regression for conservative length bounds (Q90/Q95).

Usage:
    python -m block.predictor.cara.offline_training.train_mlp_predictor \
        --input data/cara/training_data/cara_v3_all_train.json \
        --test-input data/cara/training_data/cara_v3_all_test.json \
        --output-dir models/cara/mlp/ \
        --epochs 50 --lr 1e-3 --device cpu
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

from block.predictor.cara.estimators.mlp_estimator import MLPEstimator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def evaluate(estimator: MLPEstimator, test_data: list) -> dict:
    """Evaluate estimator on test data.

    Returns dict with per-model metrics for both length and quality.
    """
    from collections import defaultdict

    length_errors = defaultdict(list)
    quality_errors = defaultdict(list)
    quality_rankings = []

    for req in test_data:
        prompt = req["prompt"]
        predictions = estimator.predict_all_models(prompt)

        true_qualities = {}
        pred_qualities = {}
        for model_name, true_vals in req["models"].items():
            if model_name not in predictions:
                continue
            pred = predictions[model_name]

            # Length
            true_len = true_vals["output_length"]
            pred_len = pred["length_mean"]
            length_errors[model_name].append(abs(true_len - pred_len))

            # Quality
            true_q = MLPEstimator._extract_quality(true_vals)
            pred_q = pred["quality_score"]
            quality_errors[model_name].append(abs(true_q - pred_q))

            true_qualities[model_name] = true_q
            pred_qualities[model_name] = pred_q

        if true_qualities and pred_qualities:
            true_best = max(true_qualities, key=true_qualities.get)
            pred_best = max(pred_qualities, key=pred_qualities.get)
            quality_rankings.append((true_best, pred_best))

    results = {"length": {}, "quality": {}}

    for model in sorted(length_errors.keys()):
        errs = length_errors[model]
        results["length"][model] = {
            "mae": float(np.mean(errs)),
            "median_ae": float(np.median(errs)),
            "acc_50": float(np.mean([1 if e <= 50 else 0 for e in errs])),
            "acc_100": float(np.mean([1 if e <= 100 else 0 for e in errs])),
            "n": len(errs),
        }

    for model in sorted(quality_errors.keys()):
        errs = quality_errors[model]
        results["quality"][model] = {
            "mae": float(np.mean(errs)),
            "n": len(errs),
        }

    if quality_rankings:
        correct = sum(1 for t, p in quality_rankings if t == p)
        results["quality"]["best_model_accuracy"] = correct / len(quality_rankings)
        results["quality"]["n_rankings"] = len(quality_rankings)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train MLP estimator for CARA length/quality prediction"
    )
    parser.add_argument("--input", required=True, help="Training data JSON")
    parser.add_argument("--test-input", default=None, help="Test data JSON")
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence transformer model for embeddings",
    )
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=[256, 128],
        help="Hidden layer dimensions",
    )
    parser.add_argument(
        "--quantiles",
        type=float,
        nargs="+",
        default=[0.9, 0.95],
        help="Quantiles for length prediction (e.g., 0.9 0.95)",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=256, help="Training batch size")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--output-dir", default="models/cara/mlp", help="Output directory")
    parser.add_argument("--device", default="cpu", help="Device for training")
    args = parser.parse_args()

    # Load training data
    with open(args.input) as f:
        if args.input.endswith(".jsonl"):
            train_data = [json.loads(line) for line in f]
        else:
            raw = json.load(f)
            train_data = raw["requests"] if "requests" in raw else raw
    logger.info(f"Training data: {len(train_data)} requests")

    # Train
    t0 = time.time()
    estimator = MLPEstimator(
        embedding_model_name=args.embedding_model,
        hidden_dims=args.hidden_dims,
        quantiles=args.quantiles,
        device=args.device,
    )
    history = estimator.train(
        train_data,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        val_split=args.val_split,
    )
    train_time = time.time() - t0
    logger.info(f"Training completed in {train_time:.1f}s")

    # Save
    estimator.save(args.output_dir)

    # Save training history
    history_path = Path(args.output_dir) / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    # Evaluate
    if args.test_input:
        with open(args.test_input) as f:
            if args.test_input.endswith(".jsonl"):
                test_data = [json.loads(line) for line in f]
            else:
                raw = json.load(f)
                test_data = raw["requests"] if "requests" in raw else raw
        logger.info(f"Test data: {len(test_data)} requests")

        t0 = time.time()
        results = evaluate(estimator, test_data)
        eval_time = time.time() - t0

        print("\n" + "=" * 70)
        print("EVALUATION RESULTS (MLP)")
        print("=" * 70)

        print("\nLength Prediction (MAE / Acc@50 / Acc@100):")
        for model, metrics in sorted(results["length"].items()):
            print(
                f"  {model}: MAE={metrics['mae']:.1f}, "
                f"Acc@50={metrics['acc_50']:.3f}, "
                f"Acc@100={metrics['acc_100']:.3f} "
                f"(n={metrics['n']})"
            )

        print("\nQuality Prediction (MAE):")
        for model, metrics in sorted(results["quality"].items()):
            if isinstance(metrics, dict) and "mae" in metrics:
                print(f"  {model}: MAE={metrics['mae']:.4f} (n={metrics['n']})")

        if "best_model_accuracy" in results["quality"]:
            print(
                f"\nBest Model Selection Accuracy: "
                f"{results['quality']['best_model_accuracy']:.3f} "
                f"(n={results['quality']['n_rankings']})"
            )

        print(f"\nEvaluation time: {eval_time:.1f}s")
        print(f"Inference speed: {eval_time / len(test_data) * 1000:.1f}ms per query")

        eval_path = Path(args.output_dir) / "evaluation_results.json"
        with open(eval_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Evaluation results saved to {eval_path}")
    else:
        logger.info("No test data provided. Skipping evaluation.")


if __name__ == "__main__":
    main()
