#!/usr/bin/env python3
"""
Train KNN-based length and quality estimator for CARA.

Usage:
    python -m block.predictor.cara.offline_training.train_knn_estimator \
        --input data/cara/training_data_train.json \
        --test-input data/cara/training_data_test.json \
        --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
        --output-dir models/cara/knn/ \
        --k 10
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

from block.predictor.cara.estimators.knn_estimator import KNNEstimator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def evaluate(estimator: KNNEstimator, test_data: list) -> dict:
    """Evaluate estimator on test data.

    Returns dict with per-model metrics for both length and quality.
    """
    from collections import defaultdict

    length_errors = defaultdict(list)  # model -> list of absolute errors
    quality_errors = defaultdict(list)
    quality_rankings = []  # list of (true_best, predicted_best)

    for req in test_data:
        prompt = req["prompt"]

        # Get predictions for all models at once
        predictions = estimator.predict_all_models(prompt)

        # Per-model evaluation
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

            # Quality (supports both old and new schema)
            true_q = KNNEstimator._extract_quality(true_vals)
            pred_q = pred["quality_score"]
            quality_errors[model_name].append(abs(true_q - pred_q))

            true_qualities[model_name] = true_q
            pred_qualities[model_name] = pred_q

        # Best model selection accuracy
        if true_qualities and pred_qualities:
            true_best = max(true_qualities, key=true_qualities.get)
            pred_best = max(pred_qualities, key=pred_qualities.get)
            quality_rankings.append((true_best, pred_best))

    # Compute aggregate metrics
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

    # Best model selection accuracy
    if quality_rankings:
        correct = sum(1 for t, p in quality_rankings if t == p)
        results["quality"]["best_model_accuracy"] = correct / len(quality_rankings)
        results["quality"]["n_rankings"] = len(quality_rankings)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train KNN estimator for CARA length/quality prediction"
    )
    parser.add_argument("--input", required=True, help="Training data JSON")
    parser.add_argument("--test-input", default=None, help="Test data JSON (for evaluation)")
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence transformer model for embeddings",
    )
    parser.add_argument("--k", type=int, default=10, help="Number of neighbors")
    parser.add_argument(
        "--output-dir",
        default="models/cara/knn",
        help="Output directory for trained model",
    )
    parser.add_argument("--device", default="cpu", help="Device for embeddings")
    args = parser.parse_args()

    # Load training data (supports both JSON and JSONL)
    with open(args.input) as f:
        if args.input.endswith(".jsonl"):
            train_data = [json.loads(line) for line in f]
        else:
            raw = json.load(f)
            train_data = raw["requests"] if "requests" in raw else raw
    logger.info(f"Training data: {len(train_data)} requests")

    # Build KNN index
    t0 = time.time()
    estimator = KNNEstimator(
        embedding_model_name=args.embedding_model,
        k=args.k,
        device=args.device,
    )
    estimator.build_index(train_data)
    build_time = time.time() - t0
    logger.info(f"Index built in {build_time:.1f}s")

    # Save model
    estimator.save(args.output_dir)

    # Evaluate on test data if provided
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
        print("EVALUATION RESULTS")
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
        print(f"Inference speed: {eval_time/len(test_data)*1000:.1f}ms per query")

        # Save evaluation results
        eval_path = Path(args.output_dir) / "evaluation_results.json"
        with open(eval_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Evaluation results saved to {eval_path}")
    else:
        # Self-evaluate on training data (just to verify it works)
        logger.info("No test data provided, running quick self-evaluation on 10 training samples...")
        results = evaluate(estimator, train_data[:10])
        print("\nSelf-evaluation (training data, 10 samples):")
        for model, metrics in sorted(results["length"].items()):
            print(f"  {model}: MAE={metrics['mae']:.1f}")


if __name__ == "__main__":
    main()
