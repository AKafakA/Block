#!/usr/bin/env python3
"""
Generate paper-ready comparison figures for CARA predictors.

Reads evaluation results from JSON and produces comparison plots.

Usage:
    python -m block.predictor.cara.offline_training.plot_predictor_comparison \
        --results-dir results/cara/predictor_comparison/ \
        --output-dir figures/cara/predictors/
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paper-ready defaults
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.figsize": (8, 5),
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

COLORS = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800", "#607D8B"]


def plot_length_mae_comparison(results: List[dict], output_dir: Path):
    """Bar chart: MAE per predictor per model."""
    predictors = [r["predictor"] for r in results]
    all_models = set()
    for r in results:
        all_models.update(r.get("length", {}).keys())
    models = sorted(all_models)

    if not models:
        return

    x = np.arange(len(models))
    width = 0.8 / len(predictors)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, r in enumerate(results):
        maes = [r.get("length", {}).get(m, {}).get("mae", 0) for m in models]
        ax.bar(x + i * width, maes, width, label=r["predictor"], color=COLORS[i % len(COLORS)])

    ax.set_xlabel("Model")
    ax.set_ylabel("MAE (tokens)")
    ax.set_title("Length Prediction Accuracy")
    ax.set_xticks(x + width * (len(predictors) - 1) / 2)
    short_names = [m.split("/")[-1] if "/" in m else m for m in models]
    ax.set_xticklabels(short_names, rotation=15, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.savefig(output_dir / "length_mae_comparison.pdf")
    fig.savefig(output_dir / "length_mae_comparison.png")
    plt.close(fig)
    logger.info("Saved length_mae_comparison")


def plot_quality_accuracy(results: List[dict], output_dir: Path):
    """Bar chart: best-model selection accuracy per predictor."""
    predictors = []
    accuracies = []
    for r in results:
        acc = r.get("quality", {}).get("_best_model_accuracy") or r.get("aggregate", {}).get("best_model_accuracy", 0)
        predictors.append(r["predictor"])
        accuracies.append(acc)

    if not predictors:
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(predictors, accuracies, color=COLORS[: len(predictors)])
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{acc:.1%}", ha="center", va="bottom", fontsize=11)

    ax.set_ylabel("Best-Model Selection Accuracy")
    ax.set_title("Quality Prediction: Best Model Selection")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)

    fig.savefig(output_dir / "quality_accuracy.pdf")
    fig.savefig(output_dir / "quality_accuracy.png")
    plt.close(fig)
    logger.info("Saved quality_accuracy")


def plot_latency_comparison(latency_results: Dict[str, dict], output_dir: Path):
    """Bar chart: MAE per latency predictor per instance type."""
    if not latency_results:
        return

    predictors = sorted(latency_results.keys())
    all_inst_types = set()
    for pred_results in latency_results.values():
        all_inst_types.update(pred_results.keys())
    inst_types = sorted(all_inst_types)

    if not inst_types:
        return

    x = np.arange(len(inst_types))
    width = 0.8 / len(predictors)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, pred in enumerate(predictors):
        maes = [latency_results[pred].get(it, {}).get("mae", 0) for it in inst_types]
        ax.bar(x + i * width, maes, width, label=pred, color=COLORS[i % len(COLORS)])

    ax.set_xlabel("Instance Type")
    ax.set_ylabel("MAE (seconds)")
    ax.set_title("Latency Prediction Accuracy")
    ax.set_xticks(x + width * (len(predictors) - 1) / 2)
    ax.set_xticklabels(inst_types, rotation=15, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.savefig(output_dir / "latency_mae_comparison.pdf")
    fig.savefig(output_dir / "latency_mae_comparison.png")
    plt.close(fig)
    logger.info("Saved latency_mae_comparison")


def plot_inference_overhead(results: List[dict], output_dir: Path):
    """Bar chart: inference time per predictor."""
    predictors = [r["predictor"] for r in results]
    ms_per_query = [r.get("ms_per_query", 0) for r in results]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(predictors, ms_per_query, color=COLORS[: len(predictors)])
    for bar, ms in zip(bars, ms_per_query):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{ms:.1f}ms", ha="center", va="bottom", fontsize=10)

    ax.set_ylabel("Inference Time (ms/query)")
    ax.set_title("Predictor Overhead")
    ax.grid(axis="y", alpha=0.3)

    fig.savefig(output_dir / "predictor_overhead.pdf")
    fig.savefig(output_dir / "predictor_overhead.png")
    plt.close(fig)
    logger.info("Saved predictor_overhead")


def main():
    parser = argparse.ArgumentParser(
        description="Generate comparison plots for CARA predictors"
    )
    parser.add_argument(
        "--results-dir",
        default="results/cara/predictor_comparison",
        help="Directory containing comparison_results.json",
    )
    parser.add_argument(
        "--latency-results",
        nargs="*",
        default=None,
        help="Latency training_metrics.json files: 'name:path' pairs",
    )
    parser.add_argument(
        "--output-dir",
        default="figures/cara/predictors",
    )
    args = parser.parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load length/quality comparison results
    comparison_file = Path(args.results_dir) / "comparison_results.json"
    if comparison_file.exists():
        with open(comparison_file) as f:
            results = json.load(f)
        logger.info(f"Loaded {len(results)} predictor results")

        plot_length_mae_comparison(results, output_path)
        plot_quality_accuracy(results, output_path)
        plot_inference_overhead(results, output_path)
    else:
        logger.warning(f"No comparison results at {comparison_file}")

    # Load latency comparison results
    if args.latency_results:
        latency_results = {}
        for entry in args.latency_results:
            name, path = entry.split(":", 1)
            with open(path) as f:
                latency_results[name] = json.load(f)
        plot_latency_comparison(latency_results, output_path)

    logger.info(f"All plots saved to {output_path}")


if __name__ == "__main__":
    main()
