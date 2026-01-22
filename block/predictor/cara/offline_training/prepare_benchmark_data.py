#!/usr/bin/env python3
"""
CARA Benchmark Data Preprocessing

Processes single-model benchmark results into training data format compatible
with the multi-model prepare_training_data.py pipeline.

Use case: Build training pipeline with available single-model data (e.g., Qwen2.5-3B),
then easily transition to multi-model data when more nodes become available.

Usage:
    python -m block.predictor.cara.offline_training.prepare_benchmark_data \
        --input cara_result/cara-*.json \
        --output data/cara/processed/benchmark_training.json \
        --model-name "Qwen/Qwen2.5-3B"

Output format matches prepare_training_data.py for compatibility:
{
    "dataset_name": "...",
    "scoring_method": "llm_judge" | "compression",
    "num_requests": N,
    "models": ["Qwen/Qwen2.5-3B"],
    "requests": [
        {
            "request_id": "...",
            "prompt": "...",
            "input_len": 128,
            "models": {
                "Qwen/Qwen2.5-3B": {
                    "output_length": 256,
                    "quality_score": 0.85,
                    "compression_ratio": 0.45,
                    "is_truncated": false,
                    "ttft": 0.5,
                    "server_latency": 2.3,
                    "instance_id": "...",
                    "host": "..."
                }
            }
        }
    ]
}
"""

import argparse
import json
import logging
import zlib
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ProcessingStats:
    """Statistics for data processing."""
    total_requests: int = 0
    filtered_empty: int = 0
    filtered_too_short: int = 0
    filtered_truncated: int = 0
    filtered_error: int = 0
    filtered_high_repetition: int = 0
    valid_requests: int = 0

    def log(self):
        """Log processing statistics."""
        logger.info("=" * 60)
        logger.info("PROCESSING STATISTICS")
        logger.info("=" * 60)
        logger.info(f"Total requests: {self.total_requests}")
        logger.info(f"Valid requests: {self.valid_requests}")
        logger.info(f"Filtered (empty): {self.filtered_empty}")
        logger.info(f"Filtered (too short): {self.filtered_too_short}")
        logger.info(f"Filtered (truncated): {self.filtered_truncated}")
        logger.info(f"Filtered (error): {self.filtered_error}")
        logger.info(f"Filtered (high repetition): {self.filtered_high_repetition}")
        logger.info("=" * 60)


def compute_compression_ratio(text: str) -> float:
    """Compute compression ratio using zlib.

    Repetitive text has low compression ratio (compresses well).

    Args:
        text: Text to analyze

    Returns:
        Compression ratio in [0, 1] where:
        - Higher ratio (~0.5-1.0) = diverse/random text
        - Lower ratio (<0.2) = highly repetitive text
    """
    if not text:
        return 1.0

    text_bytes = text.encode('utf-8')
    original_size = len(text_bytes)

    if original_size == 0:
        return 1.0

    compressed = zlib.compress(text_bytes)
    compressed_size = len(compressed)

    return compressed_size / original_size


def load_benchmark_results(input_path: Path) -> Dict:
    """Load benchmark results from JSON file.

    Args:
        input_path: Path to benchmark JSON file

    Returns:
        Parsed JSON data
    """
    logger.info(f"Loading benchmark results from: {input_path}")

    with open(input_path, 'r') as f:
        data = json.load(f)

    response_details = data.get("response_details", [])
    logger.info(f"Found {len(response_details)} requests in benchmark results")

    return data


def process_benchmark_data(
    data: Dict,
    model_name: str,
    min_output_tokens: int = 3,
    max_output_tokens: int = 1024,
    filter_truncated: bool = False,
    filter_high_repetition: bool = False,
    min_compression_ratio: float = 0.2,
) -> tuple[List[Dict], ProcessingStats]:
    """Process benchmark data into training format.

    Args:
        data: Raw benchmark JSON data
        model_name: Model name (e.g., "Qwen/Qwen2.5-3B")
        min_output_tokens: Minimum output length to keep
        max_output_tokens: Maximum output length (for truncation detection)
        filter_truncated: If True, filter out truncated responses
        filter_high_repetition: If True, filter out high repetition responses
        min_compression_ratio: Threshold for high repetition (only if filter enabled)

    Returns:
        Tuple of (processed_requests, stats)
    """
    response_details = data.get("response_details", [])
    stats = ProcessingStats(total_requests=len(response_details))

    processed_requests = []

    for detail in response_details:
        request_id = detail.get("request_id", "unknown")
        prompt = detail.get("prompt", "")
        input_len = detail.get("input_len", 0)
        output_len = detail.get("output_len", 0)
        response = detail.get("response", "")
        error = detail.get("error", "")

        # Filter: errors
        if error:
            stats.filtered_error += 1
            continue

        # Filter: empty responses
        if output_len <= 1 or not response.strip():
            stats.filtered_empty += 1
            continue

        # Filter: too short
        if output_len < min_output_tokens:
            stats.filtered_too_short += 1
            continue

        # Detect truncation
        is_truncated = output_len >= max_output_tokens

        # Filter: truncated (optional)
        if filter_truncated and is_truncated:
            stats.filtered_truncated += 1
            continue

        # Compute compression ratio
        compression_ratio = compute_compression_ratio(response)

        # Filter: high repetition (optional, disabled by default)
        if filter_high_repetition and compression_ratio < min_compression_ratio:
            stats.filtered_high_repetition += 1
            continue

        # Build processed request in compatible format
        processed_request = {
            "request_id": request_id,
            "prompt": prompt,
            "input_len": input_len,
            "models": {
                model_name: {
                    "output_length": output_len,
                    "quality_score": None,  # Will be filled by scorer if enabled
                    "compression_ratio": round(compression_ratio, 4),
                    "is_truncated": is_truncated,
                    "ttft": detail.get("ttft", 0.0),
                    "server_latency": detail.get("e2el", 0.0),
                    "instance_id": detail.get("instance_id", "unknown"),
                    "host": detail.get("host", "unknown"),
                    # Store response for potential LLM judge scoring
                    "_response": response,
                }
            }
        }

        processed_requests.append(processed_request)
        stats.valid_requests += 1

    return processed_requests, stats


def compute_quality_scores_compression(
    requests: List[Dict],
    model_name: str,
) -> None:
    """Compute quality scores based on compression ratio.

    Simple heuristic: higher compression ratio = better quality.
    Normalized to [0, 1] range.

    Args:
        requests: List of processed requests (modified in place)
        model_name: Model name key
    """
    logger.info("Computing quality scores from compression ratio...")

    for req in requests:
        model_data = req["models"][model_name]
        compression_ratio = model_data["compression_ratio"]

        # Normalize: typical range is 0.2 - 0.6 for text
        # Map to [0, 1] where 0.2 -> 0.0 and 0.6+ -> 1.0
        quality_score = max(0.0, min(1.0, (compression_ratio - 0.2) / 0.4))
        model_data["quality_score"] = round(quality_score, 4)


def compute_quality_scores_llm_judge(
    requests: List[Dict],
    model_name: str,
    judge_model: str = "Qwen/Qwen2.5-3B",
    device: str = "cuda",
    batch_size: int = 1,
) -> None:
    """Compute quality scores using LLM-as-judge.

    Args:
        requests: List of processed requests (modified in place)
        model_name: Model name key
        judge_model: HuggingFace model for judging
        device: Device for judge model
        batch_size: Batch size for inference
    """
    from block.predictor.cara.offline_training.llm_judge_scorer import LLMJudgeScorer

    logger.info(f"Computing quality scores using LLM judge: {judge_model}")

    scorer = LLMJudgeScorer(
        judge_model=judge_model,
        batch_size=batch_size,
        device=device,
    )

    for idx, req in enumerate(requests):
        if (idx + 1) % 100 == 0:
            logger.info(f"Scored {idx + 1}/{len(requests)} requests")

        model_data = req["models"][model_name]
        prompt = req["prompt"]
        response = model_data.get("_response", "")

        if not response:
            model_data["quality_score"] = 0.5
            continue

        # Score using LLM judge
        scores = scorer.score(prompt, [(model_name, response)])
        model_data["quality_score"] = round(scores.get(model_name, 0.5), 4)

    logger.info(f"Completed scoring {len(requests)} requests")


def compute_quality_scores_multi_judge(
    requests: List[Dict],
    model_name: str,
    judge_models: List[str],
    device: str = "cuda",
    batch_size: int = 8,
) -> tuple[Dict[str, List[Optional[float]]], set]:
    """Compute quality scores using multiple LLM judges.

    Args:
        requests: List of processed requests (modified in place)
        model_name: Model name key
        judge_models: List of HuggingFace model names for judging
        device: Device for judge models
        batch_size: Batch size for inference

    Returns:
        Tuple of:
        - Dict mapping judge_model_name -> list of scores (one per request, None for failures)
        - Set of request indices that had scoring failures (to be filtered out)
    """
    from block.predictor.cara.offline_training.llm_judge_scorer import LLMJudgeScorer

    all_scores = {}
    failed_indices = set()

    for judge_model in judge_models:
        logger.info(f"\n{'='*40}")
        logger.info(f"Running judge: {judge_model}")
        logger.info(f"{'='*40}")

        scorer = LLMJudgeScorer(
            judge_model=judge_model,
            batch_size=batch_size,
            device=device,
        )

        judge_scores = []
        judge_failures = 0

        for idx, req in enumerate(requests):
            if (idx + 1) % 100 == 0:
                logger.info(f"  Scored {idx + 1}/{len(requests)} requests")

            model_data = req["models"][model_name]
            prompt = req["prompt"]
            response = model_data.get("_response", "")

            if not response:
                score = None
            else:
                scores = scorer.score(prompt, [(model_name, response)])
                score = scores.get(model_name)  # None if parsing failed

            if score is None:
                failed_indices.add(idx)
                judge_failures += 1

            judge_scores.append(score)

            # Store score with judge name as key
            judge_key = f"quality_score_{judge_model.replace('/', '_')}"
            model_data[judge_key] = score

        all_scores[judge_model] = judge_scores
        logger.info(f"Completed {judge_model}: {len(judge_scores)} scores, {judge_failures} failures")

        # Free memory
        del scorer

    # Set primary quality_score to first judge's scores
    if judge_models:
        primary_judge = judge_models[0]
        for idx, req in enumerate(requests):
            model_data = req["models"][model_name]
            model_data["quality_score"] = all_scores[primary_judge][idx]

    logger.info(f"\nTotal requests with scoring failures: {len(failed_indices)}")

    return all_scores, failed_indices


def analyze_judge_scores(
    requests: List[Dict],
    judge_models: List[str],
) -> Dict:
    """Analyze correlation and distribution of scores from multiple judges.

    For multi-model case: computes per-request Spearman correlation of model rankings
    between judge pairs, then aggregates across requests.

    Args:
        requests: List of processed requests with scores from all judges
        judge_models: List of judge model names

    Returns:
        Dict containing analysis results
    """
    from scipy.stats import spearmanr

    logger.info("\n" + "=" * 60)
    logger.info("JUDGE COMPARISON ANALYSIS")
    logger.info("=" * 60)

    # Infer LLM models from the data
    llm_models = list(requests[0]["models"].keys()) if requests else []

    analysis = {
        "per_judge_stats": {},
        "pairwise_correlations": {},
        "score_differences": {},
        "per_request_ranking_correlation": {},
        "llm_models": llm_models,
    }

    num_llm_models = len(llm_models)
    num_requests = len(requests)

    logger.info(f"LLM models found: {llm_models}")

    # Collect all scores per judge
    # Structure: judge -> list of scores (one per request per model)
    all_scores_flat = {judge: [] for judge in judge_models}

    # Structure for per-request analysis: judge -> request_idx -> {model: score}
    scores_by_request = {judge: [] for judge in judge_models}

    for req in requests:
        for judge in judge_models:
            judge_key = f"quality_score_{judge.replace('/', '_')}"
            request_scores = {}
            for model in llm_models:
                if model in req["models"]:
                    score = req["models"][model].get(judge_key)
                    all_scores_flat[judge].append(score)
                    request_scores[model] = score
            scores_by_request[judge].append(request_scores)

    # Per-judge statistics
    logger.info("\n--- Per-Judge Statistics ---")
    for judge in judge_models:
        scores = np.array(all_scores_flat[judge])
        if len(scores) == 0:
            continue
        stats = {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
            "median": float(np.median(scores)),
            "q50": float(np.percentile(scores, 50)),
            "q95": float(np.percentile(scores, 95)),
        }
        analysis["per_judge_stats"][judge] = stats
        logger.info(f"\n{judge}:")
        logger.info(f"  Mean: {stats['mean']:.4f} ± {stats['std']:.4f}")
        logger.info(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
        logger.info(f"  Median: {stats['median']:.4f}")
        logger.info(f"  IQR: [{stats['q25']:.4f}, {stats['q75']:.4f}]")

    # Global flat correlation (Pearson) - comparing all scores across (request, model) pairs
    if len(judge_models) > 1:
        logger.info("\n--- Global Flat Correlation (across all request-model pairs) ---")
        for i, judge1 in enumerate(judge_models):
            for judge2 in judge_models[i+1:]:
                scores1 = np.array(all_scores_flat[judge1])
                scores2 = np.array(all_scores_flat[judge2])

                if len(scores1) != len(scores2):
                    raise ValueError(
                        f"Score count mismatch between judges: {judge1} has {len(scores1)}, "
                        f"{judge2} has {len(scores2)}. This indicates a bug in scoring."
                    )
                if len(scores1) == 0:
                    raise ValueError(
                        f"No scores found for judges {judge1} and {judge2}. "
                        "All requests may have been filtered out."
                    )

                # Pearson correlation
                pearson_corr = np.corrcoef(scores1, scores2)[0, 1]

                # Also compute Spearman on flat scores for comparison
                flat_spearman, flat_spearman_p = spearmanr(scores1, scores2)

                key = f"{judge1} vs {judge2}"
                analysis["pairwise_correlations"][key] = {
                    "pearson": float(pearson_corr),
                    "spearman": float(flat_spearman),
                    "spearman_pvalue": float(flat_spearman_p),
                    "num_samples": len(scores1),
                }

                logger.info(f"\n{key}:")
                logger.info(f"  Pearson r:  {pearson_corr:.4f}")
                logger.info(f"  Spearman ρ: {flat_spearman:.4f} (p={flat_spearman_p:.2e})")
                logger.info(f"  Num samples: {len(scores1)}")

    # Per-request model ranking correlation (only meaningful with multiple LLM models)
    if num_llm_models > 1 and len(judge_models) > 1:
        logger.info("\n--- Per-Request Model Ranking Correlation ---")
        logger.info(f"(Comparing how judges rank {num_llm_models} LLM models within each request)")

        for i, judge1 in enumerate(judge_models):
            for judge2 in judge_models[i+1:]:
                per_request_spearman = []

                for req_idx in range(num_requests):
                    scores1 = scores_by_request[judge1][req_idx]
                    scores2 = scores_by_request[judge2][req_idx]

                    # Get scores in same model order - all models should be present
                    common_models = [m for m in llm_models if m in scores1 and m in scores2]
                    if len(common_models) != len(llm_models):
                        raise ValueError(
                            f"Request {req_idx}: Expected {len(llm_models)} models but found "
                            f"{len(common_models)}. Failed requests should have been filtered out."
                        )

                    s1 = [scores1[m] for m in common_models]
                    s2 = [scores2[m] for m in common_models]

                    # Spearman correlation for this request
                    if len(set(s1)) > 1 and len(set(s2)) > 1:  # Need variance
                        corr, _ = spearmanr(s1, s2)
                        if not np.isnan(corr):
                            per_request_spearman.append(corr)

                if per_request_spearman:
                    mean_corr = float(np.mean(per_request_spearman))
                    std_corr = float(np.std(per_request_spearman))
                    median_corr = float(np.median(per_request_spearman))

                    key = f"{judge1} vs {judge2}"
                    analysis["per_request_ranking_correlation"][key] = {
                        "mean_spearman": mean_corr,
                        "std_spearman": std_corr,
                        "median_spearman": median_corr,
                        "num_valid_requests": len(per_request_spearman),
                    }

                    logger.info(f"\n{key}:")
                    logger.info(f"  Mean Spearman ρ: {mean_corr:.4f} ± {std_corr:.4f}")
                    logger.info(f"  Median Spearman ρ: {median_corr:.4f}")
                    logger.info(f"  Valid requests: {len(per_request_spearman)}/{num_requests}")
    elif num_llm_models <= 1:
        logger.info("\n--- Per-Request Model Ranking Correlation ---")
        logger.info("  Skipped: Only 1 LLM model (need 2+ models to compute ranking correlation)")

    # Score differences distribution (per judge pair, aggregated across all scores)
    if len(judge_models) > 1:
        logger.info("\n--- Score Difference Distribution ---")
        for i, judge1 in enumerate(judge_models):
            for judge2 in judge_models[i+1:]:
                scores1 = np.array(all_scores_flat[judge1])
                scores2 = np.array(all_scores_flat[judge2])

                if len(scores1) != len(scores2):
                    raise ValueError(
                        f"Score count mismatch between judges: {judge1} has {len(scores1)}, "
                        f"{judge2} has {len(scores2)}. This indicates a bug in scoring."
                    )
                if len(scores1) == 0:
                    raise ValueError(
                        f"No scores found for judges {judge1} and {judge2}. "
                        "All requests may have been filtered out."
                    )

                diffs = scores1 - scores2

                diff_stats = {
                    "mean": float(np.mean(diffs)),
                    "std": float(np.std(diffs)),
                    "abs_mean": float(np.mean(np.abs(diffs))),
                    "max_diff": float(np.max(np.abs(diffs))),
                }

                key = f"{judge1} - {judge2}"
                analysis["score_differences"][key] = diff_stats

                logger.info(f"\n{key}:")
                logger.info(f"  Mean diff: {diff_stats['mean']:.4f} ± {diff_stats['std']:.4f}")
                logger.info(f"  Mean |diff|: {diff_stats['abs_mean']:.4f}")
                logger.info(f"  Max |diff|: {diff_stats['max_diff']:.4f}")

    logger.info("\n" + "=" * 60)

    return analysis


def find_divergent_samples(
    requests: List[Dict],
    judge_models: List[str],
    threshold: float = 0.3,
) -> List[Dict]:
    """Find samples where judges disagree significantly.

    Args:
        requests: List of processed requests (scores stored in request data)
        judge_models: List of judge model names
        threshold: Score difference threshold for divergence

    Returns:
        List of divergent samples with scores
    """
    divergent = []

    # Infer LLM models from data
    llm_models = list(requests[0]["models"].keys()) if requests else []

    for req in requests:
        # For each LLM model in the request, check judge agreement
        for model_name in llm_models:
            if model_name not in req["models"]:
                continue

            model_data = req["models"][model_name]

            # Extract scores from each judge
            scores = []
            score_by_judge = {}
            for judge in judge_models:
                judge_key = f"quality_score_{judge.replace('/', '_')}"
                if judge_key in model_data:
                    score = model_data[judge_key]
                    scores.append(score)
                    score_by_judge[judge] = score

            if len(scores) < 2:
                continue

            max_diff = max(scores) - min(scores)

            if max_diff >= threshold:
                sample = {
                    "request_id": req["request_id"],
                    "llm_model": model_name,
                    "prompt": req["prompt"],
                    "response": model_data.get("_response", ""),
                    "scores": score_by_judge,
                    "max_difference": max_diff,
                    "mean_score": float(np.mean(scores)),
                    "score_std": float(np.std(scores)),
                }
                divergent.append(sample)

    # Sort by max difference descending
    divergent.sort(key=lambda x: x["max_difference"], reverse=True)

    logger.info(f"\nFound {len(divergent)} divergent samples (threshold={threshold})")

    return divergent


def save_divergent_samples(
    divergent: List[Dict],
    output_path: Path,
    analysis: Dict,
) -> None:
    """Save divergent samples to JSON file.

    Args:
        divergent: List of divergent samples
        output_path: Output file path
        analysis: Analysis results to include
    """
    output_data = {
        "num_divergent": len(divergent),
        "analysis": analysis,
        "samples": divergent,
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved {len(divergent)} divergent samples to: {output_path}")


def save_training_data(
    requests: List[Dict],
    output_path: Path,
    model_name: str,
    dataset_name: str,
    scoring_method: str,
    include_response: bool = False,
) -> None:
    """Save processed data in training format.

    Args:
        requests: List of processed requests
        output_path: Output file path
        model_name: Model name
        dataset_name: Dataset name for metadata
        scoring_method: Scoring method used
        include_response: If True, include full response text
    """
    # Clean up internal fields if not including response
    for req in requests:
        for model_data in req["models"].values():
            if not include_response and "_response" in model_data:
                del model_data["_response"]

    output_data = {
        "dataset_name": dataset_name,
        "scoring_method": scoring_method,
        "num_requests": len(requests),
        "models": [model_name],
        "requests": requests,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    file_size_mb = output_path.stat().st_size / 1024 / 1024
    logger.info(f"Saved training data to: {output_path}")
    logger.info(f"  Requests: {len(requests)}")
    logger.info(f"  File size: {file_size_mb:.2f} MB")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Preprocess CARA benchmark results for model estimation training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Input/output
    parser.add_argument(
        "-i", "--input",
        type=Path,
        required=True,
        help="Input benchmark JSON file"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output JSON file (default: auto-generated)"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="benchmark",
        help="Dataset name for output metadata"
    )

    # Model
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen2.5-3B",
        help="Model name in benchmark results"
    )

    # Filtering
    parser.add_argument(
        "--min-output-tokens",
        type=int,
        default=3,
        help="Minimum output length to keep"
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=1024,
        help="Maximum output length (for truncation detection)"
    )
    parser.add_argument(
        "--filter-truncated",
        action="store_true",
        help="Filter out truncated responses (hitting max_tokens)"
    )
    parser.add_argument(
        "--filter-high-repetition",
        action="store_true",
        help="Filter out high repetition responses (disabled by default)"
    )
    parser.add_argument(
        "--min-compression-ratio",
        type=float,
        default=0.2,
        help="Compression ratio threshold for repetition (only if --filter-high-repetition)"
    )

    # Quality scoring
    parser.add_argument(
        "--scoring-method",
        type=str,
        choices=["llm_judge", "none"],
        default="llm_judge",
        help="Quality scoring method"
    )
    parser.add_argument(
        "--judge-models",
        type=str,
        nargs="+",
        default=["Qwen/Qwen2.5-0.5B"],
        help="Judge model(s) for llm_judge scoring. Pass multiple with --compare-judges for comparison."
    )
    parser.add_argument(
        "--compare-judges",
        action="store_true",
        help="Enable judge comparison analysis (requires multiple --judge-models)"
    )
    parser.add_argument(
        "--divergence-threshold",
        type=float,
        default=0.3,
        help="Score difference threshold to flag divergent samples when comparing judges (0-1 scale)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for LLM judge (cuda, cpu)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for LLM judge inference (higher = faster but more memory)"
    )

    # Output options
    parser.add_argument(
        "--include-response",
        action="store_true",
        help="Include full response text in output (increases file size)"
    )

    # Logging
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)

    # Validate input
    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        return 1

    # Determine output path
    if args.output is None:
        output_filename = f"{args.dataset_name}_{args.scoring_method}_training.json"
        args.output = args.input.parent / output_filename

    # Comparison mode requires both flag and multiple judges
    compare_judges = args.compare_judges and len(args.judge_models) > 1

    if args.compare_judges and len(args.judge_models) <= 1:
        logger.warning("--compare-judges requires multiple --judge-models, disabling comparison")

    logger.info("=" * 60)
    logger.info("CARA BENCHMARK DATA PREPROCESSING")
    logger.info("=" * 60)
    logger.info(f"Input:          {args.input}")
    logger.info(f"Output:         {args.output}")
    logger.info(f"Model:          {args.model_name}")
    logger.info(f"Scoring:        {args.scoring_method}")
    logger.info(f"Judge model(s): {args.judge_models}")
    if compare_judges:
        logger.info(f"Compare mode:   ENABLED ({len(args.judge_models)} judges)")
        logger.info(f"Divergence threshold: {args.divergence_threshold}")
    logger.info(f"Min tokens:     {args.min_output_tokens}")
    logger.info(f"Max tokens:     {args.max_output_tokens}")
    logger.info(f"Filter truncated: {args.filter_truncated}")
    logger.info(f"Filter repetition: {args.filter_high_repetition}")
    logger.info("=" * 60)

    try:
        # Load data
        logger.info("\n[1/3] Loading benchmark data...")
        data = load_benchmark_results(args.input)

        # Process data
        logger.info("\n[2/3] Processing and filtering...")
        requests, stats = process_benchmark_data(
            data=data,
            model_name=args.model_name,
            min_output_tokens=args.min_output_tokens,
            max_output_tokens=args.max_output_tokens,
            filter_truncated=args.filter_truncated,
            filter_high_repetition=args.filter_high_repetition,
            min_compression_ratio=args.min_compression_ratio,
        )

        stats.log()

        if not requests:
            logger.error("No valid requests after filtering!")
            return 1

        # Compute quality scores
        logger.info("\n[3/3] Computing quality scores...")
        all_scores = None
        failed_indices = set()

        if args.scoring_method == "compression":
            compute_quality_scores_compression(requests, args.model_name)
        elif args.scoring_method == "llm_judge":
            all_scores, failed_indices = compute_quality_scores_multi_judge(
                requests,
                args.model_name,
                judge_models=args.judge_models,
                device=args.device,
                batch_size=args.batch_size,
            )
        else:
            logger.info("Skipping quality scoring (method=none)")

        # Filter out requests with failed scores
        if failed_indices:
            original_count = len(requests)
            requests = [req for idx, req in enumerate(requests) if idx not in failed_indices]
            logger.info(f"Filtered out {len(failed_indices)} requests with scoring failures")
            logger.info(f"Remaining requests: {len(requests)}/{original_count}")

            if not requests:
                logger.error("No valid requests remaining after filtering scoring failures!")
                return 1

        # Run comparison analysis if enabled
        if compare_judges and all_scores:
            analysis = analyze_judge_scores(requests, args.judge_models)

            # Find and save divergent samples
            divergent = find_divergent_samples(
                requests,
                args.judge_models,
                threshold=args.divergence_threshold,
            )

            if divergent:
                divergent_path = args.output.parent / f"{args.output.stem}_divergent.json"
                save_divergent_samples(divergent, divergent_path, analysis)

        # Save output
        save_training_data(
            requests=requests,
            output_path=args.output,
            model_name=args.model_name,
            dataset_name=args.dataset_name,
            scoring_method=args.scoring_method,
            include_response=args.include_response,
        )

        logger.info("\n" + "=" * 60)
        logger.info("PREPROCESSING COMPLETE")
        logger.info("=" * 60)
        return 0

    except KeyboardInterrupt:
        logger.warning("\nInterrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())