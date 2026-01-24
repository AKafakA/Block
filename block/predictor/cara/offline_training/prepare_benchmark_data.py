#!/usr/bin/env python3
"""
CARA Benchmark Data Preprocessing

Unified preprocessing script for CARA benchmark results. Supports both:
- Single-model format: Direct response in response_details
- Multi-model format: Multiple models via broadcast_results field

Features:
- Auto-detects data format (single-model vs multi-model with broadcast_results)
- Auto-detects all models from data (scans all requests for robustness)
- Handles incomplete requests (exports to JSONL for re-running benchmark)
- Multiple quality scoring methods: llm_judge, similarity, compression
- Judge comparison analysis with correlation metrics
- Divergent sample detection and export

Usage (single-model):
    python -m block.predictor.cara.offline_training.prepare_benchmark_data \\
        --input cara_result/cara-benchmark.json \\
        --scoring-method llm_judge \\
        --judge-models Qwen/Qwen2.5-0.5B

Usage (multi-model with similarity scoring):
    python -m block.predictor.cara.offline_training.prepare_benchmark_data \\
        --input data/cara/cara-best-route-training.json \\
        --scoring-method similarity \\
        --reference-model "Qwen/Qwen2.5-72B"

Usage (multi-judge comparison):
    python -m block.predictor.cara.offline_training.prepare_benchmark_data \\
        --input cara_result/cara-benchmark.json \\
        --judge-models Qwen/Qwen2.5-0.5B Qwen/Qwen2.5-3B \\
        --compare-judges \\
        --divergence-threshold 0.3

Output files:
- {output}.json: Training data with quality scores
- {output}_incomplete.jsonl: Incomplete requests for re-running (benchmark format)
- {output}_divergent.json: Samples where judges disagree (if --compare-judges)

Output format:
{
    "dataset_name": "...",
    "scoring_method": "llm_judge" | "similarity" | "compression",
    "num_requests": N,
    "models": ["Qwen/Qwen2.5-3B", "Qwen/Qwen2.5-72B", ...],
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
                },
                ...
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
    total_responses: int = 0  # Total model responses across all requests
    filtered_empty: int = 0
    filtered_too_short: int = 0
    filtered_truncated: int = 0
    filtered_error: int = 0
    filtered_high_repetition: int = 0
    valid_responses: int = 0
    valid_requests: int = 0
    incomplete_requests: int = 0  # Requests missing some models

    def log(self):
        """Log processing statistics."""
        logger.info("=" * 60)
        logger.info("PROCESSING STATISTICS")
        logger.info("=" * 60)
        logger.info(f"Total requests: {self.total_requests}")
        logger.info(f"Total model responses: {self.total_responses}")
        logger.info(f"Valid responses: {self.valid_responses}")
        logger.info(f"Valid requests (with all models): {self.valid_requests}")
        logger.info(f"Incomplete requests: {self.incomplete_requests}")
        logger.info(f"Filtered responses (empty): {self.filtered_empty}")
        logger.info(f"Filtered responses (too short): {self.filtered_too_short}")
        logger.info(f"Filtered responses (truncated): {self.filtered_truncated}")
        logger.info(f"Filtered responses (error): {self.filtered_error}")
        logger.info(f"Filtered responses (high repetition): {self.filtered_high_repetition}")
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


def detect_data_format(data: Dict) -> str:
    """Detect if data is single-model or multi-model format.

    Args:
        data: Raw benchmark JSON data

    Returns:
        "multi_model" if broadcast_results present with 2+ models, else "single_model"
    """
    response_details = data.get("response_details", [])

    for detail in response_details:
        broadcast_results = detail.get("broadcast_results", [])
        if broadcast_results and len(broadcast_results) >= 2:
            return "multi_model"

    return "single_model"


def collect_all_models(data: Dict) -> tuple[set, Dict[str, int], Dict[str, set]]:
    """Scan all requests to collect all seen models and their availability.

    Args:
        data: Raw benchmark JSON data

    Returns:
        Tuple of:
        - Set of all model names seen
        - Dict mapping model_name -> count of requests with this model
        - Dict mapping request_id -> set of models available for this request
    """
    response_details = data.get("response_details", [])
    all_models = set()
    model_counts = defaultdict(int)
    request_models = {}

    for detail in response_details:
        request_id = detail.get("request_id", "unknown")
        models_in_request = set()

        # Check broadcast_results first (multi-model format)
        broadcast_results = detail.get("broadcast_results", [])
        if broadcast_results:
            for br in broadcast_results:
                model_name = br.get("model")
                if model_name:
                    all_models.add(model_name)
                    models_in_request.add(model_name)
                    model_counts[model_name] += 1
        else:
            # Single-model format: use top-level model field
            model_name = detail.get("model")
            if model_name:
                all_models.add(model_name)
                models_in_request.add(model_name)
                model_counts[model_name] += 1

        request_models[request_id] = models_in_request

    return all_models, dict(model_counts), request_models


def log_model_statistics(
    all_models: set,
    model_counts: Dict[str, int],
    total_requests: int,
) -> None:
    """Log statistics about model availability.

    Args:
        all_models: Set of all model names
        model_counts: Dict mapping model_name -> count
        total_requests: Total number of requests
    """
    logger.info("=" * 60)
    logger.info("MODEL AVAILABILITY STATISTICS")
    logger.info("=" * 60)
    logger.info(f"Total unique models found: {len(all_models)}")
    logger.info(f"Total requests: {total_requests}")
    logger.info("")
    for model in sorted(all_models):
        count = model_counts.get(model, 0)
        pct = (count / total_requests * 100) if total_requests > 0 else 0
        logger.info(f"  {model}: {count}/{total_requests} ({pct:.1f}%)")
    logger.info("=" * 60)


def process_model_response(
    model_name: str,
    response_text: str,
    output_len: int,
    error: str,
    ttft: float,
    server_latency: float,
    instance_id: str,
    host: str,
    min_output_tokens: int,
    max_output_tokens: int,
    filter_truncated: bool,
    filter_high_repetition: bool,
    min_compression_ratio: float,
    stats: ProcessingStats,
) -> Optional[Dict]:
    """Process a single model's response with filtering.

    Args:
        model_name: Name of the model
        response_text: Generated response text
        output_len: Number of output tokens
        error: Error message if any
        ttft: Time to first token
        server_latency: End-to-end latency
        instance_id: Instance identifier
        host: Host name
        min_output_tokens: Minimum output length
        max_output_tokens: Maximum output length
        filter_truncated: Whether to filter truncated responses
        filter_high_repetition: Whether to filter repetitive responses
        min_compression_ratio: Minimum compression ratio threshold
        stats: ProcessingStats to update

    Returns:
        Processed model data dict, or None if filtered out
    """
    stats.total_responses += 1

    # Filter: errors
    if error:
        stats.filtered_error += 1
        return None

    # Filter: empty responses
    if output_len <= 1 or not response_text.strip():
        stats.filtered_empty += 1
        return None

    # Filter: too short
    if output_len < min_output_tokens:
        stats.filtered_too_short += 1
        return None

    # Detect truncation
    is_truncated = output_len >= max_output_tokens

    # Filter: truncated (optional)
    if filter_truncated and is_truncated:
        stats.filtered_truncated += 1
        return None

    # Compute compression ratio
    compression_ratio = compute_compression_ratio(response_text)

    # Filter: high repetition (optional)
    if filter_high_repetition and compression_ratio < min_compression_ratio:
        stats.filtered_high_repetition += 1
        return None

    stats.valid_responses += 1

    return {
        "output_length": output_len,
        "quality_score": None,  # Will be filled by scorer
        "compression_ratio": round(compression_ratio, 4),
        "is_truncated": is_truncated,
        "ttft": ttft,
        "server_latency": server_latency,
        "instance_id": instance_id,
        "host": host,
        "_response": response_text,
    }


def process_benchmark_data(
    data: Dict,
    expected_models: set,
    min_output_tokens: int = 3,
    max_output_tokens: int = 1024,
    filter_truncated: bool = False,
    filter_high_repetition: bool = False,
    min_compression_ratio: float = 0.2,
    require_all_models: bool = True,
) -> tuple[List[Dict], List[Dict], ProcessingStats]:
    """Process benchmark data into training format.

    Supports both single-model and multi-model (broadcast_results) formats.

    Args:
        data: Raw benchmark JSON data
        expected_models: Set of expected model names (from collect_all_models)
        min_output_tokens: Minimum output length to keep
        max_output_tokens: Maximum output length (for truncation detection)
        filter_truncated: If True, filter out truncated responses
        filter_high_repetition: If True, filter out high repetition responses
        min_compression_ratio: Threshold for high repetition (only if filter enabled)
        require_all_models: If True, filter out requests missing any expected model

    Returns:
        Tuple of (complete_requests, incomplete_requests, stats)
        - complete_requests: Requests with all expected models
        - incomplete_requests: Requests missing some models (for re-running)
    """
    response_details = data.get("response_details", [])
    stats = ProcessingStats(total_requests=len(response_details))
    data_format = detect_data_format(data)

    complete_requests = []
    incomplete_requests = []

    for detail in response_details:
        request_id = detail.get("request_id", "unknown")
        prompt = detail.get("prompt", "")
        input_len = detail.get("input_len", 0)

        models_data = {}

        if data_format == "multi_model":
            # Multi-model format: extract from broadcast_results
            broadcast_results = detail.get("broadcast_results", [])
            for br in broadcast_results:
                model_name = br.get("model")
                if not model_name:
                    continue

                model_data = process_model_response(
                    model_name=model_name,
                    response_text=br.get("generated_text", ""),
                    output_len=br.get("output_tokens", 0),
                    error=br.get("error", ""),
                    ttft=br.get("ttft", 0.0),
                    server_latency=br.get("server_latency", 0.0),
                    instance_id=br.get("instance_id", "unknown"),
                    host=br.get("host", "unknown"),
                    min_output_tokens=min_output_tokens,
                    max_output_tokens=max_output_tokens,
                    filter_truncated=filter_truncated,
                    filter_high_repetition=filter_high_repetition,
                    min_compression_ratio=min_compression_ratio,
                    stats=stats,
                )
                if model_data:
                    models_data[model_name] = model_data
        else:
            # Single-model format: extract from top-level fields
            model_name = detail.get("model")
            if not model_name and expected_models:
                # Fallback: use first expected model if not specified
                model_name = next(iter(expected_models))

            if model_name:
                model_data = process_model_response(
                    model_name=model_name,
                    response_text=detail.get("response", ""),
                    output_len=detail.get("output_len", 0),
                    error=detail.get("error", ""),
                    ttft=detail.get("ttft", 0.0),
                    server_latency=detail.get("e2el", 0.0),
                    instance_id=detail.get("instance_id", "unknown"),
                    host=detail.get("host", "unknown"),
                    min_output_tokens=min_output_tokens,
                    max_output_tokens=max_output_tokens,
                    filter_truncated=filter_truncated,
                    filter_high_repetition=filter_high_repetition,
                    min_compression_ratio=min_compression_ratio,
                    stats=stats,
                )
                if model_data:
                    models_data[model_name] = model_data

        # Skip if no valid model responses
        if not models_data:
            continue

        # Build processed request
        processed_request = {
            "request_id": request_id,
            "prompt": prompt,
            "input_len": input_len,
            "models": models_data,
        }

        # Check completeness
        available_models = set(models_data.keys())
        missing_models = expected_models - available_models

        if missing_models and require_all_models:
            stats.incomplete_requests += 1
            # Store for potential re-running
            processed_request["_missing_models"] = list(missing_models)
            incomplete_requests.append(processed_request)
        else:
            stats.valid_requests += 1
            complete_requests.append(processed_request)

    return complete_requests, incomplete_requests, stats


def compute_quality_scores_similarity(
    requests: List[Dict],
    reference_model: str,
    device: str = "cpu",
) -> None:
    """Compute quality scores based on embedding similarity to reference model.

    Args:
        requests: List of processed requests (modified in place)
        reference_model: Model name to use as reference for similarity
        device: Device for embedding model
    """
    from block.predictor.cara.offline_training.similarity_scorer import SimilarityScorer

    logger.info(f"Computing quality scores using similarity to {reference_model}...")

    scorer = SimilarityScorer(
        reference_model=reference_model,
        device=device,
    )

    for idx, req in enumerate(requests):
        if (idx + 1) % 100 == 0:
            logger.info(f"Scored {idx + 1}/{len(requests)} requests")

        prompt = req["prompt"]

        # Build responses list for all models in this request
        responses = []
        for model_name, model_data in req["models"].items():
            response_text = model_data.get("_response", "")
            if response_text:
                responses.append((model_name, response_text))

        if not responses:
            continue

        # Score using similarity
        scores = scorer.score(prompt, responses)

        # Store scores in model data
        for model_name, model_data in req["models"].items():
            model_data["quality_score"] = scores.get(model_name, 0.5)

    logger.info(f"Completed similarity scoring for {len(requests)} requests")


def compute_quality_scores_compression(requests: List[Dict]) -> None:
    """Compute quality scores based on compression ratio for all models.

    Simple heuristic: higher compression ratio = better quality.
    Normalized to [0, 1] range.

    Args:
        requests: List of processed requests (modified in place)
    """
    logger.info("Computing quality scores from compression ratio...")

    for req in requests:
        for model_name, model_data in req["models"].items():
            compression_ratio = model_data["compression_ratio"]

            # Normalize: typical range is 0.2 - 0.6 for text
            # Map to [0, 1] where 0.2 -> 0.0 and 0.6+ -> 1.0
            quality_score = max(0.0, min(1.0, (compression_ratio - 0.2) / 0.4))
            model_data["quality_score"] = round(quality_score, 4)


def compute_quality_scores_llm_judge(
    requests: List[Dict],
    judge_models: List[str],
    device: str = "cuda",
    batch_size: int = 8,
    hf_token: Optional[str] = None,
    score_min: int = 1,
    score_max: int = 10,
    use_rationale: bool = True,
) -> set:
    """Compute quality scores using multiple LLM judges for all models in each request.

    Args:
        requests: List of processed requests (modified in place)
        judge_models: List of HuggingFace model names for judging
        device: Device for judge models
        batch_size: Batch size for inference
        hf_token: HuggingFace token for gated models
        score_min: Minimum score value for rating scale
        score_max: Maximum score value for rating scale
        use_rationale: Whether to use rationale-based prompting (improves accuracy)

    Returns:
        Set of request indices that had scoring failures (to be filtered out)
    """
    from block.predictor.cara.offline_training.llm_judge_scorer import LLMJudgeScorer

    failed_indices = set()

    for judge_model in judge_models:
        logger.info(f"\n{'='*40}")
        logger.info(f"Running judge: {judge_model}")
        logger.info(f"{'='*40}")

        scorer = LLMJudgeScorer(
            judge_model=judge_model,
            batch_size=batch_size,
            device=device,
            hf_token=hf_token,
            score_min=score_min,
            score_max=score_max,
            use_rationale=use_rationale,
        )

        total_scored = 0
        judge_failures = 0

        # Build a flat list of (prompt, model_name, response) and an index map
        flat_items: List[Tuple[str, str, str]] = []
        index_map: List[Tuple[int, str]] = []  # (req_idx, model_name)
        direct_fail_positions: List[int] = []

        for idx, req in enumerate(requests):
            prompt = req["prompt"]
            for llm_model_name, model_data in req["models"].items():
                response = model_data.get("_response", "")
                if response:
                    index_map.append((idx, llm_model_name))
                    flat_items.append((prompt, llm_model_name, response))
                else:
                    # Mark as direct failure to preserve prior semantics
                    index_map.append((idx, llm_model_name))
                    flat_items.append((prompt, llm_model_name, ""))
                    direct_fail_positions.append(len(index_map) - 1)

        # Process in batches using scorer.score_pairs()
        # Results are aligned with flat_items order
        results: List[Optional[float]] = []
        for start in range(0, len(flat_items), batch_size):
            chunk = flat_items[start:start + batch_size]
            chunk_scores = scorer.score_pairs(chunk)
            results.extend(chunk_scores)

        judge_key = f"quality_score_{judge_model.replace('/', '_')}"

        # Apply results back to requests and track failures
        # Also enforce direct failures for empty responses
        for pos, score in enumerate(results):
            req_idx, llm_model_name = index_map[pos]
            if pos in direct_fail_positions:
                # Force failure (empty response)
                requests[req_idx]["models"][llm_model_name][judge_key] = None
                judge_failures += 1
                failed_indices.add(req_idx)
                continue

            if score is None:
                judge_failures += 1
                failed_indices.add(req_idx)
            else:
                total_scored += 1

            requests[req_idx]["models"][llm_model_name][judge_key] = score

        logger.info(f"Completed {judge_model}: {total_scored} scores, {judge_failures} failures")

        # Print parsing statistics
        scorer.print_parsing_stats()

        # Free memory
        del scorer

    # Set primary quality_score to first judge's scores
    if judge_models:
        primary_judge = judge_models[0]
        judge_key = f"quality_score_{primary_judge.replace('/', '_')}"
        for req in requests:
            for model_data in req["models"].values():
                model_data["quality_score"] = model_data.get(judge_key)

    logger.info(f"\nTotal requests with scoring failures: {len(failed_indices)}")

    return failed_indices


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
        logger.info(f"  IQR: [{stats['q50']:.4f}, {stats['q95']:.4f}]")

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


def save_incomplete_requests(
    incomplete_requests: List[Dict],
    output_path: Path,
    expected_models: set,
) -> None:
    """Save incomplete requests in benchmark dataset format (JSONL) for re-running.

    Output format matches collect_data.py for direct use with benchmark_serving.py:
    {"id": 0, "source": "incomplete/request_id", "prompt": "..."}

    Args:
        incomplete_requests: List of requests missing some models
        output_path: Output file path (.jsonl)
        expected_models: Set of expected model names
    """
    if not incomplete_requests:
        return

    # Collect statistics on missing models
    missing_counts = defaultdict(int)
    for req in incomplete_requests:
        for model in req.get("_missing_models", []):
            missing_counts[model] += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write in JSONL format for benchmark_serving.py
    with open(output_path, 'w', encoding='utf-8') as f:
        for idx, req in enumerate(incomplete_requests):
            record = {
                "id": idx,
                "source": f"incomplete/{req['request_id']}",
                "prompt": req["prompt"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info(f"Saved {len(incomplete_requests)} incomplete requests to: {output_path}")
    logger.info(f"  Format: JSONL (compatible with benchmark_serving.py)")
    logger.info(f"  Missing model counts:")
    for model, count in sorted(missing_counts.items()):
        logger.info(f"    {model}: {count} requests")


def save_training_data(
    requests: List[Dict],
    output_path: Path,
    models: List[str],
    dataset_name: str,
    scoring_method: str,
    include_response: bool = False,
) -> None:
    """Save processed data in training format.

    Args:
        requests: List of processed requests
        output_path: Output file path
        models: List of model names
        dataset_name: Dataset name for metadata
        scoring_method: Scoring method used
        include_response: If True, include full response text
    """
    # Clean up internal fields
    for req in requests:
        # Remove _missing_models field
        if "_missing_models" in req:
            del req["_missing_models"]
        for model_data in req["models"].values():
            if not include_response and "_response" in model_data:
                del model_data["_response"]

    output_data = {
        "dataset_name": dataset_name,
        "scoring_method": scoring_method,
        "num_requests": len(requests),
        "models": sorted(models),
        "requests": requests,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    file_size_mb = output_path.stat().st_size / 1024 / 1024
    logger.info(f"Saved training data to: {output_path}")
    logger.info(f"  Requests: {len(requests)}")
    logger.info(f"  Models: {sorted(models)}")
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

    # Model configuration
    parser.add_argument(
        "--expected-models",
        type=str,
        nargs="+",
        default=None,
        help="Override auto-detected models. If not specified, models are detected from data."
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Allow incomplete requests (missing some models). Default: require all models."
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
        choices=["llm_judge", "similarity", "compression", "none"],
        default="llm_judge",
        help="Quality scoring method. 'similarity' requires --reference-model and multi-model data."
    )
    parser.add_argument(
        "--reference-model",
        type=str,
        default=None,
        help="Reference model for similarity scoring (e.g., Qwen/Qwen2.5-72B). Required for --scoring-method=similarity."
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
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Hugging Face access token to use for gated model repos",
    )
    parser.add_argument(
        "--score-min",
        type=int,
        default=1,
        help="Minimum score value for LLM judge rating scale (default: 1)"
    )
    parser.add_argument(
        "--score-max",
        type=int,
        default=10,
        help="Maximum score value for LLM judge rating scale (default: 10)"
    )
    parser.add_argument(
        "--disable-rationale",
        action="store_true",
        help="Disable rationale/reasoning step in judge prompt (faster but less accurate)"
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

    try:
        # =================================================================
        # Step 1: Load data and detect format
        # =================================================================
        logger.info("\n[1/5] Loading benchmark data...")
        data = load_benchmark_results(args.input)
        data_format = detect_data_format(data)
        logger.info(f"Detected format: {data_format}")

        # =================================================================
        # Step 2: Collect all models from data (first pass)
        # =================================================================
        logger.info("\n[2/5] Collecting model information...")
        all_models, model_counts, request_models = collect_all_models(data)
        total_requests = len(data.get("response_details", []))

        log_model_statistics(all_models, model_counts, total_requests)

        # Determine expected models
        if args.expected_models:
            expected_models = set(args.expected_models)
            logger.info(f"Using user-specified expected models: {sorted(expected_models)}")
            # Validate all expected models exist in data
            missing = expected_models - all_models
            if missing:
                logger.error(f"Expected models not found in data: {missing}")
                logger.error(f"Available models: {sorted(all_models)}")
                return 1
        else:
            expected_models = all_models
            logger.info(f"Using auto-detected models: {sorted(expected_models)}")

        # =================================================================
        # Validate reference model for similarity scoring
        # =================================================================
        if args.scoring_method == "similarity":
            if not args.reference_model:
                logger.error("--scoring-method=similarity requires --reference-model")
                return 1
            if len(expected_models) < 2:
                logger.error("Similarity scoring requires multi-model data (2+ models)")
                return 1
            if args.reference_model not in expected_models:
                logger.error(f"Reference model '{args.reference_model}' not found in data")
                logger.error(f"Available models: {sorted(expected_models)}")
                return 1
            logger.info(f"Reference model for similarity: {args.reference_model}")

        # Log configuration
        logger.info("\n" + "=" * 60)
        logger.info("CARA BENCHMARK DATA PREPROCESSING")
        logger.info("=" * 60)
        logger.info(f"Input:          {args.input}")
        logger.info(f"Output:         {args.output}")
        logger.info(f"Data format:    {data_format}")
        logger.info(f"Models:         {sorted(expected_models)}")
        logger.info(f"Scoring:        {args.scoring_method}")
        if args.scoring_method == "llm_judge":
            logger.info(f"Judge model(s): {args.judge_models}")
        if compare_judges:
            logger.info(f"Compare mode:   ENABLED ({len(args.judge_models)} judges)")
            logger.info(f"Divergence threshold: {args.divergence_threshold}")
        logger.info(f"Require all models: {not args.allow_incomplete}")
        logger.info(f"Min tokens:     {args.min_output_tokens}")
        logger.info(f"Max tokens:     {args.max_output_tokens}")
        logger.info(f"Filter truncated: {args.filter_truncated}")
        logger.info(f"Filter repetition: {args.filter_high_repetition}")
        logger.info("=" * 60)

        # =================================================================
        # Step 3: Process and filter data
        # =================================================================
        logger.info("\n[3/5] Processing and filtering...")
        complete_requests, incomplete_requests, stats = process_benchmark_data(
            data=data,
            expected_models=expected_models,
            min_output_tokens=args.min_output_tokens,
            max_output_tokens=args.max_output_tokens,
            filter_truncated=args.filter_truncated,
            filter_high_repetition=args.filter_high_repetition,
            min_compression_ratio=args.min_compression_ratio,
            require_all_models=not args.allow_incomplete,
        )

        stats.log()

        # Save incomplete requests for re-running
        if incomplete_requests:
            incomplete_path = args.output.parent / f"{args.output.stem}_incomplete.jsonl"
            save_incomplete_requests(incomplete_requests, incomplete_path, expected_models)

        if not complete_requests:
            logger.error("No valid complete requests after filtering!")
            return 1

        requests = complete_requests

        # =================================================================
        # Step 4: Compute quality scores
        # =================================================================
        logger.info("\n[4/5] Computing quality scores...")
        failed_indices = set()

        if args.scoring_method == "compression":
            compute_quality_scores_compression(requests)
        elif args.scoring_method == "llm_judge":
            failed_indices = compute_quality_scores_llm_judge(
                requests,
                judge_models=args.judge_models,
                device=args.device,
                batch_size=args.batch_size,
                hf_token=args.hf_token,
                score_min=args.score_min,
                score_max=args.score_max,
                use_rationale=not args.disable_rationale,
            )
        elif args.scoring_method == "similarity":
            compute_quality_scores_similarity(
                requests,
                reference_model=args.reference_model,
                device=args.device,
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

        # =================================================================
        # Step 5: Analysis and save output
        # =================================================================
        logger.info("\n[5/5] Saving output...")

        # Run comparison analysis if enabled
        if compare_judges:
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

        # Save training data
        save_training_data(
            requests=requests,
            output_path=args.output,
            models=list(expected_models),
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
