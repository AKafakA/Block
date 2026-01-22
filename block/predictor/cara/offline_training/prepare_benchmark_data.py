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
        "--judge-model",
        type=str,
        default="Qwen/Qwen2.5-0.5B",
        help="Judge model for llm_judge scoring"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for LLM judge (cuda, cpu)"
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

    logger.info("=" * 60)
    logger.info("CARA BENCHMARK DATA PREPROCESSING")
    logger.info("=" * 60)
    logger.info(f"Input:          {args.input}")
    logger.info(f"Output:         {args.output}")
    logger.info(f"Model:          {args.model_name}")
    logger.info(f"Scoring:        {args.scoring_method}")
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
        if args.scoring_method == "compression":
            compute_quality_scores_compression(requests, args.model_name)
        elif args.scoring_method == "llm_judge":
            compute_quality_scores_llm_judge(
                requests,
                args.model_name,
                judge_model=args.judge_model,
                device=args.device,
            )
        else:
            logger.info("Skipping quality scoring (method=none)")

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