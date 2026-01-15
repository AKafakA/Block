#!/usr/bin/env python3
"""
CARA Model Estimation Training Data Preparation

Processes benchmark data to create training datasets for:
1. Length prediction: prompt → output_length per model
2. Model quality: prompt → quality_score per model

Handles data cleaning, filtering bad responses, and quality scoring.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict

from transformers import AutoTokenizer

from block.predictor.cara.offline_training.model_scorer import ModelScorer
from block.predictor.cara.offline_training.similarity_scorer import SimilarityScorer
from block.predictor.cara.offline_training.llm_judge_scorer import LLMJudgeScorer
from block.predictor.cara.offline_training.response_filter import ResponseFilter, ModelResponse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ProcessedRequest:
    """Processed request with cleaned model responses."""
    request_id: str
    prompt: str
    input_len: int
    models: Dict[str, Dict]  # model_name -> {output_length, quality_score, ...}


class DataProcessor:
    """Main data processing pipeline."""

    def __init__(self,
                 tokenizer: AutoTokenizer,
                 scorer: ModelScorer,
                 response_filter: ResponseFilter,
                 exclude_models: List[str] = None,
                 require_all_models: bool = False,
                 expected_models: List[str] = None):
        """
        Args:
            tokenizer: HuggingFace tokenizer for counting tokens
            scorer: ModelScorer instance for quality scoring
            response_filter: ResponseFilter for filtering bad responses
            exclude_models: List of model names to exclude from processing
            require_all_models: If True, only keep requests with all expected models
            expected_models: List of all expected model names (for require_all_models check)
        """
        self.tokenizer = tokenizer
        self.scorer = scorer
        self.filter = response_filter
        self.exclude_models = set(exclude_models or [])
        self.require_all_models = require_all_models

        # Expected models after exclusion
        if expected_models:
            self.expected_models = set(expected_models) - self.exclude_models
        else:
            self.expected_models = None

        # Statistics
        self.stats = {
            "total_requests": 0,
            "filtered_responses": 0,
            "filter_reasons": defaultdict(int),
            "filtered_per_model": defaultdict(int),  # Track filtered count per model
            "valid_requests": 0,
            "models_processed": set(),
            "excluded_models": 0,
            "missing_reference_model": 0,
            "missing_required_models": 0
        }

        if self.exclude_models:
            logger.info(f"Excluding models: {', '.join(sorted(self.exclude_models))}")
        if self.require_all_models:
            if self.expected_models:
                logger.info(
                    f"Requiring all models in each request: "
                    f"{', '.join(sorted(self.expected_models))}"
                )
            else:
                logger.info("Requiring all models (will be determined from data)")

    def process(self, input_path: Path) -> List[ProcessedRequest]:
        """Process input data file.

        Args:
            input_path: Path to cara-best-route-training.json

        Returns:
            List of ProcessedRequest objects
        """
        logger.info(f"Loading data from: {input_path}")

        with open(input_path, 'r') as f:
            data = json.load(f)

        response_details = data.get("response_details", [])
        self.stats["total_requests"] = len(response_details)

        # Auto-detect expected models if require_all_models is enabled but not set
        if self.require_all_models and self.expected_models is None:
            all_models = set()
            # Scan first 100 requests to find all models
            for request_data in response_details[:100]:
                for br in request_data.get("broadcast_results", []):
                    model_name = br.get("model", "unknown")
                    if model_name != "unknown":
                        all_models.add(model_name)
            # Remove excluded models
            self.expected_models = all_models - self.exclude_models
            logger.info(
                f"Auto-detected expected models: {', '.join(sorted(self.expected_models))}"
            )

        logger.info(f"Processing {len(response_details)} requests...")

        processed_requests = []

        for idx, request_data in enumerate(response_details):
            if (idx + 1) % 1000 == 0:
                logger.info(f"Processed {idx + 1}/{len(response_details)} requests")

            processed = self._process_request(request_data)
            if processed:
                processed_requests.append(processed)
                self.stats["valid_requests"] += 1

        logger.info(f"Processing complete: {len(processed_requests)} valid requests")
        self._log_statistics()

        return processed_requests

    def _process_request(self, request_data: Dict) -> Optional[ProcessedRequest]:
        """Process a single request.

        Args:
            request_data: Raw request data from JSON

        Returns:
            ProcessedRequest or None if all responses filtered
        """
        request_id = request_data.get("request_id", "unknown")
        prompt = request_data.get("prompt", "")
        input_len = request_data.get("input_len", 0)
        broadcast_results = request_data.get("broadcast_results", [])

        if not broadcast_results:
            logger.debug(f"Request {request_id}: No broadcast_results")
            return None

        # Parse responses
        responses = []
        for br in broadcast_results:
            model_name = br.get("model", "unknown")

            # Skip excluded models
            if model_name in self.exclude_models:
                self.stats["excluded_models"] += 1
                logger.debug(f"Request {request_id}: Excluding model {model_name}")
                continue

            resp = ModelResponse(
                model_name=model_name,
                instance_id=br.get("instance_id", "unknown"),
                host=br.get("host", "unknown"),
                generated_text=br.get("generated_text", ""),
                output_tokens=br.get("output_tokens", 0),
                ttft=br.get("ttft", 0.0),
                server_latency=br.get("server_latency", 0.0),
                success=br.get("success", False),
                error=br.get("error", ""),
                itl=br.get("itl", [])
            )
            responses.append(resp)

        # Filter responses
        valid_responses = []
        for resp in responses:
            is_valid, reason = self.filter.is_valid(resp)
            if is_valid:
                valid_responses.append(resp)
                self.stats["models_processed"].add(resp.model_name)
            else:
                self.stats["filtered_responses"] += 1
                self.stats["filter_reasons"][reason] += 1
                self.stats["filtered_per_model"][resp.model_name] += 1
                logger.debug(
                    f"Request {request_id}, Model {resp.model_name}: "
                    f"Filtered - {reason}"
                )

        if not valid_responses:
            logger.debug(f"Request {request_id}: All responses filtered")
            return None

        # Get valid model names
        valid_model_names = {resp.model_name for resp in valid_responses}

        # Check if reference model is present (for similarity scoring)
        reference_model = self._get_reference_model(valid_model_names)
        if reference_model and reference_model not in valid_model_names:
            self.stats["missing_reference_model"] += 1
            logger.debug(
                f"Request {request_id}: Reference model {reference_model} missing, "
                f"skipping request to ensure consistent scoring"
            )
            return None

        # Check if all required models are present (if require_all_models enabled)
        if self.require_all_models:
            expected = self.expected_models if self.expected_models else valid_model_names
            if valid_model_names != expected:
                missing = expected - valid_model_names
                self.stats["missing_required_models"] += 1
                logger.debug(
                    f"Request {request_id}: Missing required models {missing}, "
                    f"skipping request"
                )
                return None

        # Recompute output lengths using tokenizer
        for resp in valid_responses:
            if resp.generated_text:
                try:
                    token_ids = self.tokenizer(
                        resp.generated_text,
                        add_special_tokens=False
                    ).input_ids
                    resp.output_tokens = len(token_ids)
                except Exception as e:
                    logger.warning(
                        f"Failed to tokenize response for {resp.model_name}: {e}. "
                        f"Using original output_tokens={resp.output_tokens}"
                    )

        # Compute quality scores
        response_tuples = [
            (resp.model_name, resp.generated_text)
            for resp in valid_responses
        ]
        quality_scores = self.scorer.score(prompt, response_tuples)

        # Build models dict
        models_dict = {}
        for resp in valid_responses:
            models_dict[resp.model_name] = {
                "output_length": resp.output_tokens,
                "quality_score": quality_scores.get(resp.model_name, 0.0),
                "ttft": resp.ttft,
                "server_latency": resp.server_latency,
                "instance_id": resp.instance_id,
                "host": resp.host
            }

        return ProcessedRequest(
            request_id=request_id,
            prompt=prompt,
            input_len=input_len,
            models=models_dict
        )

    def _get_reference_model(self, available_models: set) -> Optional[str]:
        """Get the reference model for similarity scoring.

        Args:
            available_models: Set of available model names

        Returns:
            Reference model name if using similarity scoring, None otherwise
        """
        # Only applies to SimilarityScorer
        from block.predictor.cara.offline_training.similarity_scorer import SimilarityScorer
        if not isinstance(self.scorer, SimilarityScorer):
            return None

        # Return the explicitly set reference model
        return self.scorer.reference_model

    def _log_statistics(self):
        """Log processing statistics."""
        logger.info("=" * 80)
        logger.info("PROCESSING STATISTICS")
        logger.info("=" * 80)
        logger.info(f"Total requests: {self.stats['total_requests']}")
        logger.info(f"Valid requests: {self.stats['valid_requests']}")
        logger.info(
            f"Filtered requests: {self.stats['total_requests'] - self.stats['valid_requests']}"
        )
        logger.info(f"Filtered responses: {self.stats['filtered_responses']}")

        if self.stats['excluded_models'] > 0:
            logger.info(f"Excluded models (responses): {self.stats['excluded_models']}")
            logger.info(f"  Models excluded: {', '.join(sorted(self.exclude_models))}")

        if self.stats['missing_reference_model'] > 0:
            logger.info(
                f"Requests filtered (missing reference model): "
                f"{self.stats['missing_reference_model']}"
            )

        if self.stats['missing_required_models'] > 0:
            logger.info(
                f"Requests filtered (missing required models): "
                f"{self.stats['missing_required_models']}"
            )

        logger.info(f"Models processed: {len(self.stats['models_processed'])}")
        for model in sorted(self.stats['models_processed']):
            logger.info(f"  - {model}")

        # Report filtered counts per model
        if self.stats['filtered_per_model']:
            logger.info("\nFiltered responses per model:")
            for model, count in sorted(
                self.stats['filtered_per_model'].items(),
                key=lambda x: x[1],
                reverse=True
            ):
                logger.info(f"  - {model}: {count}")

        if self.stats['filter_reasons']:
            logger.info("\nTop filter reasons:")
            for reason, count in sorted(
                self.stats['filter_reasons'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]:  # Show top 10
                logger.info(f"  - {reason}: {count}")

        logger.info("=" * 80)


def save_training_data(
    processed_requests: List[ProcessedRequest],
    output_path: Path,
    dataset_name: str,
    scoring_method: str
):
    """Save processed data to JSON file.

    Output format:
    {
      "dataset_name": "best-route",
      "scoring_method": "similarity",
      "num_requests": 13500,
      "models": ["Qwen/Qwen2.5-72B", ...],
      "requests": [
        {
          "request_id": "...",
          "prompt": "...",
          "input_len": 128,
          "models": {
            "Qwen/Qwen2.5-72B": {
              "output_length": 271,
              "quality_score": 1.0,
              "ttft": 0.05,
              "server_latency": 1.69,
              "instance_id": "...",
              "host": "..."
            }
          }
        }
      ]
    }
    """
    # Collect all unique models
    all_models = set()
    for req in processed_requests:
        all_models.update(req.models.keys())

    output_data = {
        "dataset_name": dataset_name,
        "scoring_method": scoring_method,
        "num_requests": len(processed_requests),
        "models": sorted(list(all_models)),
        "requests": []
    }

    for req in processed_requests:
        output_data["requests"].append({
            "request_id": req.request_id,
            "prompt": req.prompt,
            "input_len": req.input_len,
            "models": req.models
        })

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    file_size_mb = output_path.stat().st_size / 1024 / 1024

    logger.info(f"\n✅ Saved training data to: {output_path}")
    logger.info(f"   Requests: {len(processed_requests)}")
    logger.info(f"   Models: {len(all_models)}")
    for model in sorted(all_models):
        logger.info(f"     - {model}")
    logger.info(f"   File size: {file_size_mb:.2f} MB")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare CARA model estimation training data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Input/output
    parser.add_argument(
        "-i", "--input",
        type=Path,
        default="data/cara/cara-best-route-training.json",
        help="Input JSON file"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output JSON file (default: auto-generated as [DATASET]_[SCORE]_model_estimation_training.json)"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="best-route",
        help="Dataset name for output filename"
    )

    # Tokenizer
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="Qwen/Qwen2.5-3B",
        help="HuggingFace tokenizer name or path (default: Qwen/Qwen2.5-3B)"
    )

    # Scoring
    parser.add_argument(
        "--scoring-method",
        type=str,
        choices=["similarity", "llm_judge"],
        default="similarity",
        help="Quality scoring method"
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model for similarity scoring"
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="Unbabel/M-Prometheus-7B",
        help="Judge LLM for llm_judge scoring"
    )
    parser.add_argument(
        "--judge-prompt",
        type=Path,
        default=None,
        help="Custom judge prompt template file"
    )
    parser.add_argument(
        "--reference-model",
        type=str,
        default="Qwen/Qwen2.5-72B",
        help="Reference model for similarity scoring (default: Qwen/Qwen2.5-72B)"
    )

    # Filtering
    parser.add_argument(
        "--min-output-tokens",
        type=int,
        default=3,
        help="Minimum valid output length"
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=1024,
        help="Max output tokens (responses at this length are truncated)"
    )
    parser.add_argument(
        "--min-compression-ratio",
        type=float,
        default=0.2,
        help="Minimum compression ratio (lower = more repetitive, filtered)"
    )
    parser.add_argument(
        "--exclude-models",
        nargs="+",
        default=[],
        help="Models to exclude from processing (e.g., 'Qwen/Qwen2.5-3B' 'Qwen/Qwen2.5-32B')"
    )
    parser.add_argument(
        "--require-all-models",
        action="store_true",
        help="Only keep requests where all non-excluded models have valid responses"
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for models (cpu, cuda)"
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

    # Configure logging
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)

    # Validate input
    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        return 1

    # Determine output path
    if args.output is None:
        # Format: [DATASET-NAME]_[SCORE]_model_estimation_training.json
        output_filename = (
            f"{args.dataset_name}_{args.scoring_method}_"
            f"model_estimation_training.json"
        )
        args.output = args.input.parent / output_filename

    logger.info("=" * 80)
    logger.info("CARA TRAINING DATA PREPARATION")
    logger.info("=" * 80)
    logger.info(f"Input:      {args.input}")
    logger.info(f"Output:     {args.output}")
    logger.info(f"Tokenizer:  {args.tokenizer}")
    logger.info(f"Scoring:    {args.scoring_method}")
    logger.info(f"Device:     {args.device}")
    logger.info("=" * 80)

    try:
        # Load tokenizer
        logger.info("\n[1/4] Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer,
            trust_remote_code=True
        )
        logger.info(f"✓ Tokenizer loaded: vocab_size={tokenizer.vocab_size}")

        # Create scorer
        logger.info(f"\n[2/4] Creating {args.scoring_method} scorer...")
        if args.scoring_method == "similarity":
            scorer = SimilarityScorer(
                reference_model=args.reference_model,
                embedding_model=args.embedding_model,
                device=args.device
            )
        elif args.scoring_method == "llm_judge":
            judge_prompt = None
            if args.judge_prompt:
                logger.info(f"Loading custom judge prompt from: {args.judge_prompt}")
                with open(args.judge_prompt, 'r') as f:
                    judge_prompt = f.read()

            scorer = LLMJudgeScorer(
                judge_model=args.judge_model,
                judge_prompt_template=judge_prompt,
                device=args.device
            )
        else:
            logger.error(f"Unknown scoring method: {args.scoring_method}")
            return 1
        logger.info("✓ Scorer created")

        # Create response filter
        logger.info("\n[3/4] Creating response filter...")
        response_filter = ResponseFilter(
            min_output_tokens=args.min_output_tokens,
            max_output_tokens=args.max_output_tokens,
            min_compression_ratio=args.min_compression_ratio
        )
        logger.info("✓ Filter created")

        # Create processor
        processor = DataProcessor(
            tokenizer=tokenizer,
            scorer=scorer,
            response_filter=response_filter,
            exclude_models=args.exclude_models,
            require_all_models=args.require_all_models
        )

        # Process data
        logger.info("\n[4/4] Processing data...")
        processed_requests = processor.process(args.input)

        if not processed_requests:
            logger.error("\n❌ No valid requests after filtering!")
            return 1

        # Save output
        save_training_data(
            processed_requests,
            args.output,
            args.dataset_name,
            args.scoring_method
        )

        logger.info("\n" + "=" * 80)
        logger.info("✅ PROCESSING COMPLETE!")
        logger.info("=" * 80)
        return 0

    except KeyboardInterrupt:
        logger.warning("\n\n⚠️  Interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"\n❌ Processing failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())