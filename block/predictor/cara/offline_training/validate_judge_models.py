#!/usr/bin/env python3
"""
Validate LLM Judge Models Against Human Ratings

Uses the feedbackQA dataset (McGill-NLP/feedbackQA) which contains human ratings
to evaluate which judge model correlates best with human judgment.

This helps you select the best judge model before applying it to your actual dataset.

Usage:
    python block/predictor/cara/offline_training/validate_judge_models.py \
        --judge-models meta-llama/Llama-3.2-3B-Instruct Qwen/Qwen2.5-7B-Instruct \
        --score-min 1 --score-max 10 \
        --sample-size 100 \
        --device cuda \
        --hf-token <your_token>

Features:
- Tests multiple judge models in parallel or sequentially
- Computes Pearson and Spearman correlation with human ratings
- Supports different score scales (1-4, 1-10, etc.)
- Saves detailed results and examples for analysis

Output:
- Correlation metrics for each judge model
- Best performing model recommendation
- Examples of judge ratings vs human ratings
- Detailed results saved to JSON
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import sys

import numpy as np
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Results for a single judge model."""
    judge_model: str
    score_min: int
    score_max: int
    use_rationale: bool
    num_samples: int

    # Metrics
    pearson_r: float = 0.0
    pearson_p: float = 1.0
    spearman_rho: float = 0.0
    spearman_p: float = 1.0

    # Score statistics
    mean_judge_score: float = 0.0
    std_judge_score: float = 0.0
    mean_human_score: float = 0.0
    std_human_score: float = 0.0

    # Detailed data
    predictions: List[float] = field(default_factory=list)
    ground_truth: List[float] = field(default_factory=list)
    examples: List[Dict] = field(default_factory=list)

    # Failures
    num_failures: int = 0
    failure_rate: float = 0.0

    # Parsing statistics
    parsing_stats: Dict = field(default_factory=dict)


def load_feedbackqa_dataset(sample_size: Optional[int] = None,
                            require_agreement: bool = True) -> Tuple[List[Dict], Dict[str, int]]:
    """Load feedbackQA dataset with human ratings.

    Args:
        sample_size: Number of samples to use (None = all)
        require_agreement: Only keep samples where 2 human raters agree

    Returns:
        Tuple of (samples, conversion_dict)
    """
    try:
        from datasets import load_dataset
        import pandas as pd
    except ImportError:
        logger.error("Please install required packages: pip install datasets pandas")
        sys.exit(1)

    logger.info("Loading feedbackQA dataset...")
    dataset = load_dataset("McGill-NLP/feedbackQA", trust_remote_code=True)
    ratings = pd.DataFrame(dataset["train"])

    # Extract human ratings
    ratings["review_1"] = ratings["feedback"].apply(lambda x: x["rating"][0])
    ratings["explanation_1"] = ratings["feedback"].apply(lambda x: x["explanation"][0])
    ratings["review_2"] = ratings["feedback"].apply(lambda x: x["rating"][1])
    ratings["explanation_2"] = ratings["feedback"].apply(lambda x: x["explanation"][1])

    # Map text ratings to numeric scores (1-4 scale)
    conversion_dict = {
        "Excellent": 4,
        "Acceptable": 3,
        "Could be Improved": 2,
        "Bad": 1
    }
    ratings["score_1"] = ratings["review_1"].map(conversion_dict)
    ratings["score_2"] = ratings["review_2"].map(conversion_dict)

    # Filter to samples where raters agree
    if require_agreement:
        ratings = ratings[ratings["score_1"] == ratings["score_2"]].copy()
        logger.info(f"Filtered to {len(ratings)} samples where raters agree")

    ratings["human_score"] = ratings["score_1"]  # Use first rater's score (or agreed score)

    # Sample if requested
    if sample_size and sample_size < len(ratings):
        # Stratified sampling to maintain score distribution
        ratings = ratings.groupby("human_score").apply(
            lambda x: x.sample(min(len(x), sample_size // 4), random_state=42)
        ).reset_index(drop=True)
        logger.info(f"Sampled {len(ratings)} examples (stratified by score)")

    # Convert to list of dicts
    samples = []
    for _, row in ratings.iterrows():
        samples.append({
            "question": row["question"],
            "answer": row["answer"],
            "human_score": row["human_score"],
            "human_explanation": row["explanation_1"],
        })

    logger.info(f"Loaded {len(samples)} samples from feedbackQA")
    logger.info(f"Score distribution: {ratings['human_score'].value_counts().sort_index().to_dict()}")

    return samples, conversion_dict


def evaluate_judge_model(
    judge_model: str,
    samples: List[Dict],
    device: str,
    batch_size: int,
    hf_token: Optional[str],
    score_min: int,
    score_max: int,
    use_rationale: bool,
    human_score_range: Tuple[int, int] = (1, 4),
    save_examples: int = 10,
) -> ValidationResult:
    """Evaluate a single judge model against human ratings.

    Args:
        judge_model: HuggingFace model name
        samples: List of QA samples with human scores
        device: Device for model
        batch_size: Batch size for inference
        hf_token: HuggingFace token
        score_min: Min score for judge
        score_max: Max score for judge
        use_rationale: Use rationale-based prompting
        human_score_range: Min/max of human scores (for normalization)
        save_examples: Number of examples to save in results

    Returns:
        ValidationResult with all metrics
    """
    from block.predictor.cara.offline_training.llm_judge_scorer import LLMJudgeScorer

    logger.info(f"\nEvaluating judge: {judge_model}")
    logger.info(f"Score range: {score_min}-{score_max}, Rationale: {use_rationale}")

    # Initialize judge scorer
    scorer = LLMJudgeScorer(
        judge_model=judge_model,
        batch_size=batch_size,
        device=device,
        hf_token=hf_token,
        score_min=score_min,
        score_max=score_max,
        use_rationale=use_rationale,
    )

    result = ValidationResult(
        judge_model=judge_model,
        score_min=score_min,
        score_max=score_max,
        use_rationale=use_rationale,
        num_samples=len(samples),
    )

    # Evaluate each sample
    for sample in tqdm(samples, desc=f"Scoring with {judge_model}"):
        question = sample["question"]
        answer = sample["answer"]
        human_score = sample["human_score"]

        # Get judge score (already normalized to 0-1)
        scores = scorer.score(question, [("judge", answer)])
        judge_score_normalized = scores.get("judge")

        if judge_score_normalized is None:
            result.num_failures += 1
            continue

        # Store normalized scores (0-1 range)
        # Normalize human score to same range
        human_min, human_max = human_score_range
        human_score_normalized = (human_score - human_min) / (human_max - human_min)

        result.predictions.append(judge_score_normalized)
        result.ground_truth.append(human_score_normalized)

        # Save some examples
        if len(result.examples) < save_examples:
            # Convert back to original scales for display
            judge_score_original = judge_score_normalized * (score_max - score_min) + score_min
            result.examples.append({
                "question": question[:200] + "..." if len(question) > 200 else question,
                "answer": answer[:200] + "..." if len(answer) > 200 else answer,
                "human_score": human_score,
                "judge_score": round(judge_score_original, 2),
                "judge_score_normalized": round(judge_score_normalized, 3),
                "human_score_normalized": round(human_score_normalized, 3),
                "difference": abs(judge_score_normalized - human_score_normalized),
            })

    # Compute statistics
    result.failure_rate = result.num_failures / len(samples)

    if len(result.predictions) > 1:
        # Correlation metrics
        result.pearson_r, result.pearson_p = pearsonr(result.predictions, result.ground_truth)
        result.spearman_rho, result.spearman_p = spearmanr(result.predictions, result.ground_truth)

        # Score statistics
        result.mean_judge_score = np.mean(result.predictions)
        result.std_judge_score = np.std(result.predictions)
        result.mean_human_score = np.mean(result.ground_truth)
        result.std_human_score = np.std(result.ground_truth)
    else:
        logger.warning(f"Not enough valid predictions for {judge_model}")

    # Get parsing statistics before cleanup
    result.parsing_stats = scorer.get_parsing_stats()

    # Print parsing statistics
    scorer.print_parsing_stats()

    # Clean up
    del scorer

    return result


def print_comparison_report(results: List[ValidationResult]):
    """Print a formatted comparison report."""
    print("\n" + "="*80)
    print("JUDGE MODEL VALIDATION RESULTS")
    print("="*80)
    print(f"\nComparing {len(results)} judge models against human ratings (feedbackQA)")
    print(f"Dataset: McGill-NLP/feedbackQA")

    # Sort by Spearman correlation (descending)
    results_sorted = sorted(results, key=lambda x: x.spearman_rho, reverse=True)

    print("\n" + "-"*80)
    print("CORRELATION WITH HUMAN RATINGS (higher is better)")
    print("-"*80)
    print(f"{'Model':<40} {'Pearson r':<12} {'Spearman ρ':<12} {'Success':<10}")
    print("-"*80)

    for result in results_sorted:
        model_name = result.judge_model.split("/")[-1][:38]  # Truncate long names
        success_rate = result.parsing_stats.get('success_rate', 0) * 100
        print(f"{model_name:<40} {result.pearson_r:>6.4f}       {result.spearman_rho:>6.4f}       {success_rate:>5.1f}%")

    # Print parsing method summary
    print("\n" + "-"*80)
    print("PARSING METHOD SUMMARY")
    print("-"*80)
    print(f"{'Model':<40} {'Number':<10} {'Exact':<10} {'Semantic':<10}")
    print("-"*80)

    for result in results_sorted:
        model_name = result.judge_model.split("/")[-1][:38]
        stats = result.parsing_stats
        num_rate = stats.get('number_extraction_rate', 0) * 100
        exact_rate = stats.get('exact_match_rate', 0) * 100
        sem_rate = stats.get('semantic_match_rate', 0) * 100
        print(f"{model_name:<40} {num_rate:>5.1f}%     {exact_rate:>5.1f}%     {sem_rate:>5.1f}%")

    # Semantic similarity stats
    has_semantic = any(r.parsing_stats.get('avg_semantic_similarity') is not None for r in results_sorted)
    if has_semantic:
        print("\n" + "-"*80)
        print("SEMANTIC MATCHING CONFIDENCE (when used)")
        print("-"*80)
        print(f"{'Model':<40} {'Avg Similarity':<20} {'Range':<20}")
        print("-"*80)

        for result in results_sorted:
            model_name = result.judge_model.split("/")[-1][:38]
            stats = result.parsing_stats
            avg_sim = stats.get('avg_semantic_similarity')
            if avg_sim is not None:
                min_sim = stats.get('min_semantic_similarity', 0)
                max_sim = stats.get('max_semantic_similarity', 0)
                print(f"{model_name:<40} {avg_sim:>6.3f}              [{min_sim:.3f}, {max_sim:.3f}]")
            else:
                print(f"{model_name:<40} {'N/A':<20} {'N/A':<20}")

    # Best model
    best = results_sorted[0]
    print("\n" + "="*80)
    print(f"🏆 BEST MODEL: {best.judge_model}")
    print("="*80)
    print(f"Spearman ρ: {best.spearman_rho:.4f} (p={best.spearman_p:.2e})")
    print(f"Pearson r:  {best.pearson_r:.4f} (p={best.pearson_p:.2e})")
    print(f"Scale: {best.score_min}-{best.score_max}, Rationale: {best.use_rationale}")
    print(f"Samples: {len(best.predictions)}/{best.num_samples} successful")

    # Human baseline
    print("\n" + "-"*80)
    print("BASELINE: Inter-human rater agreement")
    print("-"*80)
    print("Note: feedbackQA has 2 human raters. We filtered to samples where they agree.")
    print("If we didn't filter, human-human correlation would be ~0.56 (from HF cookbook)")
    print("Your best judge achieved {:.4f}, which is {}!".format(
        best.spearman_rho,
        "excellent" if best.spearman_rho > 0.7 else "good" if best.spearman_rho > 0.5 else "moderate"
    ))

    # Examples from best model
    if best.examples:
        print("\n" + "-"*80)
        print(f"EXAMPLE RATINGS FROM BEST MODEL ({best.judge_model})")
        print("-"*80)
        for i, ex in enumerate(best.examples[:5], 1):
            print(f"\nExample {i}:")
            print(f"  Question: {ex['question']}")
            print(f"  Answer: {ex['answer']}")
            print(f"  Human score: {ex['human_score']} (normalized: {ex['human_score_normalized']})")
            print(f"  Judge score: {ex['judge_score']} (normalized: {ex['judge_score_normalized']})")
            print(f"  Difference: {ex['difference']:.3f}")

    print("\n" + "="*80)


def save_results(results: List[ValidationResult], output_path: Path, append: bool = False):
    """Save detailed results to JSON.

    Args:
        results: List of ValidationResult objects to save
        output_path: Path to output JSON file
        append: If True, merge with existing results in the file
    """
    existing_judges = []

    # Load existing results if append mode and file exists
    if append and output_path.exists():
        try:
            with open(output_path, "r") as f:
                existing_data = json.load(f)
                existing_judges = existing_data.get("judges", [])
                logger.info(f"Loaded {len(existing_judges)} existing judge results from {output_path}")
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Could not load existing results: {e}. Starting fresh.")
            existing_judges = []

    # Create a dict of existing judges by model name for easy lookup
    existing_by_model = {j["model"]: j for j in existing_judges}

    output = {
        "dataset": "McGill-NLP/feedbackQA",
        "num_judges": 0,  # Will be updated after merging
        "judges": []
    }

    for result in results:
        output["judges"].append({
            "model": result.judge_model,
            "config": {
                "score_min": result.score_min,
                "score_max": result.score_max,
                "use_rationale": result.use_rationale,
            },
            "metrics": {
                "pearson_r": result.pearson_r,
                "pearson_p": result.pearson_p,
                "spearman_rho": result.spearman_rho,
                "spearman_p": result.spearman_p,
                "mean_judge_score": result.mean_judge_score,
                "std_judge_score": result.std_judge_score,
                "mean_human_score": result.mean_human_score,
                "std_human_score": result.std_human_score,
            },
            "samples": {
                "total": result.num_samples,
                "successful": len(result.predictions),
                "failures": result.num_failures,
                "failure_rate": result.failure_rate,
            },
            "parsing_stats": result.parsing_stats,
            "examples": result.examples,
        })

    # Track which models we've added from new results
    new_model_names = {r.judge_model for r in results}

    # Add existing judges that weren't re-evaluated
    if append:
        for model_name, judge_data in existing_by_model.items():
            if model_name not in new_model_names:
                output["judges"].append(judge_data)
                logger.info(f"Kept existing results for: {model_name}")

    # Update count
    output["num_judges"] = len(output["judges"])

    # Sort by Spearman correlation
    output["judges"].sort(key=lambda x: x["metrics"]["spearman_rho"], reverse=True)
    output["best_judge"] = output["judges"][0]["model"]

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"Detailed results saved to: {output_path}")
    if append:
        logger.info(f"Total judges in file: {output['num_judges']} (new: {len(results)}, existing: {output['num_judges'] - len(results)})")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Validate LLM judge models against human ratings (feedbackQA)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--judge-models",
        type=str,
        nargs="+",
        required=True,
        help="Judge models to evaluate (e.g., meta-llama/Llama-3.2-3B-Instruct Qwen/Qwen2.5-7B-Instruct)"
    )
    parser.add_argument(
        "--score-min",
        type=int,
        default=1,
        help="Minimum score value for judge rating scale"
    )
    parser.add_argument(
        "--score-max",
        type=int,
        default=4,
        help="Maximum score value for judge rating scale (default: 4 to match human ratings)"
    )
    parser.add_argument(
        "--disable-rationale",
        action="store_true",
        help="Disable rationale/reasoning step in judge prompt (faster but less accurate)"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Number of samples to use from feedbackQA (None = all, ~500 samples)"
    )
    parser.add_argument(
        "--require-agreement",
        action="store_true",
        default=True,
        help="Only use samples where 2 human raters agree (reduces noise)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for judge models (cuda, cpu)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for judge inference"
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace access token for gated models"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON file for detailed results (default: judge_validation_results.json)"
    )
    parser.add_argument(
        "--save-examples",
        type=int,
        default=20,
        help="Number of example ratings to save per judge"
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append results to existing output file instead of overwriting"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Set output path
    if args.output is None:
        args.output = Path("judge_validation_results.json")

    logger.info("="*80)
    logger.info("LLM JUDGE MODEL VALIDATION")
    logger.info("="*80)
    logger.info(f"Dataset: McGill-NLP/feedbackQA")
    logger.info(f"Judge models: {args.judge_models}")
    logger.info(f"Score scale: {args.score_min}-{args.score_max}")
    logger.info(f"Use rationale: {not args.disable_rationale}")
    logger.info(f"Device: {args.device}")

    # Load dataset
    samples, human_score_mapping = load_feedbackqa_dataset(
        sample_size=args.sample_size,
        require_agreement=args.require_agreement
    )

    # Human score range (feedbackQA uses 1-4)
    human_min = min(human_score_mapping.values())
    human_max = max(human_score_mapping.values())

    # Evaluate each judge model
    results = []
    for judge_model in args.judge_models:
        try:
            result = evaluate_judge_model(
                judge_model=judge_model,
                samples=samples,
                device=args.device,
                batch_size=args.batch_size,
                hf_token=args.hf_token,
                score_min=args.score_min,
                score_max=args.score_max,
                use_rationale=not args.disable_rationale,
                human_score_range=(human_min, human_max),
                save_examples=args.save_examples,
            )
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to evaluate {judge_model}: {e}")
            import traceback
            traceback.print_exc()

    if not results:
        logger.error("No successful judge evaluations!")
        sys.exit(1)

    # Print comparison report
    print_comparison_report(results)

    # Save detailed results
    save_results(results, args.output, append=args.append)

    logger.info("\n" + "="*80)
    logger.info("VALIDATION COMPLETE")
    logger.info("="*80)
    logger.info(f"Best judge: {results[0].judge_model}")
    logger.info(f"Spearman ρ: {results[0].spearman_rho:.4f}")
    logger.info(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
