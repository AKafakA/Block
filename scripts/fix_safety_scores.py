#!/usr/bin/env python3
"""Fix quality scores for harmful prompts in the cara_model_estimator dataset.

Loads the existing HF dataset (train/test/full), identifies harmful prompts
via source map, re-scores ONLY those with the safety-aware LLM judge template,
replaces the scores in-place, and uploads back preserving the original split.

Usage:
    # Dry run (no scoring, no upload, just print stats):
    python scripts/fix_safety_scores.py --dry-run

    # Full run, save locally first:
    python scripts/fix_safety_scores.py --no-upload

    # Full run with upload:
    python scripts/fix_safety_scores.py
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from block.predictor.cara.offline_training.prepare_benchmark_data import (
    strip_chat_template, HARMFUL_DATASETS, HARMFUL_SOURCE_PREFIXES,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Fix safety scores in cara_model_estimator")
    p.add_argument("--hf-dataset-json", type=Path, required=True,
                    help="Full scored dataset JSON from HF (cara_v3_all_training.json)")
    p.add_argument("--train-jsonl", type=Path, default=None,
                    help="Existing train split JSONL (preserves split)")
    p.add_argument("--test-jsonl", type=Path, default=None,
                    help="Existing test split JSONL (preserves split)")
    p.add_argument("--source-map", type=Path, required=True,
                    help="JSONL from collect_data.py with 'source' and 'prompt' fields")
    p.add_argument("--judge-model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--hf-token", type=str, default="HF_TOKEN_PLACEHOLDER")
    p.add_argument("--hf-repo", type=str, default="asdwb/cara_model_estimator")
    p.add_argument("--dry-run", action="store_true", help="Only print stats, don't score or upload")
    p.add_argument("--no-upload", action="store_true", help="Save locally, don't upload to HF")
    p.add_argument("--output-dir", type=Path, default=Path("data/cara/fixed_scores"))
    return p.parse_args()


def load_source_map(path: Path) -> dict:
    """Load prompt -> source mapping from collect_data.py output."""
    source_map = {}
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            key = item.get("prompt", "").strip()
            source_map[key] = item.get("source", "")
    logger.info(f"Source map: {len(source_map)} entries")
    return source_map


def is_harmful_source(source: str) -> bool:
    """Check if a source string indicates a harmful prompt."""
    if not source:
        return False
    prefix = source.split("/")[0]
    return prefix in HARMFUL_DATASETS or prefix in HARMFUL_SOURCE_PREFIXES


def recover_source_and_tag(requests: list, source_map: dict):
    """Recover source fields and tag harmful prompts. Returns list of harmful indices."""
    harmful_indices = []
    for idx, req in enumerate(requests):
        key = strip_chat_template(req.get("prompt", "")).strip()
        source = source_map.get(key, "")
        if source:
            req["source"] = source
            req["dataset"] = source.split("/")[0] if "/" in source else source
        if is_harmful_source(source):
            harmful_indices.append(idx)
    return harmful_indices


def score_harmful(requests, harmful_indices, args):
    """Run safety judge on harmful prompts, return {prompt_key: {model: score}}."""
    from block.predictor.cara.offline_training.llm_judge_scorer import LLMJudgeScorer

    # Build pairs
    pairs = []
    pair_map = []  # (req_idx, model_name)
    for idx in harmful_indices:
        req = requests[idx]
        prompt = req["prompt"]
        for model_name, model_data in req["models"].items():
            response = model_data.get("response", "")
            if response:
                pairs.append((prompt, model_name, response))
                pair_map.append((idx, model_name))

    logger.info(f"Scoring {len(pairs)} pairs from {len(harmful_indices)} harmful prompts...")

    scorer = LLMJudgeScorer(
        judge_model=args.judge_model,
        batch_size=args.batch_size,
        device=args.device,
        hf_token=args.hf_token,
        score_min=1,
        score_max=10,
        use_rationale=True,
    )

    is_harmful = [True] * len(pairs)
    scores = scorer.score_pairs(pairs, is_harmful=is_harmful)

    # Build lookup: stripped_prompt -> {model_name: score}
    # Use same key as original judge to REPLACE scores, not add new ones
    judge_key = args.judge_model.replace("/", "_")
    score_lookup = {}
    for (req_idx, model_name), score in zip(pair_map, scores):
        if score is not None:
            prompt_key = strip_chat_template(requests[req_idx]["prompt"]).strip()
            if prompt_key not in score_lookup:
                score_lookup[prompt_key] = {}
            score_lookup[prompt_key][model_name] = score

    logger.info(f"Scored {len(score_lookup)} unique harmful prompts")
    return score_lookup, judge_key


def apply_scores(requests, score_lookup, judge_key, source_map):
    """Apply safety scores to requests in-place. Returns count of updated records."""
    updated = 0
    tagged_harmful = 0
    for req in requests:
        key = strip_chat_template(req.get("prompt", "")).strip()
        source = source_map.get(key, "")

        # Tag harmful
        if is_harmful_source(source):
            req["is_harmful"] = True
            tagged_harmful += 1
            if "source" not in req and source:
                req["source"] = source
            if "dataset" not in req and source:
                req["dataset"] = source.split("/")[0] if "/" in source else source

        # Apply new scores
        if key in score_lookup:
            for model_name, score in score_lookup[key].items():
                md = req.get("models", {}).get(model_name, {})
                if md:
                    if "llm_judge_scores" not in md:
                        md["llm_judge_scores"] = {}
                    md["llm_judge_scores"][judge_key] = score
                    updated += 1

    return updated, tagged_harmful


def main():
    args = parse_args()
    source_map = load_source_map(args.source_map)

    # 1. Load the full HF dataset (has responses for scoring)
    logger.info(f"Loading {args.hf_dataset_json}...")
    with open(args.hf_dataset_json) as f:
        full_data = json.load(f)
    if isinstance(full_data, dict) and "requests" in full_data:
        requests = full_data["requests"]
    elif isinstance(full_data, dict) and "response_details" in full_data:
        requests = full_data["response_details"]
    elif isinstance(full_data, list):
        requests = full_data
        full_data = {"requests": requests}
    else:
        raise ValueError(f"Unknown format, top keys: {list(full_data.keys())[:5]}")
    logger.info(f"Full dataset: {len(requests)} requests")

    # 2. Identify harmful prompts
    harmful_indices = recover_source_and_tag(requests, source_map)
    logger.info(f"Harmful: {len(harmful_indices)}/{len(requests)}")
    harmful_ds = Counter(requests[i].get("dataset", "?") for i in harmful_indices)
    for k, v in harmful_ds.most_common():
        logger.info(f"  {k}: {v}")

    if args.dry_run:
        # Show current scores
        models = list(requests[harmful_indices[0]]["models"].keys()) if harmful_indices else []
        for model in models:
            scores = []
            for idx in harmful_indices:
                md = requests[idx]["models"].get(model, {})
                for _, s in md.get("llm_judge_scores", {}).items():
                    scores.append(s)
                    break
            if scores:
                logger.info(f"  {model.split('-')[-1]} current avg on harmful: {sum(scores)/len(scores):.3f}")
        logger.info("Dry run complete.")
        return

    # 3. Run safety judge on harmful prompts (using full dataset which has responses)
    score_lookup, judge_key = score_harmful(requests, harmful_indices, args)

    # 4. Apply scores to FULL dataset
    logger.info("Applying scores to full dataset...")
    n_updated, n_harmful = apply_scores(requests, score_lookup, judge_key, source_map)
    logger.info(f"Full: updated {n_updated} model scores, tagged {n_harmful} harmful")

    # Show new distribution
    models = list(requests[harmful_indices[0]]["models"].keys())
    for model in models:
        vals = []
        for idx in harmful_indices:
            s = requests[idx]["models"].get(model, {}).get("llm_judge_scores", {}).get(judge_key)
            if s is not None:
                vals.append(s)
        if vals:
            avg = sum(vals) / len(vals)
            high = sum(1 for v in vals if v > 0.5)
            logger.info(f"  {model.split('-')[-1]}: avg={avg:.3f}, refused(>0.5)={high}/{len(vals)}")

    # 5. Save updated full dataset
    args.output_dir.mkdir(parents=True, exist_ok=True)
    full_path = args.output_dir / "cara_v3_all_training.json"
    full_data["requests"] = requests
    with open(full_path, "w") as f:
        json.dump(full_data, f, ensure_ascii=False)
    logger.info(f"Saved full: {full_path}")

    # 6. Apply scores to existing train/test splits (preserve split, just replace scores)
    for split_name, split_path in [("train", args.train_jsonl), ("test", args.test_jsonl)]:
        if split_path is None or not split_path.exists():
            logger.warning(f"No {split_name} split provided, skipping")
            continue

        logger.info(f"Updating {split_name} split: {split_path}")
        split_requests = []
        with open(split_path) as f:
            for line in f:
                split_requests.append(json.loads(line))

        n_up, n_harm = apply_scores(split_requests, score_lookup, judge_key, source_map)
        logger.info(f"  {split_name}: updated {n_up} model scores, tagged {n_harm} harmful")

        out_path = args.output_dir / split_path.name
        with open(out_path, "w") as f:
            for req in split_requests:
                f.write(json.dumps(req, ensure_ascii=False) + "\n")
        logger.info(f"  Saved: {out_path}")

    # 7. Upload to HF
    if not args.no_upload:
        logger.info(f"Uploading to {args.hf_repo}...")
        from huggingface_hub import HfApi
        api = HfApi(token=args.hf_token)

        for local_file in args.output_dir.glob("*"):
            api.upload_file(
                path_or_fileobj=str(local_file),
                path_in_repo=local_file.name,
                repo_id=args.hf_repo,
                repo_type="dataset",
            )
            logger.info(f"  Uploaded: {local_file.name}")
        logger.info("Upload complete!")
    else:
        logger.info("Skipped upload (--no-upload)")


if __name__ == "__main__":
    main()
