#!/usr/bin/env python3
"""
Deterministic 80/20 train/test split for CARA training data.

Uses hash-based splitting on request_id for reproducibility across reruns.
The same split is used for all predictor types (length, quality).

Also performs post-processing:
- Strip chat template tags (e.g. Qwen ChatML) from prompts
- Recalibrate LLM judge scores from (r-1)/9 to r/10 normalization
- Recover dataset source labels by matching prompts against original JSONL
"""

import argparse
import hashlib
import json
import logging
import re
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def hash_split(request_id: str, train_ratio: float = 0.8) -> bool:
    """Deterministic split using MD5 hash of request_id.

    Returns True if request should go to train set.
    """
    h = hashlib.md5(str(request_id).encode()).hexdigest()
    # Use first 8 hex chars (32 bits) for uniform distribution
    return (int(h[:8], 16) / 0xFFFFFFFF) < train_ratio


def strip_chat_template(prompt: str) -> str:
    """Strip chat template tags from prompt, returning raw user content.

    Supports multiple chat template formats:
    - ChatML (Qwen, Yi): <|im_start|>user\\n...<|im_end|>
    - Llama-style: [INST] ... [/INST]
    - Gemma-style: <start_of_turn>user\\n...<end_of_turn>
    - Mistral/Zephyr: <|user|>\\n...<|end|>

    Returns the original prompt if no known template is detected.
    """
    # ChatML format (Qwen, Yi, etc.)
    if "<|im_start|>" in prompt:
        user_match = re.search(
            r'<\|im_start\|>user\n(.*?)<\|im_end\|>',
            prompt,
            re.DOTALL,
        )
        if user_match:
            return user_match.group(1).strip()
        # Fallback: strip all ChatML tags
        cleaned = re.sub(r'<\|im_start\|>\w*\n?', '', prompt)
        cleaned = re.sub(r'<\|im_end\|>\n?', '', cleaned)
        cleaned = cleaned.replace("You are a helpful assistant.", "").strip()
        return cleaned

    # Llama-style: [INST] content [/INST]
    if "[INST]" in prompt:
        user_match = re.search(r'\[INST\]\s*(.*?)\s*\[/INST\]', prompt, re.DOTALL)
        if user_match:
            content = user_match.group(1).strip()
            # Strip <<SYS>> system prompt if present
            content = re.sub(r'<<SYS>>.*?<</SYS>>\s*', '', content, flags=re.DOTALL)
            return content.strip()

    # Gemma-style: <start_of_turn>user\n...<end_of_turn>
    if "<start_of_turn>" in prompt:
        user_match = re.search(
            r'<start_of_turn>user\n(.*?)<end_of_turn>',
            prompt,
            re.DOTALL,
        )
        if user_match:
            return user_match.group(1).strip()

    # Mistral/Zephyr-style: <|user|>\n...<|end|>
    if "<|user|>" in prompt:
        user_match = re.search(
            r'<\|user\|>\n?(.*?)<\|end\|>',
            prompt,
            re.DOTALL,
        )
        if user_match:
            return user_match.group(1).strip()

    return prompt


def recalibrate_scores(
    old_normalized: float,
    old_min: float,
    old_max: float,
) -> float:
    """Recalibrate a score from min-max normalization to simple ratio.

    Converts from (rating - old_min) / (old_max - old_min) to rating / old_max.

    Args:
        old_normalized: Score normalized via min-max to [0, 1]
        old_min: Original scale minimum (e.g. 1 for 1-10 scale)
        old_max: Original scale maximum (e.g. 10)

    Returns:
        Score as rating / old_max (e.g. 7/10 = 0.7)

    Example:
        recalibrate_scores(0.6667, old_min=1, old_max=10)  # rating=7, -> 0.7
        recalibrate_scores(0.5, old_min=1, old_max=5)      # rating=3, -> 0.6
    """
    # Recover original rating: old_normalized = (rating - old_min) / (old_max - old_min)
    rating = old_normalized * (old_max - old_min) + old_min
    return rating / old_max


def postprocess_request(
    req: dict,
    score_old_min: float = 1.0,
    score_old_max: float = 10.0,
) -> dict:
    """Apply post-processing fixes to a request.

    - Strip chat template from prompt (auto-detects format)
    - Recalibrate LLM judge scores from min-max normalization to rating/max
    """
    req["prompt"] = strip_chat_template(req["prompt"])

    for model_name, resp in req.get("models", {}).items():
        if "llm_judge_scores" in resp:
            resp["llm_judge_scores"] = {
                judge: recalibrate_scores(
                    score,
                    old_min=score_old_min,
                    old_max=score_old_max,
                )
                for judge, score in resp["llm_judge_scores"].items()
            }

    return req


def build_source_map(source_jsonl: str) -> dict:
    """Build prompt→source mapping from original JSONL file.

    The original JSONL (e.g. best-route-v3.jsonl) has clean prompts with
    a 'source' field indicating the dataset origin. We match against
    stripped prompts to recover this field.
    """
    # Sub-dataset → top-level dataset mapping
    # mix_instruct contains: itwgpt4, unified_chip2, sharegpt, dolly_15k, laion
    # reward_bench contains: xstest-*, refusals-*, donotanswer, hep-*
    SUB_TO_PARENT = {
        "itwgpt4": "mix_instruct",
        "unified_chip2": "mix_instruct",
        "sharegpt": "mix_instruct",
        "dolly_15k": "mix_instruct",
        "laion": "mix_instruct",
        "xstest-should-respond": "reward_bench",
        "xstest-should-refuse": "reward_bench",
        "donotanswer": "reward_bench",
        "refusals-dangerous": "reward_bench",
        "refusals-offensive": "reward_bench",
        "hep-python": "reward_bench",
        "hep-cpp": "reward_bench",
        "hep-java": "reward_bench",
        "hep-js": "reward_bench",
        "hep-go": "reward_bench",
        "hep-rust": "reward_bench",
    }

    source_map = {}
    with open(source_jsonl) as f:
        for line in f:
            item = json.loads(line)
            prompt = item.get("prompt", "").strip()
            source = item.get("source", item.get("dataset", ""))
            if prompt:
                # Strip index suffix (e.g. "code_ultra_feedback/6831" -> "code_ultra_feedback")
                name = source.split("/")[0] if source else ""
                # Map sub-datasets to top-level parent dataset
                source_map[prompt] = SUB_TO_PARENT.get(name, name)
    logger.info(f"Built source map: {len(source_map)} prompts from {source_jsonl}")
    return source_map


def main():
    parser = argparse.ArgumentParser(
        description="Split CARA training data into train/test sets"
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to processed training data JSON"
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.8,
        help="Fraction of data for training (default: 0.8)"
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory (default: same as input file)"
    )
    parser.add_argument(
        "--filter-truncated", action="store_true",
        help="Remove model responses where is_truncated=True (censored at max_tokens)"
    )
    parser.add_argument(
        "--score-old-min", type=float, default=1.0,
        help="Original score scale minimum used during normalization (default: 1.0 for 1-10 scale)"
    )
    parser.add_argument(
        "--score-old-max", type=float, default=10.0,
        help="Original score scale maximum used during normalization (default: 10.0 for 1-10 scale)"
    )
    parser.add_argument(
        "--skip-postprocess", action="store_true",
        help="Skip post-processing (chat template stripping, score recalibration)"
    )
    parser.add_argument(
        "--source-jsonl", default=None,
        help="Original JSONL file with 'source' field to recover dataset labels "
             "(e.g. data/cara/best-route-v3.jsonl)"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    with open(input_path) as f:
        data = json.load(f)

    requests = data["requests"]
    logger.info(f"Loaded {len(requests)} requests from {input_path}")

    # Post-process: strip chat templates and recalibrate scores
    if not args.skip_postprocess:
        for req in requests:
            postprocess_request(
                req,
                score_old_min=args.score_old_min,
                score_old_max=args.score_old_max,
            )
        logger.info(
            f"Post-processed: stripped chat templates, recalibrated judge scores "
            f"(from min-max [{args.score_old_min}-{args.score_old_max}] to rating/{args.score_old_max})"
        )

    # Recover dataset source labels if source JSONL provided
    if args.source_jsonl:
        source_map = build_source_map(args.source_jsonl)
        matched = 0
        for req in requests:
            prompt = req["prompt"].strip()
            if prompt in source_map:
                req["dataset"] = source_map[prompt]
                matched += 1
        logger.info(
            f"Source recovery: {matched}/{len(requests)} matched "
            f"({matched/len(requests)*100:.1f}%)"
        )
        if matched < len(requests):
            logger.warning(
                f"{len(requests) - matched} requests had no source match"
            )

    # Filter truncated responses if requested (require-all: drop entire request
    # if any model response is truncated, consistent with preprocessing pipeline)
    if args.filter_truncated:
        before = len(requests)
        filtered = []
        for req in requests:
            has_truncated = any(
                resp.get("is_truncated", False)
                for resp in req.get("models", {}).values()
            )
            if not has_truncated:
                filtered.append(req)
        requests = filtered
        logger.info(
            f"Filtered truncated: dropped {before - len(requests)}/{before} "
            f"requests ({(before - len(requests))/before*100:.1f}%) "
            f"where any model response hit max_tokens"
        )

    # Deduplicate by prompt content (keep first occurrence)
    seen_prompts = set()
    deduped = []
    for req in requests:
        p = req["prompt"]
        if p not in seen_prompts:
            seen_prompts.add(p)
            deduped.append(req)
    if len(deduped) < len(requests):
        logger.info(
            f"Deduplicated: removed {len(requests) - len(deduped)} duplicate prompts, "
            f"{len(deduped)} remaining"
        )
    requests = deduped

    # Split
    train_requests = []
    test_requests = []
    for req in requests:
        if hash_split(req["request_id"], args.train_ratio):
            train_requests.append(req)
        else:
            test_requests.append(req)

    logger.info(
        f"Split: {len(train_requests)} train ({len(train_requests)/len(requests)*100:.1f}%), "
        f"{len(test_requests)} test ({len(test_requests)/len(requests)*100:.1f}%)"
    )

    # Save as JSONL (one request per line, HuggingFace compatible)
    stem = input_path.stem
    train_path = output_dir / f"{stem}_train.jsonl"
    test_path = output_dir / f"{stem}_test.jsonl"

    for path, reqs, label in [
        (train_path, train_requests, "train"),
        (test_path, test_requests, "test")
    ]:
        with open(path, 'w') as f:
            for req in reqs:
                f.write(json.dumps(req) + '\n')
        logger.info(f"Saved {label}: {path} ({len(reqs)} requests)")


if __name__ == "__main__":
    main()
