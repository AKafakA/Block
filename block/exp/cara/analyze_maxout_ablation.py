#!/usr/bin/env python3
"""Analyze max_output_tokens ablation results.

Checks per-dataset truncation rates and survival with --require-all-models.
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def get_dataset_category(source: str) -> str:
    """Map source field to one of 9 dataset categories."""
    prefix = source.split('/')[0]
    reward_bench_prefixes = ['xstest', 'hep', 'refusals', 'donotanswer']
    for rb in reward_bench_prefixes:
        if prefix.startswith(rb):
            return 'reward_bench'
    if prefix in ['dolly_15k', 'sharegpt', 'laion', 'itwgpt4', 'unified_chip2']:
        return 'mix_instruct'
    return prefix


def analyze_file(filepath: str, dataset_jsonl: str = None):
    """Analyze a single broadcast result file."""
    with open(filepath) as f:
        data = json.load(f)

    # Build prompt-to-category mapping if dataset provided
    prompt_to_cat = {}
    if dataset_jsonl:
        with open(dataset_jsonl) as f:
            for line in f:
                d = json.loads(line)
                prompt_to_cat[d['prompt'][:200]] = get_dataset_category(
                    d.get('source', 'unknown')
                )

    details = data.get('response_details', [])
    max_tokens = None  # detect from data

    # Per-category stats
    cat_total = Counter()
    cat_all_valid = Counter()
    cat_trunc = defaultdict(int)

    # Output length distribution for truncated
    all_output_lens = []
    trunc_output_lens = []

    for req in details:
        prompt = req.get('prompt', '')
        cat = prompt_to_cat.get(prompt[:200], 'unknown')
        cat_total[cat] += 1

        brs = req.get('broadcast_results', [])
        all_pass = True
        for br in brs:
            tokens = br.get('output_tokens', 0)
            success = br.get('success', False)
            all_output_lens.append(tokens)

            if not success or tokens < 3:
                all_pass = False
            elif max_tokens is None:
                # Detect max_tokens from the data (highest output_tokens value)
                pass

            # Detect truncation: output_tokens equals the max seen value
            # We'll compute this after scanning all data
            if tokens >= 1024:  # will refine below
                pass

        if all_pass:
            # Check truncation at detected max_tokens
            max_out = max(br.get('output_tokens', 0) for br in brs)
            if max_out < max(all_output_lens):
                cat_all_valid[cat] += 1
            else:
                cat_all_valid[cat] += 1  # will recompute

    # Detect actual max_tokens from data (the mode of high values)
    if all_output_lens:
        max_observed = max(all_output_lens)
        # Count values at max_observed
        at_max = sum(1 for x in all_output_lens if x == max_observed)
        if at_max > 5:
            max_tokens = max_observed
        else:
            max_tokens = max_observed

    # Recompute with detected max_tokens
    cat_total.clear()
    cat_all_valid.clear()
    cat_trunc.clear()
    trunc_per_model = defaultdict(int)

    for req in details:
        prompt = req.get('prompt', '')
        cat = prompt_to_cat.get(prompt[:200], 'unknown')
        cat_total[cat] += 1

        brs = req.get('broadcast_results', [])
        all_pass = True
        for br in brs:
            tokens = br.get('output_tokens', 0)
            model = br.get('model', 'unknown').split('/')[-1]
            success = br.get('success', False)

            if not success or tokens < 3 or tokens >= max_tokens:
                all_pass = False
                if tokens >= max_tokens:
                    cat_trunc[cat] += 1
                    trunc_per_model[model] += 1

        if all_pass:
            cat_all_valid[cat] += 1

    total = sum(cat_total.values())
    total_valid = sum(cat_all_valid.values())

    print(f"\nFile: {filepath}")
    print(f"Detected max_tokens: {max_tokens}")
    print(f"Total requests: {total}")
    print(f"All-4-valid requests: {total_valid} ({100*total_valid/total:.1f}%)")
    print()

    # Per-category breakdown
    header = f"{'Category':<22} {'Total':>5} {'4OK':>5} {'Rate':>6} {'Trunc':>6}"
    print(header)
    print('-' * len(header))
    for cat in sorted(cat_total.keys()):
        t = cat_total[cat]
        v = cat_all_valid.get(cat, 0)
        tr = cat_trunc.get(cat, 0)
        print(f"{cat:<22} {t:>5} {v:>5} {100*v/t:>5.1f}% {tr:>6}")

    print()
    print("Truncation per model:")
    for model, c in sorted(trunc_per_model.items(), key=lambda x: -x[1]):
        print(f"  {model}: {c}")

    # Output length percentiles
    if all_output_lens:
        all_output_lens.sort()
        n = len(all_output_lens)
        for p in [50, 75, 90, 95, 99, 100]:
            idx = min(int(n * p / 100), n - 1)
            print(f"  p{p} output_len: {all_output_lens[idx]}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_maxout_ablation.py <result.json> [dataset.jsonl]")
        sys.exit(1)

    filepath = sys.argv[1]
    dataset = sys.argv[2] if len(sys.argv) > 2 else None
    analyze_file(filepath, dataset)
