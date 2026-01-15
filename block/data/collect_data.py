import argparse
import json
import math
import os
from typing import Iterable, List, Optional, Sequence

import pandas as pd

from block.data import datapath

try:
    # Optional: We prefer using datasets for flexible HF loading of new sources.
    from datasets import load_dataset
    _HAS_DATASETS = True
except Exception:
    _HAS_DATASETS = False

try:
    from transformers import AutoTokenizer
    _HAS_TRANSFORMERS = True
except Exception:
    _HAS_TRANSFORMERS = False


def load_reward_bench(
    split: str = "filtered",
    include_prefixes: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Load reward-bench from HF and select subsets by prefix.

    Args:
        split: One of {"raw", "filtered"}
        include_prefixes: Subset name prefixes to keep. If None, uses
            ["xstest-", "refusals-", "donotanswer", "hep-"]

    Returns:
        DataFrame with columns ["id", "prompt"].
    """
    if include_prefixes is None:
        include_prefixes = datapath.REWARD_BENCH_DEFAULT_PREFIXES

    if split not in datapath.REWARD_BENCH_SPLITS:
        raise ValueError(f"Invalid reward-bench split: {split}")

    path = datapath.REWARD_BENCH_BASE + datapath.REWARD_BENCH_SPLITS[split]
    df = pd.read_parquet(path)

    mask = False
    for p in include_prefixes:
        mask = mask | df["subset"].str.startswith(p)
    selected = df.loc[mask].copy()
    selected["id"] = selected["subset"] + "/" + selected["id"].astype(str)
    return selected[["id", "prompt"]]


def load_code_ultra_feedback(sample_n: Optional[int] = None, seed: int = 1) -> pd.DataFrame:
    """Load CodeUltraFeedback and sample instructions as prompts.

    Returns DataFrame with ["id", "prompt"].
    """
    path = datapath.CODE_ULTRA_FEEDBACK_PATH
    df = pd.read_parquet(path)
    sdf = df[["instruction"]].reset_index().copy()
    sdf["id"] = "code_ultra_feedback/" + sdf["index"].astype(str)
    sdf = sdf[["id", "instruction"]].rename(columns={"instruction": "prompt"})
    if sample_n is not None and sample_n > 0:
        sdf = sdf.sample(n=sample_n, random_state=seed)
    return sdf


def load_mix_instruct(sample_n: Optional[int] = None, seed: int = 1) -> pd.DataFrame:
    """Load mix-instruct and form prompt as instruction + input.

    Returns DataFrame with ["id", "prompt"].
    """
    path: str = datapath.MIX_INSTRUCT_TRAIN_PATH
    df = pd.read_json(path, lines=True)
    sdf = df[["id", "instruction", "input"]].copy()
    # Concatenate with a space like the notebook
    sdf["prompt"] = sdf["instruction"].fillna("") + " " + sdf["input"].fillna("")
    sdf = sdf[["id", "prompt"]]
    if sample_n is not None and sample_n > 0:
        sdf = sdf.sample(n=sample_n, random_state=seed)
    return sdf


def load_beaver_tails(sample_n: Optional[int] = None, seed: int = 1) -> pd.DataFrame:
    """Load BeaverTails harmful prompts only and sample.

    Returns DataFrame with ["id", "prompt"].
    """
    path: str = datapath.BEAVER_TAILS_30K_TRAIN_PATH
    df = pd.read_json(path, lines=True)

    def is_prompt_harmful(category: dict) -> bool:
        # True if any category flag is True
        for _, v in category.items():
            if v:
                return True
        return False

    harmful = df[df["category"].apply(is_prompt_harmful)].copy()
    harmful = harmful[["prompt"]].drop_duplicates().reset_index()
    harmful["id"] = "beaver_tails/" + harmful["index"].astype(str)
    harmful = harmful[["id", "prompt"]]
    if sample_n is not None and sample_n > 0:
        harmful = harmful.sample(n=sample_n, random_state=seed)
    return harmful


def load_ultrachat(sample_n: Optional[int] = None, seed: int = 1, split: str = "train_sft") -> pd.DataFrame:
    """Load UltraChat (benign chats). Extract the first user turn as prompt.

    Returns DataFrame with ["id", "prompt"]. Uses datasets library.
    """
    if not _HAS_DATASETS:
        raise ImportError(
            "datasets is required for ultrachat loading. Install with: pip install datasets"
        )

    ds = load_dataset(datapath.ULTRACHAT_DATASET_NAME, split=split)

    prompts = []
    # Common patterns: columns may include 'messages' (list of {role, content}) or 'conversations'
    for idx, row in enumerate(ds):
        prompt_text = None
        if "messages" in row and isinstance(row["messages"], list):
            # Find first user message
            for m in row["messages"]:
                role = m.get("role") if isinstance(m, dict) else None
                content = m.get("content") if isinstance(m, dict) else None
                if role and role.lower() == "user" and content:
                    prompt_text = content
                    break
        elif "conversations" in row and isinstance(row["conversations"], list):
            for m in row["conversations"]:
                role = m.get("from") if isinstance(m, dict) else None
                content = m.get("value") if isinstance(m, dict) else None
                if role and role.lower() == "human" and content:
                    prompt_text = content
                    break
        elif "instruction" in row:
            # Fallback: instruction + optional input
            prompt_text = str(row.get("instruction", ""))
            if row.get("input"):
                prompt_text = (prompt_text + " " + str(row["input"])).strip()

        if prompt_text:
            prompts.append({
                "id": f"ultrachat/{idx}",
                "prompt": prompt_text,
            })

    df = pd.DataFrame(prompts)
    if sample_n is not None and sample_n > 0 and len(df) > 0:
        df = df.sample(n=min(sample_n, len(df)), random_state=seed)
    return df


def load_gsm8k(
    sample_n: Optional[int] = None,
    seed: int = 1,
    split: str = "train",
    config: str = "main",
) -> pd.DataFrame:
    """Load GSM8K math word problems; use 'question' as prompt.

    Returns DataFrame with ["id", "prompt"].
    """
    if not _HAS_DATASETS:
        raise ImportError(
            "datasets is required for gsm8k loading. Install with: pip install datasets"
        )

    # gsm8k requires a config: 'main' or 'socratic'. Default to 'main'.
    try:
        ds = load_dataset(datapath.GSM8K_DATASET_NAME, config, split=split)
    except Exception:
        # Fallback to 'socratic' if 'main' is unavailable in this mirror
        ds = load_dataset(datapath.GSM8K_DATASET_NAME, "socratic", split=split)
    records = []
    for idx, row in enumerate(ds):
        q = str(row.get("question", "")).strip()
        if not q:
            continue
        prompt = (
            f"Solve the following problem. Show your reasoning briefly and end with the final numeric answer.\n"
            f"Problem: {q}\n"
            f"Answer with only the final number on the last line."
        )
        records.append({"id": f"gsm8k/{idx}", "prompt": prompt})
    df = pd.DataFrame(records)
    if sample_n is not None and sample_n > 0 and len(df) > 0:
        df = df.sample(n=min(sample_n, len(df)), random_state=seed)
    return df

def load_squad(sample_n: Optional[int] = None, seed: int = 1, split: str = "train") -> pd.DataFrame:
    """Load SQuAD v1.1; build QA prompts with context.

    Returns DataFrame with ["id", "prompt"].
    """
    if not _HAS_DATASETS:
        raise ImportError("datasets is required for SQuAD. pip install datasets")

    ds = load_dataset(datapath.SQUAD_DATASET_NAME, split=split)
    records = []
    for idx, row in enumerate(ds):
        q = str(row.get("question", "")).strip()
        ctx = str(row.get("context", "")).strip()
        if not q:
            continue
        prompt = (
            "Answer the question based on the context.\n"
            f"Context: {ctx}\n"
            f"Question: {q}"
        )
        records.append({"id": f"squad/{idx}", "prompt": prompt})
    df = pd.DataFrame(records)
    if sample_n is not None and sample_n > 0 and len(df) > 0:
        df = df.sample(n=min(sample_n, len(df)), random_state=seed)
    return df


def load_trivia_qa(sample_n: Optional[int] = None, seed: int = 1, split: str = "train") -> pd.DataFrame:
    """Load TriviaQA; use question as prompt (open-domain).

    Returns DataFrame with ["id", "prompt"].
    """
    if not _HAS_DATASETS:
        raise ImportError("datasets is required for TriviaQA. pip install datasets")

    # Use 'rc' subset for reading-comprehension formatted version when available
    try:
        ds = load_dataset(datapath.TRIVIA_QA_DATASET_NAME, "rc", split=split)
    except Exception:
        ds = load_dataset(datapath.TRIVIA_QA_DATASET_NAME, split=split)
    records = []
    for idx, row in enumerate(ds):
        q = str(row.get("question", "")).strip()
        if not q:
            continue
        records.append({"id": f"trivia_qa/{idx}", "prompt": q})
    df = pd.DataFrame(records)
    if sample_n is not None and sample_n > 0 and len(df) > 0:
        df = df.sample(n=min(sample_n, len(df)), random_state=seed)
    return df


def load_cnn_dailymail(sample_n: Optional[int] = None, seed: int = 1, split: str = "train") -> pd.DataFrame:
    """Load CNN/DailyMail; build summarization prompts from article.

    Returns DataFrame with ["id", "prompt"].
    """
    if not _HAS_DATASETS:
        raise ImportError("datasets is required for CNN/DailyMail. pip install datasets")

    # Prefer newer config name
    try:
        ds = load_dataset(datapath.CNN_DAILYMAIL_DATASET_NAME, "3.0.0", split=split)
    except Exception:
        ds = load_dataset(datapath.CNN_DAILYMAIL_DATASET_NAME, split=split)

    records = []
    for idx, row in enumerate(ds):
        art = str(row.get("article", "")).strip()
        if not art:
            continue
        prompt = f"Summarize the following article in 3-5 sentences.\nArticle: {art}"
        records.append({"id": f"cnn_dailymail/{idx}", "prompt": prompt})
    df = pd.DataFrame(records)
    if sample_n is not None and sample_n > 0 and len(df) > 0:
        df = df.sample(n=min(sample_n, len(df)), random_state=seed)
    return df

def _balanced_counts(total: int, weights: Sequence[float]) -> List[int]:
    """Round weights to integer counts that sum to total.

    Uses largest remainder method on normalized weights.
    """
    if total < 0:
        raise ValueError("total must be non-negative")
    if not weights:
        return []
    s = sum(w for w in weights)
    if s <= 0:
        # uniform distribution if all weights are zero or negative
        weights = [1.0] * len(weights)
        s = float(len(weights))
    norm = [w / s for w in weights]
    raw = [total * w for w in norm]
    base = [int(math.floor(x)) for x in raw]
    remainder = total - sum(base)
    fracs = [(i, raw[i] - base[i]) for i in range(len(raw))]
    fracs.sort(key=lambda t: t[1], reverse=True)
    for i in range(remainder):
        base[fracs[i][0]] += 1
    return base


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect and mix datasets for routing.")
    # General
    p.add_argument(
        "-o",
        "--output",
        default="data/best_route.jsonl",
        help="Output path. Defaults to JSONL: data/mixed_dataset.jsonl",
    )
    p.add_argument(
        "--use-json",
        action="store_true",
        help="Write a JSON array instead of JSONL (default: JSONL)",
    )
    p.add_argument("--seed", type=int, default=1, help="Random seed (default: 1)")
    p.add_argument(
        "--index-start",
        type=int,
        default=0,
        help="Starting index for assigned ids (default: 0)",
    )
    p.add_argument(
        "--stats-output",
        type=str,
        default=None,
        help="Optional path to write a JSON summary of dataset distribution",
    )

    # Dataset selection and sizing
    allowed = [
        "reward_bench",
        "code_ultra_feedback",
        "beaver_tails",
        "mix_instruct",
        "ultrachat",
        "gsm8k",
        "squad",
        "cnn_dailymail",
        "ai2_arc",
    ]
    p.add_argument(
        "--datasets",
        nargs="+",
        choices=allowed,
        default=allowed,
        help="Datasets to include (space-separated)",
    )
    p.add_argument(
        "-n",
        "--total-n",
        type=int,
        default=10000,
        help=(
            "Total number of samples across all datasets. "
            "Counts per dataset are derived using --ratios."
        ),
    )
    p.add_argument(
        "--ratios",
        type=float,
        nargs="+",
        help=(
            "Relative ratios per dataset (same order as --datasets). "
            "If omitted, uses equal ratios."
        ),
    )
    p.add_argument(
        "--min-per",
        type=int,
        nargs="*",
        help=(
            "Minimum samples per dataset (same order as --datasets). "
            "Defaults to none. If provided, length must match --datasets."
        ),
    )

    # reward-bench filters
    p.add_argument(
        "--reward-bench-split",
        choices=list(datapath.REWARD_BENCH_SPLITS.keys()),
        default="filtered",
        help="reward-bench split to use (default: filtered)",
    )
    p.add_argument(
        "--reward-bench-prefix",
        action="append",
        dest="reward_bench_prefixes",
        help=(
            "Subset prefix to include (repeatable). Default prefixes are "
            + ", ".join(datapath.REWARD_BENCH_DEFAULT_PREFIXES)
        ),
    )

    # Prompt length filtering (applies to all datasets)
    p.add_argument(
        "--max-prompt-length",
        type=int,
        default=700,
        help="Maximum prompt token length; discard prompts exceeding this",
    )
    p.add_argument(
        "--tokenizer",
        type=str,
        default="Qwen/Qwen2.5-3B",
        help="HF tokenizer name for prompt length computation",
    )

    return p.parse_args()


def _assign_counts(
    datasets: List[str],
    total_n: int,
    ratios: Optional[List[float]],
    min_per: Optional[List[int]],
    capacities: List[int],
) -> List[int]:
    """Compute final per-dataset counts with floors and capacity-aware reallocation.

    Steps:
    - Base allocation from ratios summing to total_n via largest remainder.
    - Apply floors: target_i = max(base_i, min_per[i])
    - Let target_total = sum(target_i)
    - Cap by capacity: assign_i = min(target_i, capacity_i)
    - Reallocate deficits from capped datasets proportionally to ratios across
      datasets with spare capacity until no deficits or no spare.

    Returns final counts. Sum equals min(target_total, sum(capacities)).
    """
    k = len(datasets)
    if ratios is None:
        ratios = [1.0] * k
    if len(ratios) != k:
        raise SystemExit("--ratios must have the same length as --datasets")
    if min_per is None:
        min_per = [0] * k
    if len(min_per) not in (0, k):
        raise SystemExit("--min-per must be empty or match length of --datasets")
    if len(min_per) == 0:
        min_per = [0] * k

    base = _balanced_counts(total_n, ratios)
    target = [max(base[i], int(min_per[i])) for i in range(k)]

    assigned = [min(target[i], int(capacities[i])) for i in range(k)]

    # Total deficit to reallocate from capped datasets
    remaining_deficit = sum(max(0, target[i] - assigned[i]) for i in range(k))

    # Reallocate until deficits are covered or no spare remains
    while remaining_deficit > 0:
        # Eligible indices with spare capacity
        eligible = [i for i in range(k) if capacities[i] - assigned[i] > 0]
        if not eligible:
            break

        # Compute total spare among eligible
        total_spare = sum(int(capacities[i]) - assigned[i] for i in eligible)
        if total_spare <= 0:
            break

        # Weights among eligible
        w_sum = sum(ratios[i] for i in eligible)
        if w_sum <= 0:
            weights = [1.0] * len(eligible)
        else:
            weights = [ratios[i] for i in eligible]

        # Allocate the remaining deficit (capped by spare) by largest remainder
        to_allocate = min(remaining_deficit, total_spare)
        adds = _balanced_counts(to_allocate, weights)

        # Cap adds by spare and apply
        applied_total = 0
        for idx, inc in zip(eligible, adds):
            inc_cap = min(inc, int(capacities[idx]) - assigned[idx])
            if inc_cap > 0:
                assigned[idx] += inc_cap
                applied_total += inc_cap

        # If nothing applied (e.g., all eligible were at cap), stop
        if applied_total == 0:
            break

        remaining_deficit -= applied_total

    return assigned


def _load_all_full(datasets: List[str], seed: int, args) -> List[pd.DataFrame]:
    """Load full datasets (after filtering), no sampling. Columns: id, prompt."""
    loaded: List[pd.DataFrame] = []
    for name in datasets:
        if name == "reward_bench":
            df = load_reward_bench(
                split=args.reward_bench_split,
                include_prefixes=(
                    args.reward_bench_prefixes
                    if args.reward_bench_prefixes and len(args.reward_bench_prefixes) > 0
                    else None
                ),
            )
        elif name == "code_ultra_feedback":
            df = load_code_ultra_feedback(sample_n=None, seed=seed)
        elif name == "beaver_tails":
            df = load_beaver_tails(sample_n=None, seed=seed)
        elif name == "mix_instruct":
            df = load_mix_instruct(sample_n=None, seed=seed)
        elif name == "ultrachat":
            df = load_ultrachat(sample_n=None, seed=seed)
        elif name == "gsm8k":
            df = load_gsm8k(sample_n=None, seed=seed)
        elif name == "squad":
            df = load_squad(sample_n=None, seed=seed)
        elif name == "cnn_dailymail":
            df = load_cnn_dailymail(sample_n=None, seed=seed)
        elif name == "ai2_arc":
            # Combine ARC-Challenge and ARC-Easy train splits for sufficient capacity
            if not _HAS_DATASETS:
                raise ImportError("datasets is required for ai2_arc. pip install datasets")
            recs = []
            for subset in ("ARC-Challenge", "ARC-Easy"):
                ds = load_dataset(datapath.AI2_ARC_DATASET_NAME, subset, split="train")
                for idx, row in enumerate(ds):
                    q = str(row.get("question", "")).strip()
                    choices = row.get("choices", {}) or {}
                    labels = choices.get("label", [])
                    texts = choices.get("text", [])
                    options = [f"{lbl}. {txt}" for lbl, txt in zip(labels, texts)]
                    if not q or not options:
                        continue
                    prompt = (
                        f"Question: {q}\n"
                        f"Choices:\n- " + "\n- ".join(options) + "\n"
                        "Answer with only the single letter (A/B/C/D)."
                    )
                    recs.append({"id": f"ai2_arc/{subset}/{idx}", "prompt": prompt})
            df = pd.DataFrame(recs)
        else:
            raise SystemExit(f"Unknown dataset: {name}")
        loaded.append(df)
    return loaded


def _filter_df_by_token_length(
    df: pd.DataFrame,
    tokenizer: "AutoTokenizer",
    max_len: int,
    batch_size: int = 256,
) -> pd.DataFrame:
    """Filter DataFrame rows where prompt token length <= max_len.

    Uses the tokenizer to count tokens without truncation; does not modify prompts.
    """
    if df.empty:
        return df
    prompts = df["prompt"].tolist()
    keep = [False] * len(prompts)
    i = 0
    while i < len(prompts):
        batch = prompts[i:i + batch_size]
        try:
            enc = tokenizer(
                batch,
                add_special_tokens=False,
                padding=False,
                truncation=False,
                return_attention_mask=False,
            )
            input_ids = enc["input_ids"]
            lengths = [len(ids) for ids in input_ids]
        except Exception:
            lengths = [len(tokenizer(x, add_special_tokens=False).input_ids) for x in batch]
        for j, L in enumerate(lengths):
            keep[i + j] = (L <= max_len)
        i += batch_size
    return df.loc[keep].reset_index(drop=True)


def _write_json_array(path: str, records: Iterable[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(list(records), f, ensure_ascii=False)


def _write_jsonl(path: str, records: Iterable[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    datasets: List[str] = list(args.datasets)

    if not _HAS_TRANSFORMERS:
        raise SystemExit("transformers not installed. Install with: pip install transformers")

    # Load tokenizer for prompt-length filtering
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    # Load all datasets fully (post-filtering) to know capacities
    full_dfs = _load_all_full(datasets, args.seed, args)
    # Apply prompt length filter across all datasets
    full_dfs = [
        _filter_df_by_token_length(df, tokenizer, args.max_prompt_length)
        for df in full_dfs
    ]
    capacities = [len(df) for df in full_dfs]

    counts = _assign_counts(
        datasets=datasets,
        total_n=args.total_n,
        ratios=args.ratios,
        min_per=(args.min_per if args.min_per is not None and len(args.min_per) > 0 else None),
        capacities=capacities,
    )

    # Sample per dataset according to final counts
    parts: List[pd.DataFrame] = []
    assigned_counts = {}
    for name, df, n, cap in zip(datasets, full_dfs, counts, capacities):
        if n <= 0 or cap <= 0:
            print(f"{name}: assigned 0 (empty dataset)")
            continue
        df = df.rename(columns={"id": "source"})
        if n < cap:
            sampled = df.sample(n=n, random_state=args.seed)
        else:
            sampled = df
        parts.append(sampled[["source", "prompt"]])
        assigned_counts[name] = len(sampled)
        print(f"{name}: assigned {len(sampled)} (capacity {cap})")

    if not parts:
        raise SystemExit("No records sampled; check inputs and filters.")

    mixed = pd.concat(parts, ignore_index=True)
    mixed.insert(0, "id", range(args.index_start, args.index_start + len(mixed)))

    total = len(mixed)
    print(f"Total records: {total}")

    # Final distribution summary
    print("\nFinal distribution by dataset:")
    print(f"{'Dataset':<20} {'Count':>8} {'Percent':>10}")
    print("-" * 40)
    for name in datasets:
        cnt = assigned_counts.get(name, 0)
        pct = (100.0 * cnt / total) if total > 0 else 0.0
        print(f"{name:<20} {cnt:>8} {pct:>9.2f}%")

    # Optional: write stats JSON
    if args.stats_output:
        os.makedirs(os.path.dirname(args.stats_output), exist_ok=True)
        stats = {
            "total": total,
            "max_prompt_length": args.max_prompt_length,
            "tokenizer": args.tokenizer,
            "datasets": [
                {
                    "name": name,
                    "capacity": cap,
                    "assigned": assigned_counts.get(name, 0),
                    "ratio": 1.0 if args.ratios is None else float(args.ratios[datasets.index(name)])
                }
                for name, cap in zip(datasets, capacities)
            ],
        }
        with open(args.stats_output, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"\nWrote stats: {args.stats_output}")
    records = mixed.to_dict(orient="records")
    if args.use_json:
        _write_json_array(args.output, records)
        print(f"Wrote JSON array: {args.output}")
    else:
        _write_jsonl(args.output, records)
        print(f"Wrote JSONL: {args.output}")


if __name__ == "__main__":
    main()
