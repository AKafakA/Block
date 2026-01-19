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
        DataFrame with columns ["id", "prompt", "response"].
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
    selected["response"] = selected["chosen"]  # Use chosen response for length filtering
    return selected[["id", "prompt", "response"]]


def load_code_ultra_feedback(sample_n: Optional[int] = None, seed: int = 1) -> pd.DataFrame:
    """Load CodeUltraFeedback and sample instructions as prompts.

    Returns DataFrame with ["id", "prompt", "response"].
    """
    path = datapath.CODE_ULTRA_FEEDBACK_PATH
    df = pd.read_parquet(path)
    sdf = df[["instruction", "responses"]].reset_index().copy()
    sdf["id"] = "code_ultra_feedback/" + sdf["index"].astype(str)
    # Use first response for length filtering
    sdf["response"] = sdf["responses"].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else "")
    sdf = sdf[["id", "instruction", "response"]].rename(columns={"instruction": "prompt"})
    if sample_n is not None and sample_n > 0:
        sdf = sdf.sample(n=sample_n, random_state=seed)
    return sdf


def load_mix_instruct(sample_n: Optional[int] = None, seed: int = 1) -> pd.DataFrame:
    """Load mix-instruct and form prompt as instruction + input.

    Returns DataFrame with ["id", "prompt", "response"].
    """
    path: str = datapath.MIX_INSTRUCT_TRAIN_PATH
    df = pd.read_json(path, lines=True)
    sdf = df[["id", "instruction", "input", "output"]].copy()
    # Concatenate with a space like the notebook
    sdf["prompt"] = sdf["instruction"].fillna("") + " " + sdf["input"].fillna("")
    sdf["response"] = sdf["output"].fillna("")
    sdf = sdf[["id", "prompt", "response"]]
    if sample_n is not None and sample_n > 0:
        sdf = sdf.sample(n=sample_n, random_state=seed)
    return sdf


def load_beaver_tails(sample_n: Optional[int] = None, seed: int = 1) -> pd.DataFrame:
    """Load BeaverTails harmful prompts only and sample.

    Returns DataFrame with ["id", "prompt", "response"].
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
    harmful = harmful[["prompt", "response"]].drop_duplicates(subset=["prompt"]).reset_index()
    harmful["id"] = "beaver_tails/" + harmful["index"].astype(str)
    harmful = harmful[["id", "prompt", "response"]]
    if sample_n is not None and sample_n > 0:
        harmful = harmful.sample(n=sample_n, random_state=seed)
    return harmful

def load_gsm8k(
    sample_n: Optional[int] = None,
    seed: int = 1,
    split: str = "train",
    config: str = "main",
) -> pd.DataFrame:
    """Load GSM8K math word problems; use 'question' as prompt.

    Returns DataFrame with ["id", "prompt", "response"].
    """
    if not _HAS_DATASETS:
        raise ImportError(
            "datasets is required for gsm8k loading. Install with: pip install datasets"
        )

    # gsm8k requires a config: 'main' or 'socratic'. Default to 'main'.
    # Use streaming=True to avoid PyArrow cache issues
    try:
        ds = load_dataset(datapath.GSM8K_DATASET_NAME, config, split=split, streaming=True)
    except Exception:
        # Fallback to 'socratic' if 'main' is unavailable in this mirror
        ds = load_dataset(datapath.GSM8K_DATASET_NAME, "socratic", split=split, streaming=True)
    records = []
    for idx, row in enumerate(ds):
        q = str(row.get("question", "")).strip()
        answer = str(row.get("answer", "")).strip()
        if not q:
            continue
        prompt = (
            f"Solve the following problem. Show your reasoning briefly and end with the final numeric answer.\n"
            f"Problem: {q}\n"
            f"Answer with only the final number on the last line."
        )
        records.append({"id": f"gsm8k/{idx}", "prompt": prompt, "response": answer})
    df = pd.DataFrame(records)
    if sample_n is not None and sample_n > 0 and len(df) > 0:
        df = df.sample(n=min(sample_n, len(df)), random_state=seed)
    return df

def load_squad(sample_n: Optional[int] = None, seed: int = 1, split: str = "train") -> pd.DataFrame:
    """Load SQuAD v1.1; build QA prompts with context.

    Returns DataFrame with ["id", "prompt", "response"].
    Note: SQuAD answers are short extracts, so response filtering has limited impact.
    """
    if not _HAS_DATASETS:
        raise ImportError("datasets is required for SQuAD. pip install datasets")

    # Use streaming=True to avoid PyArrow cache issues
    ds = load_dataset(datapath.SQUAD_DATASET_NAME, split=split, streaming=True)
    records = []
    for idx, row in enumerate(ds):
        q = str(row.get("question", "")).strip()
        ctx = str(row.get("context", "")).strip()
        answers = row.get("answers", {})
        answer_texts = answers.get("text", []) if isinstance(answers, dict) else []
        answer = answer_texts[0] if answer_texts else ""
        if not q:
            continue
        prompt = (
            "Answer the question based on the context.\n"
            f"Context: {ctx}\n"
            f"Question: {q}"
        )
        records.append({"id": f"squad/{idx}", "prompt": prompt, "response": answer})
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
    # Use streaming=True to avoid PyArrow cache issues
    try:
        ds = load_dataset(datapath.TRIVIA_QA_DATASET_NAME, "rc", split=split, streaming=True)
    except Exception:
        ds = load_dataset(datapath.TRIVIA_QA_DATASET_NAME, split=split, streaming=True)
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


def load_sharegpt_firstturn(sample_n: Optional[int] = None, seed: int = 1) -> pd.DataFrame:
    """Load ShareGPT-style dataset via HF and extract first human→assistant turn.

    Uses datasets library to stream. Returns DataFrame with ["id", "prompt", "response"].
    """
    if not _HAS_DATASETS:
        raise ImportError("datasets is required for ShareGPT loading. pip install datasets")

    # Try common splits; some repos only expose 'train'
    ds = None
    for split in ("train", "validation", "test"):
        try:
            ds = load_dataset(datapath.SHAREGPT_DATASET_NAME, split=split, streaming=True)
            break
        except Exception:
            ds = None
    if ds is None:
        # Last attempt: some datasets expose a default split name
        ds = load_dataset(datapath.SHAREGPT_DATASET_NAME, split="train", streaming=True)

    records = []
    for idx, row in enumerate(ds):
        conv = row.get("conversations", [])
        if not isinstance(conv, list) or len(conv) < 2:
            continue
        first = conv[0]
        second = conv[1]
        # Handle various key names
        role1 = first.get("from") or first.get("role")
        text1 = first.get("value") or first.get("content")
        role2 = second.get("from") or second.get("role")
        text2 = second.get("value") or second.get("content")
        if role1 and text1 and role2 and text2:
            # Normalize roles: expect human/user then gpt/assistant
            if role1.lower() in ("human", "user") and role2.lower() in ("gpt", "assistant"):
                records.append({
                    "id": f"sharegpt/{idx}",
                    "prompt": str(text1).strip(),
                    "response": str(text2).strip(),
                })
        if sample_n is not None and len(records) >= sample_n * 2:
            break

    df = pd.DataFrame(records)
    if sample_n is not None and sample_n > 0 and len(df) > 0:
        df = df.sample(n=min(sample_n, len(df)), random_state=seed)
    return df


def load_cnn_dailymail(sample_n: Optional[int] = None, seed: int = 1, split: str = "train") -> pd.DataFrame:
    """Load CNN/DailyMail; build summarization prompts from article.

    Returns DataFrame with ["id", "prompt", "response"].
    """
    if not _HAS_DATASETS:
        raise ImportError("datasets is required for CNN/DailyMail. pip install datasets")

    # Prefer newer config name. Use streaming=True to avoid PyArrow cache issues
    try:
        ds = load_dataset(datapath.CNN_DAILYMAIL_DATASET_NAME, "3.0.0", split=split, streaming=True)
    except Exception:
        ds = load_dataset(datapath.CNN_DAILYMAIL_DATASET_NAME, split=split, streaming=True)

    records = []
    for idx, row in enumerate(ds):
        art = str(row.get("article", "")).strip()
        highlights = str(row.get("highlights", "")).strip()
        if not art:
            continue
        prompt = f"Summarize the following article in 3-5 sentences.\nArticle: {art}"
        records.append({"id": f"cnn_dailymail/{idx}", "prompt": prompt, "response": highlights})
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
    # no local-path flags for online-only mode

    # Dataset selection and sizing
    allowed = [
        "reward_bench",
        "code_ultra_feedback",
        "beaver_tails",
        "mix_instruct",
        "sharegpt",
        "gsm8k",
        "squad",
        "cnn_dailymail",
        "ai2_arc",
    ]
    # Keep default datasets aligned with prior behavior (exclude sharegpt by default)
    default_datasets = [
        "reward_bench",
        "code_ultra_feedback",
        "beaver_tails",
        "mix_instruct",
        "sharegpt",
        "gsm8k",
        "squad",
        "cnn_dailymail",
        "ai2_arc",
    ]
    p.add_argument(
        "--datasets",
        nargs="+",
        choices=allowed,
        default=default_datasets,
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

    # Length filtering (applies to all datasets)
    # Parameters match vLLM's is_valid_sequence and benchmark_serving.py
    p.add_argument(
        "--max-prompt-len",
        type=int,
        default=1024,
        help="Maximum prompt token length (default: 1024). "
             "Matches vLLM's is_valid_sequence max_prompt_len.",
    )
    p.add_argument(
        "--min-prompt-len",
        type=int,
        default=4,
        help="Minimum prompt token length for inclusion (default: 4)",
    )
    p.add_argument(
        "--max-output-len",
        type=int,
        default=None,
        help="Maximum output/response token length. "
             "If specified, filters entries with longer responses. "
             "Maps to --custom-output-len in benchmark_serving.py.",
    )
    p.add_argument(
        "--max-total-len",
        type=int,
        default=2048,
        help="Maximum total token length (prompt + response, default: 2048). "
             "Matches vLLM's is_valid_sequence max_total_len. "
             "Filtering happens BEFORE sampling to ensure balanced ratios.",
    )
    p.add_argument(
        "--tokenizer",
        type=str,
        default="Qwen/Qwen2.5-3B",
        help="HF tokenizer name for length computation",
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
    """Load full datasets (after filtering), no sampling. Columns: id, prompt, response."""
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
        
        elif name == "gsm8k":
            df = load_gsm8k(sample_n=None, seed=seed)
        elif name == "squad":
            df = load_squad(sample_n=None, seed=seed)
        elif name == "cnn_dailymail":
            df = load_cnn_dailymail(sample_n=None, seed=seed)
        elif name == "sharegpt":
            df = load_sharegpt_firstturn(sample_n=None, seed=seed)
        elif name == "ai2_arc":
            # Combine ARC-Challenge and ARC-Easy train splits for sufficient capacity
            if not _HAS_DATASETS:
                raise ImportError("datasets is required for ai2_arc. pip install datasets")
            recs = []
            for subset in ("ARC-Challenge", "ARC-Easy"):
                # Use streaming=True to avoid PyArrow cache issues
                ds = load_dataset(datapath.AI2_ARC_DATASET_NAME, subset, split="train", streaming=True)
                for idx, row in enumerate(ds):
                    q = str(row.get("question", "")).strip()
                    choices = row.get("choices", {}) or {}
                    labels = choices.get("label", [])
                    texts = choices.get("text", [])
                    answer_key = str(row.get("answerKey", "")).strip()
                    options = [f"{lbl}. {txt}" for lbl, txt in zip(labels, texts)]
                    if not q or not options:
                        continue
                    prompt = (
                        f"Question: {q}\n"
                        f"Choices:\n- " + "\n- ".join(options) + "\n"
                        "Answer with only the single letter (A/B/C/D)."
                    )
                    # Response is just the answer key (very short)
                    recs.append({"id": f"ai2_arc/{subset}/{idx}", "prompt": prompt, "response": answer_key})
            df = pd.DataFrame(recs)
        else:
            raise SystemExit(f"Unknown dataset: {name}")
        loaded.append(df)
    return loaded


def _filter_df_by_token_length(
    df: pd.DataFrame,
    tokenizer: "AutoTokenizer",
    min_prompt_len: int = 4,
    max_prompt_len: int = 1024,
    max_output_len: Optional[int] = None,
    max_total_len: int = 2048,
    batch_size: int = 256,
) -> pd.DataFrame:
    """Filter DataFrame rows based on token lengths (matches vLLM's is_valid_sequence logic).

    Uses the tokenizer to count tokens without truncation; does not modify content.

    Args:
        df: DataFrame with 'prompt' column and optionally 'response' column
        tokenizer: HuggingFace tokenizer
        max_prompt_len: Maximum prompt token length (default: 1024)
        max_output_len: Maximum output/response token length (if None, skip this check)
        max_total_len: Maximum total token length prompt + response (default: 2048)
        batch_size: Batch size for tokenization
    """
    if df.empty:
        return df

    prompts = df["prompt"].tolist()
    has_response = "response" in df.columns
    responses = df["response"].tolist() if has_response else None

    keep = [False] * len(prompts)

    def get_lengths(texts: List[str]) -> List[int]:
        """Tokenize texts and return lengths."""
        try:
            enc = tokenizer(
                texts,
                add_special_tokens=False,
                padding=False,
                truncation=False,
                return_attention_mask=False,
            )
            return [len(ids) for ids in enc["input_ids"]]
        except Exception:
            return [len(tokenizer(x, add_special_tokens=False).input_ids) for x in texts]

    i = 0
    while i < len(prompts):
        batch_prompts = prompts[i:i + batch_size]
        prompt_lengths = get_lengths(batch_prompts)

        if has_response:
            batch_responses = responses[i:i + batch_size]
            # Handle None/empty responses
            batch_responses = [str(r) if r else "" for r in batch_responses]
            response_lengths = get_lengths(batch_responses)
        else:
            response_lengths = [0] * len(batch_prompts)

        for j, (p_len, r_len) in enumerate(zip(prompt_lengths, response_lengths)):
            # Match vLLM's is_valid_sequence logic
            prompt_ok = (p_len >= min_prompt_len) and (p_len <= max_prompt_len)
            output_ok = (max_output_len is None) or (r_len <= max_output_len)
            total_ok = (p_len + r_len) <= max_total_len
            keep[i + j] = prompt_ok and output_ok and total_ok

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


def _generate_output_path_with_constraints(
    base_path: str,
    max_prompt_len: int,
    max_output_len: Optional[int],
    max_total_len: int,
) -> str:
    """Generate output path with length constraints encoded in filename.

    Format: {base}-p{max_prompt}-o{max_output}-t{max_total}.{ext}
    Example: best-route-p1024-o1024-t2048.jsonl

    This allows benchmark_serving.py to auto-detect constraints from filename.
    """
    import os
    dir_name = os.path.dirname(base_path)
    base_name = os.path.basename(base_path)

    # Split extension
    if base_name.endswith('.jsonl'):
        name, ext = base_name[:-6], '.jsonl'
    elif base_name.endswith('.json'):
        name, ext = base_name[:-5], '.json'
    else:
        name, ext = base_name, ''

    # Build constraint suffix
    suffix_parts = [f"p{max_prompt_len}"]
    if max_output_len is not None:
        suffix_parts.append(f"o{max_output_len}")
    suffix_parts.append(f"t{max_total_len}")
    suffix = "-".join(suffix_parts)

    new_name = f"{name}-{suffix}{ext}"
    return os.path.join(dir_name, new_name) if dir_name else new_name


def main() -> None:
    args = parse_args()
    datasets: List[str] = list(args.datasets)

    if not _HAS_TRANSFORMERS:
        raise SystemExit("transformers not installed. Install with: pip install transformers")

    # Load tokenizer for prompt-length filtering
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    # Load all datasets fully (post-filtering) to know capacities
    full_dfs = _load_all_full(datasets, args.seed, args)
    # Apply length filters across all datasets (matches vLLM's is_valid_sequence)
    full_dfs = [
        _filter_df_by_token_length(
            df, tokenizer,
            min_prompt_len=args.min_prompt_len,
            max_prompt_len=args.max_prompt_len,
            max_output_len=args.max_output_len,
            max_total_len=args.max_total_len,
        )
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

    # Generate output path with length constraints encoded in filename
    output_path = _generate_output_path_with_constraints(
        args.output,
        max_prompt_len=args.max_prompt_len,
        max_output_len=args.max_output_len,
        max_total_len=args.max_total_len,
    )

    # Optional: write stats JSON
    if args.stats_output:
        stats_path = _generate_output_path_with_constraints(
            args.stats_output,
            max_prompt_len=args.max_prompt_len,
            max_output_len=args.max_output_len,
            max_total_len=args.max_total_len,
        )
        os.makedirs(os.path.dirname(stats_path) or ".", exist_ok=True)
        stats = {
            "total": total,
            "length_constraints": {
                "max_prompt_len": args.max_prompt_len,
                "max_output_len": args.max_output_len,
                "max_total_len": args.max_total_len,
            },
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
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"\nWrote stats: {stats_path}")

    records = mixed.to_dict(orient="records")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    if args.use_json:
        _write_json_array(output_path, records)
        print(f"Wrote JSON array: {output_path}")
    else:
        _write_jsonl(output_path, records)
        print(f"Wrote JSONL: {output_path}")

    print(f"\nLength constraints encoded in filename:")
    print(f"  max_prompt_len: {args.max_prompt_len}")
    if args.max_output_len:
        print(f"  max_output_len: {args.max_output_len}")
    print(f"  max_total_len: {args.max_total_len}")


if __name__ == "__main__":
    main()
