import argparse
import json
import math
import os
from typing import Iterable, List, Optional, Sequence

import pandas as pd

from block.data import datapath


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

    # Dataset selection and sizing
    allowed = ["reward_bench", "code_ultra_feedback", "beaver_tails", "mix_instruct"]
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
        else:
            raise SystemExit(f"Unknown dataset: {name}")
        loaded.append(df)
    return loaded


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

    # Load all datasets fully (post-filtering) to know capacities
    full_dfs = _load_all_full(datasets, args.seed, args)
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
        print(f"{name}: assigned {len(sampled)} (capacity {cap})")

    if not parts:
        raise SystemExit("No records sampled; check inputs and filters.")

    mixed = pd.concat(parts, ignore_index=True)
    mixed.insert(0, "id", range(args.index_start, args.index_start + len(mixed)))

    print(f"Total records: {len(mixed)}")
    records = mixed.to_dict(orient="records")
    if args.use_json:
        _write_json_array(args.output, records)
        print(f"Wrote JSON array: {args.output}")
    else:
        _write_jsonl(args.output, records)
        print(f"Wrote JSONL: {args.output}")


if __name__ == "__main__":
    main()
