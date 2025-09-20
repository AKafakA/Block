#!/usr/bin/env python3
"""Utility to compare Block baseline vs optimized offline simulation outputs.

Given two output roots (e.g., simulation_analysis/full_runs/baseline_seq and
simulation_analysis/full_runs/optimized_fast), this script locates the most
recent run per scheduler, checks that the emitted request metrics match exactly,
and emits a JSON summary with runtime metadata extracted from the driver logs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "baseline_root",
        type=Path,
        help="Output directory for the sequential baseline (fast_predict off).",
    )
    parser.add_argument(
        "optimized_root",
        type=Path,
        help="Output directory for the optimized run (fast_predict on).",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing run.log files for each setup. If omitted, the script "
            "looks for run.log in the provided roots."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("simulation_analysis/block_run_comparison.json"),
        help="Where to write the JSON summary.",
    )
    return parser.parse_args()


def latest_run_dir(root: Path, scheduler: str) -> Path:
    candidate_root = root / scheduler
    if not candidate_root.exists():
        raise FileNotFoundError(f"Missing scheduler directory: {candidate_root}")
    runs = [p for p in candidate_root.iterdir() if p.is_dir()]
    if not runs:
        raise FileNotFoundError(f"No completed runs found under {candidate_root}")
    return max(runs, key=lambda path: path.name)


def load_metrics(path: Path) -> pd.DataFrame:
    csv_path = path / "request_metrics.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing request_metrics.csv at {csv_path}")
    df = pd.read_csv(csv_path)
    return df.sort_values("request_id").reset_index(drop=True)


def compare_frames(baseline: pd.DataFrame, optimized: pd.DataFrame) -> Dict[str, int]:
    summary: Dict[str, int] = {
        "row_count": len(baseline),
        "column_count": baseline.shape[1],
        "mismatched_cells": 0,
    }
    if baseline.shape != optimized.shape:
        summary["mismatched_cells"] = -1
        return summary

    diff = (baseline != optimized).to_numpy()
    summary["mismatched_cells"] = int(diff.sum())
    return summary


def extract_runtime(log_path: Path) -> Optional[float]:
    if not log_path.exists():
        return None
    for line in reversed(log_path.read_text().splitlines()):
        if "Simulation took:" in line:
            try:
                return float(line.split("Simulation took:")[1].split("s")[0])
            except (IndexError, ValueError):
                return None
    return None


def main() -> None:
    args = parse_args()
    schedulers = ["block_offline", "block_star_offline"]
    comparison: Dict[str, Dict[str, int]] = {}

    for scheduler in schedulers:
        baseline_dir = latest_run_dir(args.baseline_root, scheduler)
        optimized_dir = latest_run_dir(args.optimized_root, scheduler)
        baseline_metrics = load_metrics(baseline_dir)
        optimized_metrics = load_metrics(optimized_dir)
        comparison[scheduler] = compare_frames(baseline_metrics, optimized_metrics)

    log_root = args.log_dir
    if log_root is None:
        log_root = args.baseline_root.parent

    baseline_log = (log_root / "baseline_seq" / "run.log").resolve()
    optimized_log = (log_root / "optimized_fast" / "run.log").resolve()

    summary = {
        "baseline_root": str(args.baseline_root.resolve()),
        "optimized_root": str(args.optimized_root.resolve()),
        "comparisons": comparison,
        "baseline_runtime_seconds": extract_runtime(baseline_log),
        "optimized_runtime_seconds": extract_runtime(optimized_log),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
