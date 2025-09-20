#!/usr/bin/env python3
"""Aggregate request-level metrics for one or more scheduler outputs.

Usage:
    PYTHONPATH=. simulation_analysis/summarize_request_metrics.py \
        experiment_output/offline/qps32_full --output simulation_analysis/qps32_summary.json

The script discovers scheduler subdirectories, loads the most recent run for
that scheduler, and reports aggregate statistics (mean, p50, p90, p99) for key
columns. It also captures summary counts (number of requests, restarts, etc.).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

SUMMARY_COLUMNS = {
    # Highlighted headline metrics first
    "request_e2e_time": "E2E latency (s)",
    "prefill_e2e_time": "TTFT (s)",
    # Prefer the clearer alias if present; fall back handled below
    "request_waiting_time": "Waiting time (s)",
    # Supporting metrics
    "request_execution_time": "Execution time (s)",
    "request_execution_plus_preemption_time": "Execution+preemption (s)",
}

PERCENTILES = [50, 90, 95, 99]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        type=Path,
        help="Directory containing per-scheduler subfolders.",
    )
    parser.add_argument(
        "--schedulers",
        nargs="*",
        help="Optional subset of schedulers to include (defaults to all directories under root).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("simulation_analysis/request_metric_summary.json"),
        help="Path to write the JSON summary.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional path to write a tabular CSV summary.",
    )
    return parser.parse_args()


def discover_schedulers(root: Path, allowlist: Iterable[str] | None) -> List[str]:
    if allowlist:
        return sorted(set(name.lower() for name in allowlist))
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def latest_run_dir(scheduler_root: Path) -> Path:
    runs = [p for p in scheduler_root.iterdir() if p.is_dir()]
    if not runs:
        raise FileNotFoundError(f"No runs found under {scheduler_root}")
    return max(runs, key=lambda path: path.name)


def summarize_scheduler(root: Path, scheduler: str) -> Dict[str, object]:
    scheduler_root = root / scheduler
    if not scheduler_root.exists():
        raise FileNotFoundError(f"Missing scheduler directory {scheduler_root}")

    run_dir = latest_run_dir(scheduler_root)
    csv_path = run_dir / "request_metrics.csv"
    if not csv_path.exists():
        return {
            "scheduler": scheduler,
            "run_dir": str(run_dir.resolve()),
            "num_requests": 0,
            "num_restarts": 0,
            "metrics": {},
            "warning": f"Missing request_metrics.csv at {csv_path}",
        }

    df = pd.read_csv(csv_path)
    df = df.replace({np.inf: np.nan, -np.inf: np.nan})

    summary: Dict[str, object] = {
        "scheduler": scheduler,
        "run_dir": str(run_dir.resolve()),
        "num_requests": int(df.shape[0]),
        "num_restarts": int(df.get("request_num_restarts", pd.Series(dtype=float)).sum()),
    }

    stats: Dict[str, Dict[str, float]] = {}
    for column, friendly in SUMMARY_COLUMNS.items():
        if column not in df.columns:
            # Backward-compat: map waiting time to old scheduling delay name
            if column == "request_waiting_time" and "request_scheduling_delay" in df.columns:
                column = "request_scheduling_delay"
            # Backward-compat: map TTFT to prefill_e2e_time if present (already preferred)
            # No alternate column name needed here; skip if missing.
            else:
                continue
        series = df[column].astype(float)
        series = series.dropna()
        if series.empty:
            continue
        percentiles = np.percentile(series, PERCENTILES)
        stats[friendly] = {
            "mean": float(series.mean()),
            "min": float(series.min()),
            "max": float(series.max()),
            **{f"p{p}": float(percentiles[idx]) for idx, p in enumerate(PERCENTILES)},
        }
    summary["metrics"] = stats

    return summary


def main() -> None:
    args = parse_args()
    schedulers = discover_schedulers(args.root, args.schedulers)
    results = []
    for scheduler in schedulers:
        try:
            results.append(summarize_scheduler(args.root, scheduler))
        except FileNotFoundError as exc:
            results.append(
                {
                    "scheduler": scheduler,
                    "run_dir": None,
                    "num_requests": 0,
                    "num_restarts": 0,
                    "metrics": {},
                    "warning": str(exc),
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))

    if args.csv:
        rows = []
        for res in results:
            scheduler = res["scheduler"]
            for metric_name, values in res.get("metrics", {}).items():
                row = {"scheduler": scheduler, "metric": metric_name}
                row.update(values)
                rows.append(row)
        if rows:
            df = pd.DataFrame(rows)
            args.csv.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(args.csv, index=False)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
