#!/usr/bin/env python3
"""Run a suite of offline simulations and produce summary reports.

Designed for remote automation: launches each scheduler sequentially,
prints progress every configurable interval, and writes consolidated
TTFT / E2E / waiting-time statistics plus throughput estimates.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd


DEFAULT_SCHEDULERS: Sequence[str] = (
    "block_offline",
    "block_star_offline",
    "infass_pp",
    "llumnix_minus",
    "random",
    "round_robin",
)

SUMMARY_COLUMNS = (
    ("request_e2e_time", "E2E latency (s)"),
    ("prefill_e2e_time", "TTFT (s)"),
    ("request_waiting_time", "Waiting time (s)"),
)

PERCENTILES = [50, 90, 95, 99]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to place per-scheduler outputs (run logs + CSVs).",
    )
    parser.add_argument(
        "--analysis-file",
        type=Path,
        default=None,
        help="Optional path to write JSON analysis summary (defaults to output_dir/analysis_summary.json).",
    )
    parser.add_argument(
        "--csv-file",
        type=Path,
        default=None,
        help="Optional path to write tabular CSV summary.",
    )
    parser.add_argument(
        "--schedulers",
        nargs="*",
        default=DEFAULT_SCHEDULERS,
        help="Schedulers to run (defaults to Block, Block*, and heuristics).",
    )
    parser.add_argument(
        "--num-replicas",
        type=int,
        required=True,
        help="Number of logical replicas to simulate.",
    )
    parser.add_argument(
        "--qps",
        type=float,
        required=True,
        help="Target queries-per-second for the trace replay.",
    )
    parser.add_argument(
        "--max-requests",
        type=int,
        default=0,
        help="Limit number of requests (0 = full trace).",
    )
    parser.add_argument(
        "--trace-file",
        type=Path,
        default=Path("data/trace_data/sharegpt/sharegpt_val_10k_llama2.csv"),
        help="Ground-truth trace CSV.",
    )
    parser.add_argument(
        "--predicted-trace-file",
        type=Path,
        default=Path(
            "data/trace_data/sharegpt/generate/llama/sharegpt-llama-7b-val-10k-predicted.json"
        ),
        help="Predicted decode-length trace for Block*.",
    )
    parser.add_argument(
        "--block-noise",
        type=float,
        default=10.0,
        help="Noise percentage for Block schedulers (default: 10).",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=1800,
        help="Seconds between progress reports (default: 1800 = 30 minutes).",
    )
    parser.add_argument(
        "--python-executable",
        type=str,
        default=sys.executable,
        help="Python interpreter to use for launching simulations.",
    )
    return parser.parse_args()


def tail_progress(log_path: Path) -> str:
    if not log_path.exists():
        return "log pending"
    with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
        lines = deque(handle, maxlen=50)
    for line in reversed(lines):
        if "Processed" in line:
            return line.strip()
    if lines:
        return lines[-1].strip()
    return "no output yet"


def parse_runtime_seconds(log_path: Path) -> float | None:
    if not log_path.exists():
        return None
    pattern = re.compile(r"Simulation took: ([0-9.]+)s")
    matches = pattern.findall(log_path.read_text(encoding="utf-8", errors="ignore"))
    if not matches:
        return None
    return float(matches[-1])


def run_scheduler(
    scheduler: str,
    args: argparse.Namespace,
    repo_root: Path,
) -> Dict[str, object]:
    scheduler_dir = args.output_dir / scheduler
    scheduler_dir.mkdir(parents=True, exist_ok=True)
    log_path = scheduler_dir / "run.log"

    cmd = [
        args.python_executable,
        "scripts/run_offline_simulations.py",
        "--schedulers",
        scheduler,
        "--num-replicas",
        str(args.num_replicas),
        "--trace-file",
        str(args.trace_file),
        "--predicted-trace-file",
        str(args.predicted_trace_file),
        "--qps",
        str(args.qps),
        "--max-requests",
        str(args.max_requests),
        "--block-noise",
        str(args.block_noise),
        "--output-dir",
        str(args.output_dir),
    ]

    if scheduler in {"block_offline", "block_star_offline"}:
        cmd.extend([
            "--fast-predict",
            "on",
            "--block-parallel-enable",
            "on",
            "--deterministic-noise",
            "on",
        ])

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{repo_root}:{env.get('PYTHONPATH', '')}" if env.get("PYTHONPATH") else str(repo_root)

    print(f"[{datetime.now()}] Starting {scheduler}")
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(cmd, cwd=repo_root, env=env, stdout=log_file, stderr=log_file)

    next_report = time.time() + args.progress_interval
    last_status = "pending"
    while True:
        ret = process.poll()
        now = time.time()
        if now >= next_report or ret is not None:
            last_status = tail_progress(log_path)
            print(f"[{datetime.now()}] [{scheduler}] {last_status}")
            next_report = now + args.progress_interval
        if ret is not None:
            if ret != 0:
                raise RuntimeError(f"Scheduler {scheduler} failed (exit code {ret})")
            break
        time.sleep(15)

    runtime = parse_runtime_seconds(log_path)
    print(f"[{datetime.now()}] Completed {scheduler} (runtime={runtime or 'unknown'} s)")
    return {
        "scheduler": scheduler,
        "log_path": str(log_path.resolve()),
        "runtime_seconds": runtime,
    }


def load_request_metrics(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "request_id" in df.columns:
        key = "request_id"
    elif "Request Id" in df.columns:
        key = "Request Id"
    else:
        key = None
    if key:
        df = df.sort_values(key).reset_index(drop=True)
    return df


def summarise_metrics(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    for column, label in SUMMARY_COLUMNS:
        col = column
        if col not in df.columns:
            if col == "request_waiting_time" and "request_scheduling_delay" in df.columns:
                col = "request_scheduling_delay"
            else:
                continue
        series = df[col].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
        if series.empty:
            continue
        percentiles = np.percentile(series, PERCENTILES)
        summary[label] = {
            "mean": float(series.mean()),
            "min": float(series.min()),
            "max": float(series.max()),
            **{f"p{p}": float(percentiles[idx]) for idx, p in enumerate(PERCENTILES)},
        }
    return summary


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    run_records: Dict[str, Dict[str, object]] = {}
    for scheduler in args.schedulers:
        info = run_scheduler(scheduler, args, repo_root)
        run_records[scheduler] = info

    analysis: Dict[str, object] = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "schedulers": {},
    }

    rows_for_csv: List[Dict[str, object]] = []

    for scheduler in args.schedulers:
        scheduler_dir = args.output_dir / scheduler
        latest_runs = [p for p in scheduler_dir.iterdir() if p.is_dir()]
        if not latest_runs:
            print(f"[WARN] No run directory found for {scheduler}; skipping analysis")
            continue
        run_dir = max(latest_runs, key=lambda path: path.name)
        csv_path = run_dir / "request_metrics.csv"
        if not csv_path.exists():
            print(f"[WARN] Missing request_metrics.csv for {scheduler} at {csv_path}")
            continue

        df = load_request_metrics(csv_path)
        metrics = summarise_metrics(df)
        runtime = run_records.get(scheduler, {}).get("runtime_seconds")
        num_requests = int(df.shape[0])
        throughput = None
        if runtime and runtime > 0:
            throughput = num_requests / runtime

        analysis_entry = {
            "run_dir": str(run_dir.resolve()),
            "log_path": run_records.get(scheduler, {}).get("log_path"),
            "num_requests": num_requests,
            "runtime_seconds": runtime,
            "throughput_rps": throughput,
            "metrics": metrics,
        }
        analysis["schedulers"][scheduler] = analysis_entry

        for label, values in metrics.items():
            row = {
                "scheduler": scheduler,
                "metric": label,
                **values,
            }
            if runtime and runtime > 0:
                row["runtime_seconds"] = runtime
                row["throughput_rps"] = throughput
            rows_for_csv.append(row)

    # Comparative summary vs heuristics baselines (non-Block schedulers)
    heuristics = [
        sched for sched in args.schedulers if sched not in {"block_offline", "block_star_offline"}
    ]

    comparisons: Dict[str, Dict[str, float | None]] = {}
    if heuristics:
        # Throughput (higher is better)
        heuristic_tp = [
            analysis["schedulers"].get(sched, {}).get("throughput_rps")
            for sched in heuristics
            if analysis["schedulers"].get(sched)
            and analysis["schedulers"][sched].get("throughput_rps")
        ]
        best_heuristic_tp = max(heuristic_tp) if heuristic_tp else None

        def pct_improvement(candidate: float | None, baseline: float | None, lower_is_better: bool) -> float | None:
            if candidate is None or baseline is None or baseline == 0:
                return None
            if lower_is_better:
                return (baseline - candidate) / baseline * 100.0
            return (candidate / baseline - 1.0) * 100.0

        for block_sched in ("block_offline", "block_star_offline"):
            sched_entry = analysis["schedulers"].get(block_sched)
            if not sched_entry:
                continue

            entry_summary: Dict[str, float | None] = {}

            # Throughput improvement
            entry_summary["throughput_vs_best_heuristic_pct"] = pct_improvement(
                sched_entry.get("throughput_rps"),
                best_heuristic_tp,
                lower_is_better=False,
            )

            # Latency improvements (lower is better)
            for column, label in SUMMARY_COLUMNS:
                metric_name = label
                heuristic_values = []
                for heur in heuristics:
                    heur_entry = analysis["schedulers"].get(heur)
                    if not heur_entry:
                        continue
                    heur_metrics = heur_entry.get("metrics", {})
                    if metric_name in heur_metrics:
                        heuristic_values.append(heur_metrics[metric_name].get("p50"))
                baseline_metric = min(heuristic_values) if heuristic_values else None

                block_metrics = sched_entry.get("metrics", {}).get(metric_name)
                block_value = block_metrics.get("p50") if block_metrics else None

                entry_summary[f"{metric_name} p50 vs heur pct"] = pct_improvement(
                    block_value,
                    baseline_metric,
                    lower_is_better=True,
                )

            comparisons[block_sched] = entry_summary

    if comparisons:
        analysis["comparisons"] = comparisons

    analysis_path = args.analysis_file or (args.output_dir / "analysis_summary.json")
    analysis_path.parent.mkdir(parents=True, exist_ok=True)
    analysis_path.write_text(json.dumps(analysis, indent=2))
    print(f"Analysis written to {analysis_path}")

    if args.csv_file:
        args.csv_file.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows_for_csv).to_csv(args.csv_file, index=False)
        print(f"CSV summary written to {args.csv_file}")


if __name__ == "__main__":
    main()
