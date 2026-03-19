#!/usr/bin/env python3
"""
End-to-end benchmark runner for CARA scheduler evaluation.

Sends requests through cara_serve and collects per-request metrics.
Supports multiple scheduling strategies for comparison.

Usage:
    # Single scheduler test
    python -m block.exp.cara.run_e2e_benchmark \
        --scheduler-url http://localhost:8200 \
        --test-data data/cara/best-route-v3-test-500.jsonl \
        --scheduling cara \
        --qps 10 --num-requests 100

    # Full comparison across all schedulers
    python -m block.exp.cara.run_e2e_benchmark \
        --scheduler-url http://localhost:8200 \
        --test-data data/cara/best-route-v3-test-500.jsonl \
        --scheduling all \
        --qps 10 --num-requests 500

    # Load sweep
    python -m block.exp.cara.run_e2e_benchmark \
        --scheduler-url http://localhost:8200 \
        --test-data data/cara/best-route-v3-test-500.jsonl \
        --scheduling cara,random,shortest_queue \
        --qps 5,10,15,20 --num-requests 500
"""

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import aiohttp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

ALL_SCHEDULERS = [
    "random", "round_robin", "shortest_queue",
    "quality_greedy", "cost_greedy", "length_aware", "cara",
]


async def send_request(
    session: aiohttp.ClientSession,
    url: str,
    request_data: dict,
    request_id: str,
    budget_tokens: int = 256,
    ttft_slo_ms: float = 5000,
    tpot_slo_ms: float = 200,
) -> dict:
    """Send a single request to the scheduler and collect metrics."""
    payload = {
        "prompt": request_data.get("prompt", ""),
        "max_tokens": request_data.get("max_tokens", 256),
        "request_id": request_id,
        "prompt_len": request_data.get("input_len", len(request_data.get("prompt", "").split())),
        "rso": {
            "budget": budget_tokens,
            "ttft_slo_ms": ttft_slo_ms,
            "tpot_slo_ms": tpot_slo_ms,
        },
    }

    start_time = time.time()
    try:
        async with session.post(
            f"{url}/v1/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            e2e_latency = time.time() - start_time
            result = await resp.json()

            return {
                "request_id": request_id,
                "success": result.get("success", resp.status == 200),
                "e2e_latency": e2e_latency,
                "ttft": result.get("ttft", 0),
                "tpot": result.get("tpot", 0),
                "server_latency": result.get("server_latency", 0),
                "output_tokens": result.get("output_tokens", 0),
                "input_tokens": payload.get("prompt_len", 0),
                "model_used": result.get("model", "unknown"),
                "instance_id": result.get("instance_id", "unknown"),
                "host": result.get("host", "unknown"),
                "budget_tokens": budget_tokens,
                "budget_compliant": (
                    result.get("output_tokens", 0) <= budget_tokens
                    if result.get("output_tokens") else None
                ),
            }
    except Exception as e:
        return {
            "request_id": request_id,
            "success": False,
            "e2e_latency": time.time() - start_time,
            "error": str(e),
        }


async def run_benchmark(
    scheduler_url: str,
    test_data: List[dict],
    num_requests: int,
    qps: float,
    budget_tokens: int = 256,
    ttft_slo_ms: float = 5000,
    tpot_slo_ms: float = 200,
) -> List[dict]:
    """Run benchmark at specified QPS."""
    results = []
    interval = 1.0 / qps if qps > 0 else 0
    num_requests = min(num_requests, len(test_data))

    logger.info(f"Sending {num_requests} requests at {qps} QPS...")

    async with aiohttp.ClientSession() as session:
        tasks = []
        for i in range(num_requests):
            req_data = test_data[i % len(test_data)]
            request_id = f"bench_{i:05d}"

            task = send_request(
                session, scheduler_url, req_data, request_id,
                budget_tokens, ttft_slo_ms, tpot_slo_ms,
            )
            tasks.append(task)

            # Rate limiting
            if interval > 0 and i < num_requests - 1:
                await asyncio.sleep(interval)

        results = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter out exceptions
    clean_results = []
    for r in results:
        if isinstance(r, Exception):
            clean_results.append({"success": False, "error": str(r)})
        else:
            clean_results.append(r)

    return clean_results


def compute_metrics(results: List[dict]) -> dict:
    """Compute aggregate metrics from benchmark results."""
    import numpy as np

    successful = [r for r in results if r.get("success", False)]
    if not successful:
        return {"error": "No successful requests", "total": len(results), "successful": 0}

    latencies = [r["e2e_latency"] for r in successful]
    ttfts = [r["ttft"] for r in successful if r.get("ttft", 0) > 0]
    tpots = [r["tpot"] for r in successful if r.get("tpot", 0) > 0]
    output_tokens = [r["output_tokens"] for r in successful if r.get("output_tokens", 0) > 0]

    # Budget compliance
    budget_results = [r for r in successful if r.get("budget_compliant") is not None]
    budget_rate = (
        sum(1 for r in budget_results if r["budget_compliant"]) / len(budget_results)
        if budget_results else 0
    )

    # Model distribution
    model_counts = {}
    for r in successful:
        model = r.get("model_used", "unknown")
        model_counts[model] = model_counts.get(model, 0) + 1

    metrics = {
        "total_requests": len(results),
        "successful": len(successful),
        "failed": len(results) - len(successful),
        "success_rate": len(successful) / len(results),
        # Latency
        "avg_latency": float(np.mean(latencies)),
        "p50_latency": float(np.percentile(latencies, 50)),
        "p95_latency": float(np.percentile(latencies, 95)),
        "p99_latency": float(np.percentile(latencies, 99)),
        # TTFT
        "avg_ttft": float(np.mean(ttfts)) if ttfts else 0,
        "p95_ttft": float(np.percentile(ttfts, 95)) if ttfts else 0,
        # TPOT
        "avg_tpot": float(np.mean(tpots)) if tpots else 0,
        # Output
        "avg_output_tokens": float(np.mean(output_tokens)) if output_tokens else 0,
        # Budget
        "budget_compliance_rate": budget_rate,
        # Model distribution
        "model_distribution": model_counts,
    }

    return metrics


def print_comparison_table(all_results: dict):
    """Print comparison table across schedulers/QPS levels."""
    print("\n" + "=" * 100)
    print("BENCHMARK RESULTS COMPARISON")
    print("=" * 100)

    header = (
        f"{'Scheduler':<18} {'QPS':>4} {'N':>5} {'OK%':>5} "
        f"{'AvgLat':>7} {'P95Lat':>7} {'P99Lat':>7} "
        f"{'AvgTTFT':>8} {'Budget%':>8} {'Models':>20}"
    )
    print(header)
    print("-" * 100)

    for key, data in sorted(all_results.items()):
        m = data["metrics"]
        models_str = ", ".join(
            f"{k}:{v}" for k, v in sorted(
                m.get("model_distribution", {}).items()
            )
        )[:20]
        print(
            f"  {data['scheduler']:<16} {data['qps']:>4} {m['total_requests']:>5} "
            f"{m['success_rate']*100:>4.0f}% "
            f"{m['avg_latency']:>7.2f} {m['p95_latency']:>7.2f} {m['p99_latency']:>7.2f} "
            f"{m['avg_ttft']:>8.3f} {m['budget_compliance_rate']*100:>7.0f}% "
            f"{models_str:>20}"
        )


async def switch_scheduler(scheduler_url: str, scheduler_name: str) -> bool:
    """Switch the scheduler's scheduling strategy via config update.

    Note: This requires restarting cara_serve with --scheduling flag.
    For benchmarking, we assume the scheduler is started with the right strategy.
    Returns True if the switch was acknowledged.
    """
    # In the current architecture, scheduling strategy is set at startup.
    # For multi-scheduler benchmarking, we'd need to restart cara_serve
    # or add a runtime config endpoint. For now, we log and proceed.
    logger.info(f"Scheduler strategy: {scheduler_name}")
    return True


def load_test_data(path: str) -> List[dict]:
    """Load test data from JSONL or JSON file."""
    data = []
    with open(path) as f:
        if path.endswith(".jsonl"):
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        else:
            raw = json.load(f)
            data = raw if isinstance(raw, list) else raw.get("requests", [raw])
    return data


def main():
    parser = argparse.ArgumentParser(description="CARA E2E Benchmark Runner")
    parser.add_argument("--scheduler-url", default="http://localhost:8200",
                        help="CARA scheduler URL")
    parser.add_argument("--test-data", required=True,
                        help="Test data file (JSONL or JSON)")
    parser.add_argument("--scheduling", default="random",
                        help="Comma-separated schedulers or 'all'. "
                             "Options: random,round_robin,shortest_queue,"
                             "quality_greedy,cost_greedy,length_aware,cara")
    parser.add_argument("--qps", default="10",
                        help="Comma-separated QPS levels (e.g., '5,10,15,20')")
    parser.add_argument("--num-requests", type=int, default=100,
                        help="Number of requests per run")
    parser.add_argument("--budget-tokens", type=int, default=256,
                        help="Token budget for RSO")
    parser.add_argument("--ttft-slo-ms", type=float, default=5000,
                        help="TTFT SLO in milliseconds")
    parser.add_argument("--tpot-slo-ms", type=float, default=200,
                        help="TPOT SLO in milliseconds")
    parser.add_argument("--output-dir", default="experiment_output/e2e_benchmark",
                        help="Output directory for results")
    parser.add_argument("--warmup", type=int, default=10,
                        help="Number of warmup requests before benchmark")
    args = parser.parse_args()

    # Parse schedulers
    if args.scheduling == "all":
        schedulers = ALL_SCHEDULERS
    else:
        schedulers = [s.strip() for s in args.scheduling.split(",")]

    # Parse QPS levels
    qps_levels = [float(q.strip()) for q in args.qps.split(",")]

    # Load test data
    test_data = load_test_data(args.test_data)
    logger.info(f"Loaded {len(test_data)} test requests from {args.test_data}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for scheduler in schedulers:
        for qps in qps_levels:
            run_key = f"{scheduler}_qps{qps}"
            logger.info(f"\n{'='*60}")
            logger.info(f"Running: {scheduler} @ {qps} QPS, {args.num_requests} requests")
            logger.info(f"{'='*60}")

            # Note: In production, you'd restart cara_serve with --scheduling <name>
            # For now, we log which scheduler is being tested
            logger.info(
                f"Ensure cara_serve is running with --scheduling {scheduler}"
            )

            # Run benchmark
            results = asyncio.run(run_benchmark(
                args.scheduler_url, test_data, args.num_requests,
                qps, args.budget_tokens, args.ttft_slo_ms, args.tpot_slo_ms,
            ))

            # Compute metrics
            metrics = compute_metrics(results)

            all_results[run_key] = {
                "scheduler": scheduler,
                "qps": qps,
                "num_requests": args.num_requests,
                "metrics": metrics,
            }

            # Save per-run results
            run_output = output_dir / f"{run_key}.jsonl"
            with open(run_output, "w") as f:
                for r in results:
                    f.write(json.dumps(r) + "\n")

            logger.info(
                f"  Results: {metrics['successful']}/{metrics['total_requests']} ok, "
                f"avg_lat={metrics['avg_latency']:.2f}s, "
                f"p95={metrics['p95_latency']:.2f}s, "
                f"budget={metrics['budget_compliance_rate']*100:.0f}%"
            )

    # Print comparison
    print_comparison_table(all_results)

    # Save summary
    summary_path = output_dir / "benchmark_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
