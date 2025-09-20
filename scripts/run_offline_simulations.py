#!/usr/bin/env python3
"""Driver to launch Vidur offline simulations for large-scale scenarios."""

import argparse
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
from vidur.config import (
    BlockOfflineGlobalSchedulerConfig,
    BlockStarOfflineGlobalSchedulerConfig,
    ClusterConfig,
    InfassPlusPlusGlobalSchedulerConfig,
    LinearRegressionExecutionTimePredictorConfig,
    LlumnixMinusGlobalSchedulerConfig,
    MetricsConfig,
    NoisySimulationRequestTimelinePredictorConfig,
    RandomForrestExecutionTimePredictorConfig,
    RandomGlobalSchedulerConfig,
    ReplicaConfig,
    RoundRobinGlobalSchedulerConfig,
    SarathiSchedulerConfig,
    SimulationConfig,
    TraceRequestGeneratorConfig,
)
from vidur.entities import Cluster, Replica, Request
from vidur.simulator import Simulator
from vidur.scheduler.utils.memory_planner import MemoryPlanner
from vidur.types.optimal_global_scheduler_target_metric import TargetMetric


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--schedulers",
        nargs="+",
        default=["block_offline", "infass_pp", "llumnix_minus"],
        help="Schedulers to evaluate.",
    )
    parser.add_argument(
        "--num-replicas",
        type=int,
        default=12,
        help="Number of replicas (logical GPUs) to simulate.",
    )
    parser.add_argument(
        "--trace-file",
        type=str,
        default="data/trace_data/sharegpt/sharegpt_val_10k_llama2.csv",
        help="Ground-truth trace CSV path.",
    )
    parser.add_argument(
        "--predicted-trace-file",
        type=str,
        default="data/trace_data/sharegpt/generate/llama/sharegpt-llama-7b-val-10k-predicted.json",
        help="Trace file containing predicted response lengths.",
    )
    parser.add_argument(
        "--time-scale-factor",
        type=float,
        default=None,
        help="Direct scaling multiplier for arrival times (overrides inferred value).",
    )
    parser.add_argument(
        "--qps",
        type=float,
        default=None,
        help="Target arrival rate (requests/sec). Overrides --time-scale-factor if provided.",
    )
    parser.add_argument(
        "--noise-fraction",
        type=float,
        default=None,
        help="Deprecated: use --block-noise instead (still honored if set).",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Simulation seed.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("simulation_analysis/offline"),
        help="Base directory for simulation outputs (default: simulation_analysis/offline).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Maximum tokens per request enforced by the trace loader.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="a30",
        help="Device SKU (must match entries under data/profiling).",
    )
    parser.add_argument(
        "--network-device",
        type=str,
        default="a30_single_gpu",
        help="Network SKU (used by execution-time predictor).",
    )
    parser.add_argument(
        "--execution-model",
        choices=["linear", "rf"],
        default="linear",
        help="Execution-time predictor backend (linear is faster for offline sweeps).",
    )
    parser.add_argument(
        "--max-requests",
        type=int,
        default=200,
        help="Replay only the first N requests (default: 200 to keep runtimes manageable).",
    )
    parser.add_argument(
        "--block-noise",
        type=float,
        default=10.0,
        help="Noise percentage for Block schedulers (e.g., 10 => ±10% multiplicative noise).",
    )
    parser.add_argument(
        "--block-target-metric",
        type=str,
        default="min_latency",
        help="Target metric for Block schedulers (e.g., min_latency, min_scheduling_delay).",
    )
    parser.add_argument(
        "--block-parallel-enable",
        choices=["on", "off"],
        default="off",
        help="Enable process-based per-replica what-if evaluation (single toggle).",
    )
    parser.add_argument(
        "--deterministic-noise",
        choices=["on", "off"],
        default="off",
        help="Use deterministic noise schedule (required for parallel Block with noise).",
    )
    parser.add_argument(
        "--fast-predict",
        choices=["on", "off"],
        default="off",
        help="Enable fast snapshot-based Block predictor (experimental; default off for parity).",
    )
    return parser.parse_args()


def ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_scheduler_config(
    name: str,
    noise_fraction_fraction: float,
    random_seed: int,
    block_metric: str,
    fast_predict: bool,
    parallel: bool,
    deterministic_noise: bool,
):
    normalized = name.lower()
    if normalized == "block_offline":
        predictor_cfg = NoisySimulationRequestTimelinePredictorConfig(
            noise_fraction=noise_fraction_fraction,
            random_seed=random_seed,
        )
        return BlockOfflineGlobalSchedulerConfig(
            target_metric=block_metric,
            fast_predict=fast_predict,
            request_timeline_predictor_config=predictor_cfg,
            parallel=parallel,
            parallel_workers=0,
            parallel_backend="process",
            deterministic_noise=deterministic_noise,
        )
    if normalized == "block_star_offline":
        predictor_cfg = NoisySimulationRequestTimelinePredictorConfig(
            noise_fraction=noise_fraction_fraction,
            random_seed=random_seed,
        )
        return BlockStarOfflineGlobalSchedulerConfig(
            target_metric=block_metric,
            fast_predict=fast_predict,
            request_timeline_predictor_config=predictor_cfg,
            parallel=parallel,
            parallel_workers=0,
            parallel_backend="process",
            deterministic_noise=deterministic_noise,
        )
    if normalized in {"infass_pp", "infass++", "infass"}:
        return InfassPlusPlusGlobalSchedulerConfig()
    if normalized in {"llumnix_minus", "llumnix-", "llumnix"}:
        return LlumnixMinusGlobalSchedulerConfig()
    if normalized == "random":
        return RandomGlobalSchedulerConfig()
    if normalized in {"round_robin", "round-robin", "rr"}:
        return RoundRobinGlobalSchedulerConfig()

    raise ValueError(f"Unsupported scheduler {name}")


def run_simulation(
    scheduler_name: str,
    args: argparse.Namespace,
    output_dir: Path,
) -> None:
    reset_entity_ids()
    noise_fraction = resolve_block_noise_fraction(args)
    parallel_enabled = args.block_parallel_enable == "on"
    deterministic_noise = args.deterministic_noise == "on"

    if parallel_enabled and noise_fraction > 0 and not deterministic_noise:
        raise ValueError(
            "Parallel Block offline scheduler requires deterministic noise when noise_fraction > 0. "
            "Re-run with --deterministic-noise on or --block-noise 0."
        )

    scheduler_config = build_scheduler_config(
        scheduler_name,
        noise_fraction,
        args.random_seed,
        resolve_block_target_metric(args.block_target_metric),
        args.fast_predict == "on",
        parallel_enabled,
        deterministic_noise,
    )

    time_scale_factor = resolve_time_scale_factor(
        args.trace_file,
        args.qps,
        args.time_scale_factor,
        args.max_requests,
    )

    use_predicted_lengths = scheduler_name.lower() in {"block_star_offline"}

    metrics_config = MetricsConfig(
        output_dir=str(output_dir / scheduler_name.lower()),
        create_output_dir=True,
        enable_chrome_trace=False,
        write_json_trace=False,
        store_plots=False,
    )

    request_generator_config = TraceRequestGeneratorConfig(
        trace_file=args.trace_file,
        predicted_trace_file=args.predicted_trace_file,
        use_predicted_decode_tokens=use_predicted_lengths,
        time_scale_factor=time_scale_factor,
        max_tokens=args.max_tokens,
        max_requests=args.max_requests,
    )

    replica_config = ReplicaConfig(
        device=args.device,
        network_device=args.network_device,
    )

    base_sarathi_config = SarathiSchedulerConfig()
    replica = Replica(replica_config, request_generator_config)
    memory_planner = MemoryPlanner(replica_config, replica)
    max_blocks_per_sequence = (
        request_generator_config.max_tokens // base_sarathi_config.block_size
    )
    computed_num_blocks = (
        max_blocks_per_sequence * memory_planner.get_max_request_slots()
    )
    base_sarathi_config.num_blocks = computed_num_blocks

    cluster_config = ClusterConfig(
        num_replicas=args.num_replicas,
        replica_config=replica_config,
        global_scheduler_config=scheduler_config,
        replica_scheduler_config=base_sarathi_config,
    )

    if args.execution_model == "rf":
        execution_config = RandomForrestExecutionTimePredictorConfig(skip_cpu_overhead_modeling=True)
    else:
        execution_config = LinearRegressionExecutionTimePredictorConfig(skip_cpu_overhead_modeling=True)

    simulation_config = SimulationConfig(
        seed=args.random_seed,
        cluster_config=cluster_config,
        request_generator_config=request_generator_config,
        execution_time_predictor_config=execution_config,
        metrics_config=metrics_config,
    )

    simulator = Simulator(simulation_config)
    simulator.run()


def main() -> None:
    args = parse_args()
    ensure_output_dir(args.output_dir)
    for scheduler in normalise_scheduler_names(args.schedulers):
        run_simulation(scheduler, args, args.output_dir)


def normalise_scheduler_names(names: Iterable[str]) -> List[str]:
    seen = []
    for name in names:
        lower = name.lower()
        if lower not in seen:
            seen.append(lower)
    return seen


def reset_entity_ids() -> None:
    Replica._id = -1
    Request._id = -1
    Cluster._id = -1


def resolve_block_noise_fraction(args: argparse.Namespace) -> float:
    if args.noise_fraction is not None:
        return args.noise_fraction
    # convert percent to fraction
    return max(args.block_noise, 0.0) / 100.0


def resolve_block_target_metric(metric_name: str) -> str:
    candidate = metric_name.replace("-", "_").upper()
    try:
        metric = TargetMetric.from_str(candidate)
    except KeyError as exc:
        raise ValueError(
            f"Unknown Block target metric '{metric_name}'."
        ) from exc
    return str(metric)


def resolve_time_scale_factor(
    trace_file: str,
    target_qps: Optional[float],
    explicit_scale: Optional[float],
    max_requests: int,
) -> float:
    if explicit_scale is not None and target_qps is None:
        return explicit_scale

    df = pd.read_csv(trace_file, usecols=None)
    if max_requests and max_requests > 0:
        df = df.head(max_requests)

    if "arrived_at" in df.columns:
        arrivals = df["arrived_at"].to_numpy()
    else:
        arrivals = df.index.to_numpy(dtype=float)

    if len(arrivals) <= 1:
        base_rate = 1.0
    else:
        duration = arrivals[-1] - arrivals[0]
        if duration <= 0:
            base_rate = float(len(arrivals))
        else:
            base_rate = len(arrivals) / duration

    if target_qps:
        scale = base_rate / target_qps
        print(
            f"[offline-sim] Base rate {base_rate:.4f} req/s, target QPS {target_qps:.4f}, "
            f"time_scale_factor -> {scale:.6f}"
        )
    elif explicit_scale is not None:
        scale = explicit_scale
    else:
        scale = 1.0

    return scale


if __name__ == "__main__":
    main()
