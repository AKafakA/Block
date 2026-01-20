# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
r"""

Coped from vllm.benchmarks.serving_benchmark.py with modifications for Cara changes.
"""

import argparse
import asyncio
import contextlib
import importlib.util
import json
import os
import random
import shutil
import time
import uuid
import warnings
from collections.abc import AsyncGenerator, Iterable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Literal

import aiohttp
import numpy as np
from tqdm.asyncio import tqdm

# Use compatibility layer for vLLM benchmarks (supports both old and new vLLM versions)
from block.benchmark.cara.vllm_compat import (
    add_dataset_parser,
    ASYNC_REQUEST_FUNCS,
    OPENAI_COMPATIBLE_BACKENDS,
    RequestFuncInput,
    wait_for_endpoint,
    convert_to_pytorch_benchmark_format,
    write_to_json,
    get_tokenizer,
    freeze_gc_heap,
    join_host_port,
)

from transformers import PreTrainedTokenizerBase


from block.benchmark.cara.cara_end_point_func import RequestFuncOutput, CARA_ASYNC_REQUEST_FUNCS
# Import SampleRequest from dataset.py (local canonical implementation)
from block.benchmark.cara.dataset import get_samples, SampleRequest
from block.global_scheduler.cara.utils import set_ulimit

# Merge CARA async request functions with vLLM's request functions
ASYNC_REQUEST_FUNCS = {**ASYNC_REQUEST_FUNCS, **CARA_ASYNC_REQUEST_FUNCS}


def parse_length_constraints_from_filename(filepath: str) -> dict:
    """Parse length constraints from dataset filename.

    Expected format: {name}-p{max_prompt}-o{max_output}-t{max_total}.jsonl
    Example: best-route-p1024-o1024-t2048.jsonl

    Returns dict with keys: max_prompt_len, max_output_len (optional), max_total_len
    Values are None if not found in filename.
    """
    import re
    import os

    basename = os.path.basename(filepath)
    constraints = {
        "max_prompt_len": None,
        "max_output_len": None,
        "max_total_len": None,
    }

    # Match patterns like -p1024, -o1024, -t2048
    prompt_match = re.search(r'-p(\d+)', basename)
    output_match = re.search(r'-o(\d+)', basename)
    total_match = re.search(r'-t(\d+)', basename)

    if prompt_match:
        constraints["max_prompt_len"] = int(prompt_match.group(1))
    if output_match:
        constraints["max_output_len"] = int(output_match.group(1))
    if total_match:
        constraints["max_total_len"] = int(total_match.group(1))

    return constraints


def apply_filename_constraints_to_args(args) -> None:
    """Apply length constraints from dataset filename to args if not explicitly set.

    Only applies constraints if the corresponding arg was not provided (uses default).
    """
    if not args.dataset_path:
        return

    constraints = parse_length_constraints_from_filename(args.dataset_path)

    applied = []
    # Apply max_prompt_len if found and not explicitly set (check against default 1024)
    if constraints["max_prompt_len"] is not None:
        if not hasattr(args, '_max_prompt_len_set') or not args._max_prompt_len_set:
            args.max_prompt_len = constraints["max_prompt_len"]
            applied.append(f"max_prompt_len={constraints['max_prompt_len']}")

    # Apply max_output_len -> custom_output_len if found
    if constraints["max_output_len"] is not None:
        if not hasattr(args, '_custom_output_len_set') or not args._custom_output_len_set:
            args.custom_output_len = constraints["max_output_len"]
            applied.append(f"custom_output_len={constraints['max_output_len']}")

    # Apply max_total_len if found
    if constraints["max_total_len"] is not None:
        if not hasattr(args, '_max_total_len_set') or not args._max_total_len_set:
            args.max_total_len = constraints["max_total_len"]
            applied.append(f"max_total_len={constraints['max_total_len']}")

    if applied:
        print(f"Auto-detected length constraints from filename: {', '.join(applied)}")

MILLISECONDS_TO_SECONDS_CONVERSION = 1000

TERM_PLOTLIB_AVAILABLE = (importlib.util.find_spec("termplotlib") is not None) and (
    shutil.which("gnuplot") is not None
)


class TaskType(Enum):
    GENERATION = "generation"
    POOLING = "pooling"


@dataclass
class BenchmarkMetrics:
    completed: int
    failed: int
    total_input: int
    total_output: int
    request_throughput: float
    request_goodput: float
    output_throughput: float
    total_token_throughput: float
    mean_ttft_ms: float
    median_ttft_ms: float
    std_ttft_ms: float
    percentiles_ttft_ms: list[tuple[float, float]]
    mean_tpot_ms: float
    median_tpot_ms: float
    std_tpot_ms: float
    percentiles_tpot_ms: list[tuple[float, float]]
    mean_itl_ms: float
    median_itl_ms: float
    std_itl_ms: float
    percentiles_itl_ms: list[tuple[float, float]]
    # E2EL stands for end-to-end latency per request.
    # It is the time taken on the client side from sending
    # a request to receiving a complete response.
    mean_e2el_ms: float
    median_e2el_ms: float
    std_e2el_ms: float
    percentiles_e2el_ms: list[tuple[float, float]]
    # Max output tokens per second and concurrent requests at that peak
    max_output_tokens_per_s: float
    max_concurrent_requests: int


@dataclass
class EmbedBenchmarkMetrics:
    completed: int
    failed: int
    total_input: int
    request_throughput: float
    total_token_throughput: float
    mean_e2el_ms: float
    std_e2el_ms: float
    median_e2el_ms: float
    percentiles_e2el_ms: float


def _get_current_request_rate(
    ramp_up_strategy: Literal["linear", "exponential"] | None,
    ramp_up_start_rps: int | None,
    ramp_up_end_rps: int | None,
    request_index: int,
    total_requests: int,
    request_rate: float,
) -> float:
    if (
        ramp_up_strategy
        and ramp_up_start_rps is not None
        and ramp_up_end_rps is not None
    ):
        progress = request_index / max(total_requests - 1, 1)
        if ramp_up_strategy == "linear":
            increase = (ramp_up_end_rps - ramp_up_start_rps) * progress
            return ramp_up_start_rps + increase
        elif ramp_up_strategy == "exponential":
            ratio = ramp_up_end_rps / ramp_up_start_rps
            return ramp_up_start_rps * (ratio**progress)
        else:
            raise ValueError(f"Unknown ramp-up strategy: {ramp_up_strategy}")
    return request_rate


async def get_request(
    input_requests: list[SampleRequest],
    request_rate: float,
    burstiness: float = 1.0,
    ramp_up_strategy: Literal["linear", "exponential"] | None = None,
    ramp_up_start_rps: int | None = None,
    ramp_up_end_rps: int | None = None,
) -> AsyncGenerator[tuple[SampleRequest, float], None]:
    """
    Asynchronously generates requests at a specified rate
    with OPTIONAL burstiness and OPTIONAL ramp-up strategy.

    Args:
        input_requests:
            A list of input requests, each represented as a SampleRequest.
        request_rate:
            The rate at which requests are generated (requests/s).
        burstiness (optional):
            The burstiness factor of the request generation.
            Only takes effect when request_rate is not inf.
            Default value is 1, which follows a Poisson process.
            Otherwise, the request intervals follow a gamma distribution.
            A lower burstiness value (0 < burstiness < 1) results
            in more bursty requests, while a higher burstiness value
            (burstiness > 1) results in a more uniform arrival of requests.
        ramp_up_strategy (optional):
            The ramp-up strategy. Can be "linear" or "exponential".
            If None, uses constant request rate (specified by request_rate).
        ramp_up_start_rps (optional):
            The starting request rate for ramp-up.
        ramp_up_end_rps (optional):
            The ending request rate for ramp-up.
    """
    assert burstiness > 0, (
        f"A positive burstiness factor is expected, but given {burstiness}."
    )
    # Convert to list to get length for ramp-up calculations
    if isinstance(input_requests, Iterable) and not isinstance(input_requests, list):
        input_requests = list(input_requests)

    total_requests = len(input_requests)
    assert total_requests > 0, "No requests provided."

    # Precompute delays among requests to minimize request send laggings
    request_rates = []
    delay_ts = []
    for request_index, request in enumerate(input_requests):
        current_request_rate = _get_current_request_rate(
            ramp_up_strategy,
            ramp_up_start_rps,
            ramp_up_end_rps,
            request_index,
            total_requests,
            request_rate,
        )
        assert current_request_rate > 0.0, (
            f"Obtained non-positive request rate {current_request_rate}."
        )
        request_rates.append(current_request_rate)
        if current_request_rate == float("inf"):
            delay_ts.append(0)
        elif burstiness == float("inf"):
            # when burstiness tends to infinity, the delay time becomes constant
            # and tends to the inverse of the request rate
            delay_ts.append(1.0 / current_request_rate)
        else:
            theta = 1.0 / (current_request_rate * burstiness)

            # Sample the request interval from the gamma distribution.
            # If burstiness is 1, it follows exponential distribution.
            delay_ts.append(np.random.gamma(shape=burstiness, scale=theta))

    # Calculate the cumulative delay time from the first sent out requests.
    for i in range(1, len(delay_ts)):
        delay_ts[i] += delay_ts[i - 1]
    if ramp_up_strategy is None and delay_ts[-1] != 0:
        # When ramp_up_strategy is not set, we assume the request rate is fixed
        # and all requests should be sent in target_total_delay_s, the following
        # logic would re-scale delay time to ensure the final delay_ts
        # align with target_total_delay_s.
        #
        # NOTE: If we simply accumulate the random delta values
        # from the gamma distribution, their sum would have 1-2% gap
        # from target_total_delay_s. The purpose of the following logic is to
        # close the gap for stabilizing the throughput data
        # from different random seeds.
        target_total_delay_s = total_requests / request_rate
        normalize_factor = target_total_delay_s / delay_ts[-1]
        delay_ts = [delay * normalize_factor for delay in delay_ts]

    start_ts = time.time()
    for request_index, request in enumerate(input_requests):
        if delay_ts[request_index] > 0:
            current_ts = time.time()
            sleep_interval_s = start_ts + delay_ts[request_index] - current_ts
            if sleep_interval_s > 0:
                await asyncio.sleep(sleep_interval_s)
        yield request, request_rates[request_index]


def calculate_metrics_for_embeddings(
    outputs: list[RequestFuncOutput], dur_s: float, selected_percentiles: list[float]
) -> EmbedBenchmarkMetrics:
    """Calculate the metrics for the embedding requests.

    Args:
        outputs: The outputs of the requests.
        dur_s: The duration of the benchmark.
        selected_percentiles: The percentiles to select.

    Returns:
        The calculated benchmark metrics.
    """
    total_input = 0
    completed = 0
    failed = 0
    e2els: list[float] = []
    for i in range(len(outputs)):
        if outputs[i].success:
            e2els.append(outputs[i].latency)
            completed += 1
            total_input += outputs[i].prompt_len
        else:
            failed += 1

    if completed == 0:
        warnings.warn(
            "All requests failed. This is likely due to a misconfiguration "
            "on the benchmark arguments.",
            stacklevel=2,
        )
    metrics = EmbedBenchmarkMetrics(
        completed=completed,
        failed=failed,
        total_input=total_input,
        request_throughput=completed / dur_s,
        total_token_throughput=total_input / dur_s,
        mean_e2el_ms=np.mean(e2els or 0) * 1000,
        std_e2el_ms=np.std(e2els or 0) * 1000,
        median_e2el_ms=np.median(e2els or 0) * 1000,
        percentiles_e2el_ms=[
            (p, np.percentile(e2els or 0, p) * 1000) for p in selected_percentiles
        ],
    )
    return metrics


def calculate_metrics(
    input_requests: list[SampleRequest],
    outputs: list[RequestFuncOutput],
    dur_s: float,
    tokenizer: PreTrainedTokenizerBase,
    selected_percentiles: list[float],
    goodput_config_dict: dict[str, float],
) -> tuple[BenchmarkMetrics, list[int]]:
    """Calculate the metrics for the benchmark.

    Args:
        input_requests: The input requests.
        outputs: The outputs of the requests.
        dur_s: The duration of the benchmark.
        tokenizer: The tokenizer to use.
        selected_percentiles: The percentiles to select.
        goodput_config_dict: The goodput configuration.

    Returns:
        A tuple of the benchmark metrics and the actual output lengths.
    """
    actual_output_lens: list[int] = []
    total_input = 0
    completed = 0
    good_completed = 0
    itls: list[float] = []
    tpots: list[float] = []
    all_tpots: list[float] = []
    ttfts: list[float] = []
    e2els: list[float] = []
    for i in range(len(outputs)):
        if outputs[i].success:
            output_len = outputs[i].output_tokens

            if not output_len:
                # We use the tokenizer to count the number of output tokens
                # for some serving backends instead of looking at
                # len(outputs[i].itl) since multiple output tokens may be
                # bundled together
                # Note : this may inflate the output token count slightly
                output_len = len(
                    tokenizer(
                        outputs[i].generated_text, add_special_tokens=False
                    ).input_ids
                )
            actual_output_lens.append(output_len)
            total_input += input_requests[i].prompt_len
            tpot = 0
            if output_len > 1:
                latency_minus_ttft = outputs[i].latency - outputs[i].ttft
                tpot = latency_minus_ttft / (output_len - 1)
                tpots.append(tpot)
            # Note: if output_len <= 1, we regard tpot as 0 for goodput
            all_tpots.append(tpot)
            itls += outputs[i].itl
            ttfts.append(outputs[i].ttft)
            e2els.append(outputs[i].latency)
            completed += 1
        else:
            actual_output_lens.append(0)

    if goodput_config_dict:
        valid_metrics = []
        slo_values = []

        if "ttft" in goodput_config_dict:
            valid_metrics.append(ttfts)
            slo_values.append(
                goodput_config_dict["ttft"] / MILLISECONDS_TO_SECONDS_CONVERSION
            )
        if "tpot" in goodput_config_dict:
            valid_metrics.append(all_tpots)
            slo_values.append(
                goodput_config_dict["tpot"] / MILLISECONDS_TO_SECONDS_CONVERSION
            )
        if "e2el" in goodput_config_dict:
            valid_metrics.append(e2els)
            slo_values.append(
                goodput_config_dict["e2el"] / MILLISECONDS_TO_SECONDS_CONVERSION
            )

        for req_metric in zip(*valid_metrics):
            is_good_req = all([s >= r for s, r in zip(slo_values, req_metric)])
            if is_good_req:
                good_completed += 1

    if completed == 0:
        warnings.warn(
            "All requests failed. This is likely due to a misconfiguration "
            "on the benchmark arguments.",
            stacklevel=2,
        )

    # Calculate max output tokens per second metric
    max_output_tokens_per_s = 0.0
    max_concurrent_requests = 0

    # Find the time range across all successful requests
    successful_outputs = [output for output in outputs if output.success]
    failed_outputs = [output for output in outputs if not output.success]

    # Log detailed failed request information to a file for debugging
    if failed_outputs:
        from datetime import datetime
        from collections import Counter
        error_log_filename = os.path.abspath(os.path.expanduser(
            f"failed_requests_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        ))

        # Analyze failure patterns
        failures_by_instance = Counter(f.instance_id or 'Unknown' for f in failed_outputs)
        failures_by_host = Counter(f.host or 'Unknown' for f in failed_outputs)
        failures_by_model = Counter(f.model or 'Unknown' for f in failed_outputs)

        with open(error_log_filename, 'w') as error_log:
            error_log.write("=" * 80 + "\n")
            error_log.write(f"FAILED REQUESTS DEBUG LOG\n")
            error_log.write(f"Generated: {datetime.now().isoformat()}\n")
            error_log.write(f"Total Failed Requests: {len(failed_outputs)}\n")
            error_log.write("=" * 80 + "\n\n")

            # Summary section
            error_log.write("FAILURE SUMMARY:\n")
            error_log.write("-" * 80 + "\n")
            error_log.write(f"\nFailures by Instance:\n")
            for instance_id, count in failures_by_instance.most_common():
                error_log.write(f"  {instance_id}: {count} failures\n")

            error_log.write(f"\nFailures by Host:\n")
            for host, count in failures_by_host.most_common():
                error_log.write(f"  {host}: {count} failures\n")

            error_log.write(f"\nFailures by Model:\n")
            for model, count in failures_by_model.most_common():
                error_log.write(f"  {model}: {count} failures\n")
            error_log.write("\n" + "=" * 80 + "\n")

            # Detailed failure information
            error_log.write("\nDETAILED FAILURE INFORMATION:\n")
            error_log.write("=" * 80 + "\n")

            for idx, failed_output in enumerate(failed_outputs, 1):
                error_log.write(f"\n{'='*80}\n")
                error_log.write(f"Failed Request #{idx}\n")
                error_log.write(f"{'='*80}\n")
                error_log.write(f"Request ID:     {failed_output.request_id or 'N/A'}\n")
                error_log.write(f"Instance ID:    {failed_output.instance_id or 'N/A'}\n")
                error_log.write(f"Host:           {failed_output.host or 'N/A'}\n")
                error_log.write(f"Model:          {failed_output.model or 'N/A'}\n")
                error_log.write(f"Start Time:     {failed_output.start_time}\n")
                error_log.write(f"Prompt Length:  {failed_output.prompt_len}\n")
                error_log.write(f"\nError Message:\n")
                error_log.write("-" * 80 + "\n")
                error_log.write(failed_output.error or 'No error message')
                error_log.write("\n" + "-" * 80 + "\n")

        print(f"\n⚠️  {len(failed_outputs)} failed requests detected!")
        print(f"📝 Detailed error logs written to: {error_log_filename}")
        print(f"   Open in vim: vim {error_log_filename}")
        print(f"    Summary: {dict(failures_by_instance.most_common(3))}\n")

    if successful_outputs:
        min_start_time = min(output.start_time for output in successful_outputs)
        max_end_time = max(
            output.start_time + output.latency for output in successful_outputs
        )

        # Create second buckets (ceiling to ensure we capture all time)
        duration_seconds = int(np.ceil(max_end_time - min_start_time)) + 1
        tokens_per_second = np.zeros(duration_seconds)
        concurrent_requests_per_second = np.zeros(duration_seconds)

        for i, output in enumerate(successful_outputs):
            # Calculate token generation timestamp using
            # start_time, ttft, and itl
            token_times = [output.start_time + output.ttft]
            current_time = token_times[0]
            for itl_value in output.itl:
                current_time += itl_value
                token_times.append(current_time)

            # Add tokens to second buckets
            for token_time in token_times:
                second_bucket = int(token_time - min_start_time)
                if 0 <= second_bucket < duration_seconds:
                    tokens_per_second[second_bucket] += 1

            # Track concurrent requests for each second this request was active
            request_start_second = int(output.start_time - min_start_time)
            request_end_second = int(
                (output.start_time + output.latency) - min_start_time
            )

            for second in range(request_start_second, request_end_second + 1):
                concurrent_requests_per_second[second] += 1

        # Find the maximum tokens per second and corresponding
        # concurrent requests
        if len(tokens_per_second) > 0:
            max_output_tokens_per_s = float(np.max(tokens_per_second))
            max_concurrent_requests = int(np.max(concurrent_requests_per_second))

        if TERM_PLOTLIB_AVAILABLE:
            import termplotlib as tpl

            fig = tpl.figure()
            fig.plot(
                np.arange(len(tokens_per_second)),
                tokens_per_second,
                title="Output tokens per second",
            )
            fig.plot(
                np.arange(len(concurrent_requests_per_second)),
                concurrent_requests_per_second,
                title="Concurrent requests per second",
            )
            fig.show()
        else:
            print("tip: install termplotlib and gnuplot to plot the metrics")

    metrics = BenchmarkMetrics(
        completed=completed,
        failed=len(failed_outputs),
        total_input=total_input,
        total_output=sum(actual_output_lens),
        request_throughput=completed / dur_s,
        request_goodput=good_completed / dur_s,
        output_throughput=sum(actual_output_lens) / dur_s,
        total_token_throughput=(total_input + sum(actual_output_lens)) / dur_s,
        mean_ttft_ms=np.mean(ttfts or 0)
        * 1000,  # ttfts is empty if streaming is not supported by the endpoint
        std_ttft_ms=np.std(ttfts or 0) * 1000,
        median_ttft_ms=np.median(ttfts or 0) * 1000,
        percentiles_ttft_ms=[
            (p, np.percentile(ttfts or 0, p) * 1000) for p in selected_percentiles
        ],
        mean_tpot_ms=np.mean(tpots or 0) * 1000,
        std_tpot_ms=np.std(tpots or 0) * 1000,
        median_tpot_ms=np.median(tpots or 0) * 1000,
        percentiles_tpot_ms=[
            (p, np.percentile(tpots or 0, p) * 1000) for p in selected_percentiles
        ],
        mean_itl_ms=np.mean(itls or 0) * 1000,
        std_itl_ms=np.std(itls or 0) * 1000,
        median_itl_ms=np.median(itls or 0) * 1000,
        percentiles_itl_ms=[
            (p, np.percentile(itls or 0, p) * 1000) for p in selected_percentiles
        ],
        mean_e2el_ms=np.mean(e2els or 0) * 1000,
        std_e2el_ms=np.std(e2els or 0) * 1000,
        median_e2el_ms=np.median(e2els or 0) * 1000,
        percentiles_e2el_ms=[
            (p, np.percentile(e2els or 0, p) * 1000) for p in selected_percentiles
        ],
        max_output_tokens_per_s=max_output_tokens_per_s,
        max_concurrent_requests=max_concurrent_requests,
    )

    return metrics, actual_output_lens


async def benchmark(
    task_type: TaskType,
    endpoint_type: str,
    api_url: str,
    base_url: str,
    model_id: str,
    model_name: str,
    tokenizer: PreTrainedTokenizerBase,
    input_requests: list[SampleRequest],
    logprobs: int | None,
    request_rate: float,
    burstiness: float,
    disable_tqdm: bool,
    num_warmups: int,
    profile: bool,
    selected_percentile_metrics: list[str],
    selected_percentiles: list[float],
    ignore_eos: bool,
    goodput_config_dict: dict[str, float],
    max_concurrency: int | None,
    lora_modules: Iterable[str] | None,
    extra_headers: dict | None,
    extra_body: dict | None,
    ramp_up_strategy: Literal["linear", "exponential"] | None = None,
    ramp_up_start_rps: int | None = None,
    ramp_up_end_rps: int | None = None,
    ready_check_timeout_sec: int = 600,
):
    try:
        request_func = ASYNC_REQUEST_FUNCS[endpoint_type]
    except KeyError:
        raise ValueError(f"Unknown backend: {endpoint_type}") from None

    # Reuses connections across requests to reduce TLS handshake overhead.
    connector = aiohttp.TCPConnector(
        limit=max_concurrency or 0,
        limit_per_host=max_concurrency or 0,
        ttl_dns_cache=300,
        use_dns_cache=True,
        keepalive_timeout=60,
        enable_cleanup_closed=True,
        force_close=False,
        ssl=("https://" in api_url),
    )

    session = aiohttp.ClientSession(
        connector=connector,
        trust_env=True,
        timeout=aiohttp.ClientTimeout(total=6 * 60 * 60),
    )

    print("Checking endpoint readiness...")
    test_prompt, test_prompt_len, test_output_len, test_mm_content = (
        input_requests[0].prompt,
        input_requests[0].prompt_len,
        input_requests[0].expected_output_len,
        input_requests[0].multi_modal_data,
    )

    assert (
        test_mm_content is None
        or isinstance(test_mm_content, dict)
        or (
            isinstance(test_mm_content, list)
            and all(isinstance(item, dict) for item in test_mm_content)
        )
    ), "multi_modal_data must be a dict or list[dict]"
    test_input = RequestFuncInput(
        model=model_id,
        model_name=model_name,
        prompt=test_prompt,
        api_url=api_url,
        prompt_len=test_prompt_len,
        output_len=test_output_len,
        logprobs=logprobs,
        multi_modal_content=test_mm_content,
        ignore_eos=ignore_eos,
        extra_headers=extra_headers,
        extra_body=extra_body,
    )

    if ready_check_timeout_sec > 0:
        if endpoint_type == "cara":
            # Fast health-check for CARA server to avoid long warmup generation
            from urllib.parse import urlparse, urlunparse
            parsed = urlparse(api_url)
            base = urlunparse((parsed.scheme, parsed.netloc, "", "", "", ""))
            health_url = base.rstrip("/") + "/health"
            deadline = time.time() + ready_check_timeout_sec
            ok = False
            print(f"Waiting for endpoint to become ready (GET {health_url})...")
            while time.time() < deadline:
                try:
                    async with session.get(health_url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                        if resp.status == 200:
                            ok = True
                            break
                except Exception:
                    pass
                await asyncio.sleep(1)
            if not ok:
                raise TimeoutError(
                    f"Endpoint did not become ready within {ready_check_timeout_sec} seconds"
                )
            print("Endpoint is ready.")
        else:
            test_output = await wait_for_endpoint(
                request_func,
                test_input,
                session,
                timeout_seconds=ready_check_timeout_sec,
            )
            if getattr(test_output, "success", True) is False:
                raise ValueError(
                    "Initial test run failed - Please make sure benchmark "
                    "arguments are correctly specified. "
                    f"Error: {test_output.error}"
                )
            else:
                print("Endpoint is ready.")
    else:
        print("Skipping endpoint ready check.")

    if num_warmups > 0:
        print(f"Warming up with {num_warmups} requests...")
        warmup_pbar = None if disable_tqdm else tqdm(total=num_warmups)
        warmup_semaphore = (
            asyncio.Semaphore(max_concurrency)
            if max_concurrency
            else contextlib.nullcontext()
        )
        warmup_tasks = []

        async def warmup_limited_request_func():
            async with warmup_semaphore:
                return await request_func(
                    request_func_input=test_input, session=session, pbar=warmup_pbar
                )

        for _ in range(num_warmups):
            request_task = asyncio.create_task(warmup_limited_request_func())
            warmup_tasks.append(request_task)
        _ = await asyncio.gather(*warmup_tasks)

        if warmup_pbar is not None:
            warmup_pbar.close()
        print("Warmup run completed.")

    print("Starting main benchmark run...")

    if lora_modules:
        # For each input request, choose a LoRA module at random.
        lora_modules = iter(
            [random.choice(lora_modules) for _ in range(len(input_requests))]
        )

    if profile:
        print("Starting profiler...")
        profile_input = RequestFuncInput(
            model=model_id,
            model_name=model_name,
            prompt=test_prompt,
            api_url=base_url + "/start_profile",
            prompt_len=test_prompt_len,
            output_len=test_output_len,
            logprobs=logprobs,
            multi_modal_content=test_mm_content,
            ignore_eos=ignore_eos,
            extra_headers=extra_headers,
            extra_body=extra_body,
        )
        profile_output = await request_func(
            request_func_input=profile_input, session=session
        )
        if profile_output.success:
            print("Profiler started")

    distribution = "Poisson process" if burstiness == 1.0 else "Gamma distribution"

    if ramp_up_strategy is not None:
        print(f"Traffic ramp-up strategy: {ramp_up_strategy}.")
        print(
            f"Will increase RPS from {ramp_up_start_rps} to "
            f"{ramp_up_end_rps} RPS over the duration of the benchmark."
        )
    else:
        print(f"Traffic request rate: {request_rate}")

    print(f"Burstiness factor: {burstiness} ({distribution})")
    print(f"Maximum request concurrency: {max_concurrency}")

    pbar = None if disable_tqdm else tqdm(total=len(input_requests))

    semaphore = (
        asyncio.Semaphore(max_concurrency)
        if max_concurrency
        else contextlib.nullcontext()
    )

    async def limited_request_func(request_func_input, session, pbar):
        async with semaphore:
            return await request_func(
                request_func_input=request_func_input, session=session, pbar=pbar
            )

    benchmark_start_time = time.perf_counter()
    tasks: list[asyncio.Task] = []

    rps_change_events = []
    last_int_rps = -1
    if ramp_up_strategy is not None and ramp_up_start_rps is not None:
        last_int_rps = ramp_up_start_rps
        rps_change_events.append(
            {
                "rps": last_int_rps,
                "timestamp": datetime.now().isoformat(),
            }
        )

    async for request, current_request_rate in get_request(
        input_requests,
        request_rate,
        burstiness,
        ramp_up_strategy,
        ramp_up_start_rps,
        ramp_up_end_rps,
    ):
        if ramp_up_strategy is not None:
            current_int_rps = int(current_request_rate)
            if current_int_rps > last_int_rps:
                timestamp = datetime.now().isoformat()
                for rps_val in range(last_int_rps + 1, current_int_rps + 1):
                    rps_change_events.append({"rps": rps_val, "timestamp": timestamp})
                last_int_rps = current_int_rps
        prompt, prompt_len, output_len, mm_content, request_id = (
            request.prompt,
            request.prompt_len,
            request.expected_output_len,
            request.multi_modal_data,
            request.request_id,
        )
        req_model_id, req_model_name = model_id, model_name
        if lora_modules:
            req_lora_module = next(lora_modules)
            req_model_id, req_model_name = req_lora_module, req_lora_module

        # Enforce client-side cap so prompt_len + output_len <= args.max_total_len
        # This keeps requests within backend max-model-len (commonly 1024 in our deployments)
        allowed_output_len = max(0, (args.max_total_len or 1024) - (prompt_len or 0))
        capped_output_len = min(output_len or 0, allowed_output_len)

        request_func_input = RequestFuncInput(
            model=req_model_id,
            model_name=req_model_name,
            prompt=prompt,
            api_url=api_url,
            prompt_len=prompt_len,
            output_len=capped_output_len,
            logprobs=logprobs,
            multi_modal_content=mm_content,
            ignore_eos=ignore_eos,
            extra_headers=extra_headers,
            extra_body=extra_body,
            request_id=request_id,
        )
        tasks.append(
            asyncio.create_task(
                limited_request_func(
                    request_func_input=request_func_input, session=session, pbar=pbar
                )
            )
        )
    outputs: list[RequestFuncOutput] = await asyncio.gather(*tasks)

    if pbar is not None:
        pbar.close()

    benchmark_duration = time.perf_counter() - benchmark_start_time

    if task_type == TaskType.GENERATION:
        metrics, actual_output_lens = calculate_metrics(
            input_requests=input_requests,
            outputs=outputs,
            dur_s=benchmark_duration,
            tokenizer=tokenizer,
            selected_percentiles=selected_percentiles,
            goodput_config_dict=goodput_config_dict,
        )
    else:
        metrics = calculate_metrics_for_embeddings(
            outputs=outputs,
            dur_s=benchmark_duration,
            selected_percentiles=selected_percentiles,
        )
        actual_output_lens = 0

    print("{s:{c}^{n}}".format(s=" Serving Benchmark Result ", n=50, c="="))
    print("{:<40} {:<10}".format("Successful requests:", metrics.completed))
    print("{:<40} {:<10}".format("Failed requests:", metrics.failed))
    if max_concurrency is not None:
        print("{:<40} {:<10}".format("Maximum request concurrency:", max_concurrency))
    if request_rate != float("inf"):
        print("{:<40} {:<10.2f}".format("Request rate configured (RPS):", request_rate))
    print("{:<40} {:<10.2f}".format("Benchmark duration (s):", benchmark_duration))
    print("{:<40} {:<10}".format("Total input tokens:", metrics.total_input))
    if isinstance(metrics, BenchmarkMetrics):
        print("{:<40} {:<10}".format("Total generated tokens:", metrics.total_output))
    print(
        "{:<40} {:<10.2f}".format(
            "Request throughput (req/s):", metrics.request_throughput
        )
    )
    if goodput_config_dict:
        print(
            "{:<40} {:<10.2f}".format(
                "Request goodput (req/s):", metrics.request_goodput
            )
        )
    if isinstance(metrics, BenchmarkMetrics):
        print(
            "{:<40} {:<10.2f}".format(
                "Output token throughput (tok/s):", metrics.output_throughput
            )
        )
        print(
            "{:<40} {:<10.2f}".format(
                "Peak output token throughput (tok/s):", metrics.max_output_tokens_per_s
            )
        )
        print(
            "{:<40} {:<10.2f}".format(
                "Peak concurrent requests:", metrics.max_concurrent_requests
            )
        )
    print(
        "{:<40} {:<10.2f}".format(
            "Total token throughput (tok/s):", metrics.total_token_throughput
        )
    )

    if isinstance(metrics, BenchmarkMetrics):
        result = {
            "duration": benchmark_duration,
            "completed": metrics.completed,
            "failed": metrics.failed,
            "total_input_tokens": metrics.total_input,
            "total_output_tokens": metrics.total_output,
            "request_throughput": metrics.request_throughput,
            "request_goodput": metrics.request_goodput if goodput_config_dict else None,
            "output_throughput": metrics.output_throughput,
            "total_token_throughput": metrics.total_token_throughput,
            "max_output_tokens_per_s": metrics.max_output_tokens_per_s,
            "max_concurrent_requests": metrics.max_concurrent_requests,
        }
        # CARA-specific: add unified per-request records for easier analysis
        # Map save_detailed metric names to per-request keys
        per_req_keys = [
            "request_id", "prompt", "input_len", "output_len", "response",
            "ttft", "itl", "e2el", "scheduling_overhead", "model", "host",
            "instance_id", "error", "broadcast_results"
        ]
        # Build per-request dicts (fill None for non-selected fields later)
        reqs = []
        for i in range(len(outputs)):
            out = outputs[i]
            req = {
                "request_id": getattr(out, "request_id", str(i)),
                "prompt": input_requests[i].prompt if i < len(input_requests) else None,
                "input_len": out.prompt_len,
                "output_len": actual_output_lens[i] if i < len(actual_output_lens) else None,
                "response": out.generated_text,
                "ttft": out.ttft,
                "itl": out.itl,
                "e2el": out.latency,
                "scheduling_overhead": out.scheduling_overhead,
                "model": out.model,
                "host": out.host,
                "instance_id": out.instance_id,
                "error": out.error,
                "broadcast_results": getattr(out, "broadcast_results", []),
            }
            # Ensure all expected keys exist
            for k in per_req_keys:
                req.setdefault(k, None)
            reqs.append(req)
        result["response_details"] = reqs
    else:
        result = {
            "duration": benchmark_duration,
            "completed": metrics.completed,
            "total_input_tokens": metrics.total_input,
            "request_throughput": metrics.request_throughput,
            "total_token_throughput": metrics.total_token_throughput,
            "input_lens": [output.prompt_len for output in outputs],
            "errors": [output.error for output in outputs],
        }

    if rps_change_events:
        result["rps_change_events"] = rps_change_events

    def process_one_metric(
        # E.g., "ttft"
        metric_attribute_name: str,
        # E.g., "TTFT"
        metric_name: str,
        # E.g., "Time to First Token"
        metric_header: str,
    ):
        # This function prints and adds statistics of the specified
        # metric.
        if metric_attribute_name not in selected_percentile_metrics:
            return
        print("{s:{c}^{n}}".format(s=metric_header, n=50, c="-"))
        print(
            "{:<40} {:<10.2f}".format(
                f"Mean {metric_name} (ms):",
                getattr(metrics, f"mean_{metric_attribute_name}_ms"),
            )
        )
        print(
            "{:<40} {:<10.2f}".format(
                f"Median {metric_name} (ms):",
                getattr(metrics, f"median_{metric_attribute_name}_ms"),
            )
        )
        result[f"mean_{metric_attribute_name}_ms"] = getattr(
            metrics, f"mean_{metric_attribute_name}_ms"
        )
        result[f"median_{metric_attribute_name}_ms"] = getattr(
            metrics, f"median_{metric_attribute_name}_ms"
        )
        result[f"std_{metric_attribute_name}_ms"] = getattr(
            metrics, f"std_{metric_attribute_name}_ms"
        )
        for p, value in getattr(metrics, f"percentiles_{metric_attribute_name}_ms"):
            p_word = str(int(p)) if int(p) == p else str(p)
            print("{:<40} {:<10.2f}".format(f"P{p_word} {metric_name} (ms):", value))
            result[f"p{p_word}_{metric_attribute_name}_ms"] = value

    if task_type == TaskType.GENERATION:
        process_one_metric("ttft", "TTFT", "Time to First Token")
        process_one_metric("tpot", "TPOT", "Time per Output Token (excl. 1st token)")
        process_one_metric("itl", "ITL", "Inter-token Latency")
    process_one_metric("e2el", "E2EL", "End-to-end Latency")

    # CARA-specific: Report scheduling overhead metrics
    if endpoint_type == "cara":
        scheduling_overheads = [output.scheduling_overhead for output in outputs if output.success]
        if scheduling_overheads:
            print("{s:{c}^{n}}".format(s="Scheduling Overhead (CARA Network + Routing)", n=50, c="-"))
            mean_overhead_ms = np.mean(scheduling_overheads) * 1000
            median_overhead_ms = np.median(scheduling_overheads) * 1000
            std_overhead_ms = np.std(scheduling_overheads) * 1000
            print("{:<40} {:<10.2f}".format("Mean Scheduling Overhead (ms):", mean_overhead_ms))
            print("{:<40} {:<10.2f}".format("Median Scheduling Overhead (ms):", median_overhead_ms))
            result["mean_scheduling_overhead_ms"] = mean_overhead_ms
            result["median_scheduling_overhead_ms"] = median_overhead_ms
            result["std_scheduling_overhead_ms"] = std_overhead_ms
            for p in selected_percentiles:
                p_value = np.percentile(scheduling_overheads, p) * 1000
                p_word = str(int(p)) if int(p) == p else str(p)
                print("{:<40} {:<10.2f}".format(f"P{p_word} Scheduling Overhead (ms):", p_value))
                result[f"p{p_word}_scheduling_overhead_ms"] = p_value

    # Print prompt and response length statistics
    if isinstance(metrics, BenchmarkMetrics):
        print("{s:{c}^{n}}".format(s="Prompt & Response Length Statistics", n=50, c="-"))

        # Get successful requests only for accurate statistics
        successful_input_lens = [outputs[i].prompt_len for i in range(len(outputs)) if outputs[i].success]
        successful_output_lens = [actual_output_lens[i] for i in range(len(outputs)) if outputs[i].success]

        if successful_input_lens and successful_output_lens:
            # Prompt length statistics
            mean_prompt_len = np.mean(successful_input_lens)
            median_prompt_len = np.median(successful_input_lens)
            std_prompt_len = np.std(successful_input_lens)

            print("{:<40} {:<10.2f}".format("Mean Prompt Length (tokens):", mean_prompt_len))
            print("{:<40} {:<10.2f}".format("Median Prompt Length (tokens):", median_prompt_len))
            print("{:<40} {:<10.2f}".format("Std Prompt Length (tokens):", std_prompt_len))

            for p in selected_percentiles:
                p_value = np.percentile(successful_input_lens, p)
                p_word = str(int(p)) if int(p) == p else str(p)
                print("{:<40} {:<10.2f}".format(f"P{p_word} Prompt Length (tokens):", p_value))
                result[f"p{p_word}_prompt_len"] = p_value

            print()

            # Response length statistics
            mean_output_len = np.mean(successful_output_lens)
            median_output_len = np.median(successful_output_lens)
            std_output_len = np.std(successful_output_lens)

            print("{:<40} {:<10.2f}".format("Mean Response Length (tokens):", mean_output_len))
            print("{:<40} {:<10.2f}".format("Median Response Length (tokens):", median_output_len))
            print("{:<40} {:<10.2f}".format("Std Response Length (tokens):", std_output_len))

            for p in selected_percentiles:
                p_value = np.percentile(successful_output_lens, p)
                p_word = str(int(p)) if int(p) == p else str(p)
                print("{:<40} {:<10.2f}".format(f"P{p_word} Response Length (tokens):", p_value))
                result[f"p{p_word}_output_len"] = p_value

            print()

            # Correlation between prompt and response lengths
            if len(successful_input_lens) > 1:
                # Check if there's variance in both arrays (std > 0)
                # If either has zero variance, correlation is undefined
                if std_prompt_len > 0 and std_output_len > 0:
                    correlation = np.corrcoef(successful_input_lens, successful_output_lens)[0, 1]
                    print("{:<40} {:<10.4f}".format("Prompt-Response Correlation:", correlation))
                    result["prompt_response_correlation"] = correlation
                else:
                    print("{:<40} {:<10}".format("Prompt-Response Correlation:", "N/A (zero variance)"))
                    result["prompt_response_correlation"] = None

            # Add statistics to result
            result["mean_prompt_len"] = mean_prompt_len
            result["median_prompt_len"] = median_prompt_len
            result["std_prompt_len"] = std_prompt_len
            result["mean_output_len"] = mean_output_len
            result["median_output_len"] = median_output_len
            result["std_output_len"] = std_output_len

    print("=" * 50)

    if profile:
        print("Stopping profiler...")
        profile_input = RequestFuncInput(
            model=model_id,
            prompt=test_prompt,
            api_url=base_url + "/stop_profile",
            prompt_len=test_prompt_len,
            output_len=test_output_len,
            logprobs=logprobs,
        )
        profile_output = await request_func(
            request_func_input=profile_input, session=session
        )
        if profile_output.success:
            print("Profiler stopped")

    await session.close()
    return result


def check_goodput_args(args):
    # Check and parse goodput arguments
    goodput_config_dict = {}
    VALID_NAMES = ["ttft", "tpot", "e2el"]
    if args.goodput:
        goodput_config_dict = parse_goodput(args.goodput)
        for slo_name, slo_val in goodput_config_dict.items():
            if slo_name not in VALID_NAMES:
                raise ValueError(
                    f"Invalid metric name found, {slo_name}: {slo_val}. "
                    "The service level objective name should be one of "
                    f"{str(VALID_NAMES)}. "
                )
            if slo_val < 0:
                raise ValueError(
                    f"Invalid value found, {slo_name}: {slo_val}. "
                    "The service level objective value should be "
                    "non-negative."
                )
    return goodput_config_dict


def parse_goodput(slo_pairs):
    goodput_config_dict = {}
    try:
        for slo_pair in slo_pairs:
            slo_name, slo_val = slo_pair.split(":")
            goodput_config_dict[slo_name] = float(slo_val)
    except ValueError as err:
        raise argparse.ArgumentTypeError(
            "Invalid format found for service level objectives. "
            'Specify service level objectives for goodput as "KEY:VALUE" '
            "pairs, where the key is a metric name, and the value is a "
            "number in milliseconds."
        ) from err
    return goodput_config_dict


def save_to_pytorch_benchmark_format(
    args: argparse.Namespace, results: dict[str, Any], file_name: str
) -> None:
    metrics = [
        "median_ttft_ms",
        "mean_ttft_ms",
        "std_ttft_ms",
        "p99_ttft_ms",
        "mean_tpot_ms",
        "median_tpot_ms",
        "std_tpot_ms",
        "p99_tpot_ms",
        "median_itl_ms",
        "mean_itl_ms",
        "std_itl_ms",
        "p99_itl_ms",
    ]
    # These raw data might be useful, but they are rather big. They can be added
    # later if needed
    ignored_metrics = ["ttfts", "itls", "generated_texts", "errors"]
    pt_records = convert_to_pytorch_benchmark_format(
        args=args,
        metrics={k: [results[k]] for k in metrics if k in results},
        extra_info={
            k: results[k]
            for k in results
            if k not in metrics and k not in ignored_metrics
        },
    )
    if pt_records:
        # Don't use json suffix here as we don't want CI to pick it up
        pt_file = f"{os.path.splitext(file_name)[0]}.pytorch.json"
        write_to_json(pt_file, pt_records)


def add_cli_args(parser: argparse.ArgumentParser):
    add_dataset_parser(parser)
    # # Normalize default for custom datasets across environments (A100/P100)
    # # vLLM's parser on some nodes may default to 256; override to 1024 for consistency.
    # parser.set_defaults(custom_output_len=1024)
    # # Align dataset sampling with backend cap defaults (commonly 1024 on CARA deployments)
    # parser.set_defaults(max_total_len=1024)
    # Note: --custom-output-len, --max-prompt-len, --min-prompt-len, --max-total-len
    # are provided by add_dataset_parser (vLLM compat layer).

    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="The label (prefix) of the benchmark results. If not specified, "
        "the value of '--backend' will be used as the label.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="cara",
        choices=list(ASYNC_REQUEST_FUNCS.keys()),
        help="The type of backend or endpoint to use for the benchmark.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Server or API base url if not using http host and port.",
    )
    # Use 127.0.0.1 here instead of localhost to force the use of ipv4
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8200)
    parser.add_argument(
        "--endpoint",
        type=str,
        default="/v1/completions",
        help="API endpoint.",
    )
    parser.add_argument(
        "--header",
        metavar="KEY=VALUE",
        nargs="*",
        help="Key-value pairs (e.g, --header x-additional-info=0.3.3) "
        "for headers to be passed with each request. These headers override "
        "per backend constants and values set via environment variable, and "
        "will be overridden by other arguments (such as request ids).",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=None,
        help="Maximum number of concurrent requests. This can be used "
        "to help simulate an environment where a higher level component "
        "is enforcing a maximum number of concurrent requests. While the "
        "--request-rate argument controls the rate at which requests are "
        "initiated, this argument will control how many are actually allowed "
        "to execute at a time. This means that when used in combination, the "
        "actual request rate may be lower than specified with --request-rate, "
        "if the server is not processing requests fast enough to keep up.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-3B",
        help="Name of the model running at cara service also, so we can use its tokenizer to avoid redownloading.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        help="Name or path of the tokenizer, if not using the default tokenizer.",  # noqa: E501
    )
    parser.add_argument(
        "--tokenizer-mode",
        type=str,
        default="auto",
        help="""Tokenizer mode:\n
        - "auto" will use the tokenizer from `mistral_common` for Mistral models
        if available, otherwise it will use the "hf" tokenizer.\n
        - "hf" will use the fast tokenizer if available.\n
        - "slow" will always use the slow tokenizer.\n
        - "mistral" will always use the tokenizer from `mistral_common`.\n
        - "deepseek_v32" will always use the tokenizer from `deepseek_v32`.\n
        - Other custom values can be supported via plugins.""",
    )
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument(
        "--logprobs",
        type=int,
        default=None,
        help=(
            "Number of logprobs-per-token to compute & return as part of "
            "the request. If unspecified, then either (1) if beam search "
            "is disabled, no logprobs are computed & a single dummy "
            "logprob is returned for each token; or (2) if beam search "
            "is enabled 1 logprob per token is computed"
        ),
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Number of requests per second. If this is inf, "
        "then all the requests are sent at time 0. "
        "Otherwise, we use Poisson process or gamma distribution "
        "to synthesize the request arrival times.",
    )
    parser.add_argument(
        "--burstiness",
        type=float,
        default=1.0,
        help="Burstiness factor of the request generation. "
        "Only take effect when request_rate is not inf. "
        "Default value is 1, which follows Poisson process. "
        "Otherwise, the request intervals follow a gamma distribution. "
        "A lower burstiness value (0 < burstiness < 1) results in more "
        "bursty requests. A higher burstiness value (burstiness > 1) "
        "results in a more uniform arrival of requests.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code from huggingface",
    )
    parser.add_argument(
        "--disable-tqdm",
        action="store_true",
        help="Specify to disable tqdm progress bar.",
    )
    parser.add_argument(
        "--num-warmups",
        type=int,
        default=0,
        help="Number of warmup requests.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Use vLLM Profiling. --profiler-config must be provided on the server.",
    )
    parser.add_argument(
        "--save-result",
        action="store_true",
        help="Specify to save benchmark results to a json file",
    )
    parser.add_argument(
        "--save-detailed",
        action="store_true",
        help=(
            "Save results and include per-request response_details "
            "(request_id, prompt, input_len, output_len, response, ttft, itl, "
            "e2el, scheduling_overhead, model, host, instance_id, error, "
            "broadcast_results). Equivalent to enabling --save-result and "
            "adding detailed fields."
        ),
    )
    parser.add_argument(
        "--append-result",
        action="store_true",
        help="Append the benchmark result to the existing json file.",
    )
    parser.add_argument(
        "--metadata",
        metavar="KEY=VALUE",
        nargs="*",
        help="Key-value pairs (e.g, --metadata version=0.3.3 tp=1) "
        "for metadata of this run to be saved in the result JSON file "
        "for record keeping purposes.",
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default=None,
        help="Specify directory to save benchmark json results."
        "If not specified, results are saved in the current directory.",
    )
    parser.add_argument(
        "--result-filename",
        type=str,
        default=None,
        help="Specify the filename to save benchmark json results."
        "If not specified, results will be saved in "
        "{label}-{args.request_rate}qps-{base_model_id}-{current_dt}.json"  # noqa
        " format.",
    )
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help="Set ignore_eos flag when sending the benchmark request."
        "Warning: ignore_eos is not supported in deepspeed_mii and tgi.",
    )
    parser.add_argument(
        "--percentile-metrics",
        type=str,
        default=None,
        help="Comma-separated list of selected metrics to report percentiles. "
        "This argument specifies the metrics to report percentiles. "
        'Allowed metric names are "ttft", "tpot", "itl", "e2el". '
        'If not specified, defaults to "ttft,tpot,itl" for generative models '
        'and "e2el" for pooling models.',
    )
    parser.add_argument(
        "--metric-percentiles",
        type=str,
        default="50,99",
        help="Comma-separated list of percentiles for selected metrics. "
        'To report 25-th, 50-th, and 75-th percentiles, use "25,50,75". '
        'Default value is "99".'
        'Use "--percentile-metrics" to select metrics.',
    )
    parser.add_argument(
        "--goodput",
        nargs="+",
        required=False,
        help='Specify service level objectives for goodput as "KEY:VALUE" '
        "pairs, where the key is a metric name, and the value is in "
        'milliseconds. Multiple "KEY:VALUE" pairs can be provided, '
        "separated by spaces. Allowed request level metric names are "
        '"ttft", "tpot", "e2el". For more context on the definition of '
        "goodput, refer to DistServe paper: https://arxiv.org/pdf/2401.09670 "
        "and the blog: https://hao-ai-lab.github.io/blogs/distserve",
    )
    # NOTE: --request-id-prefix is added by add_dataset_parser (from vLLM or
    # our compatibility layer). Avoid redefining it here to prevent argparse
    # conflicts on older/newer vLLM versions.

    sampling_group = parser.add_argument_group("sampling parameters")
    sampling_group.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Top-p sampling parameter. Only has effect on openai-compatible backends.",
    )
    sampling_group.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-k sampling parameter. Only has effect on openai-compatible backends.",
    )
    sampling_group.add_argument(
        "--min-p",
        type=float,
        default=None,
        help="Min-p sampling parameter. Only has effect on openai-compatible backends.",
    )
    sampling_group.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Temperature sampling parameter. Only has effect on "
        "openai-compatible backends. If not specified, default to greedy "
        "decoding (i.e. temperature==0.0).",
    )
    sampling_group.add_argument(
        "--frequency-penalty",
        type=float,
        default=None,
        help="Frequency penalty sampling parameter. Only has effect on "
        "openai-compatible backends.",
    )
    sampling_group.add_argument(
        "--presence-penalty",
        type=float,
        default=None,
        help="Presence penalty sampling parameter. Only has effect on "
        "openai-compatible backends.",
    )
    sampling_group.add_argument(
        "--repetition-penalty",
        type=float,
        default=None,
        help="Repetition penalty sampling parameter. Only has effect on "
        "openai-compatible backends.",
    )
    sampling_group.add_argument(
        "--early-stopping",
        action="store_true",
        help="Enable early stopping (backend-dependent; forwarded to backend via request body)",
    )
    sampling_group.add_argument(
        "--common-prefix-len",
        type=int,
        default=None,
        help="Common prefix length shared by all prompts (used by random dataset)",
    )

    parser.add_argument(
        "--served-model-name",
        type=str,
        default=None,
        help="The model name used in the API. "
        "If not specified, the model name will be the "
        "same as the `--model` argument. ",
    )

    parser.add_argument(
        "--lora-modules",
        nargs="+",
        default=None,
        help="A subset of LoRA module names passed in when "
        "launching the server. For each request, the "
        "script chooses a LoRA module at random.",
    )

    parser.add_argument(
        "--ramp-up-strategy",
        type=str,
        default=None,
        choices=["linear", "exponential"],
        help="The ramp-up strategy. This would be used to "
        "ramp up the request rate from initial RPS to final "
        "RPS rate (specified by --ramp-up-start-rps and "
        "--ramp-up-end-rps.) over the duration of the benchmark.",
    )
    parser.add_argument(
        "--ramp-up-start-rps",
        type=int,
        default=None,
        help="The starting request rate for ramp-up (RPS). "
        "Needs to be specified when --ramp-up-strategy is used.",
    )
    parser.add_argument(
        "--ramp-up-end-rps",
        type=int,
        default=None,
        help="The ending request rate for ramp-up (RPS). "
        "Needs to be specified when --ramp-up-strategy is used.",
    )
    parser.add_argument(
        "--ready-check-timeout-sec",
        type=int,
        default=600,
        help="Maximum time to wait for the endpoint to become ready "
        "in seconds (default: 600 seconds / 10 minutes). If set to 0, "
        "the ready check will be skipped.",
    )

    parser.add_argument(
        "--extra-body",
        help="A JSON string representing extra body parameters to include "
        "in each request."
        'Example: \'{"chat_template_kwargs":{"enable_thinking":false}}\'',
        type=json.loads,
        default=None,
    )



def main(args: argparse.Namespace) -> dict[str, Any]:
    return asyncio.run(main_async(args))


async def main_async(args: argparse.Namespace) -> dict[str, Any]:
    print(args)

    # Set file descriptor limit to handle high concurrency
    if set_ulimit():
        print("Successfully set ulimit for benchmark client")
    else:
        print("Warning: Failed to increase ulimit - may encounter 'Too many open files' errors with high concurrency")

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Validate ramp-up arguments
    if args.ramp_up_strategy is not None:
        if args.request_rate != float("inf"):
            raise ValueError(
                "When using ramp-up, do not specify --request-rate. "
                "The request rate will be controlled by ramp-up parameters. "
                "Please remove the --request-rate argument."
            )
        if args.ramp_up_start_rps is None or args.ramp_up_end_rps is None:
            raise ValueError(
                "When using --ramp-up-strategy, both --ramp-up-start-rps and "
                "--ramp-up-end-rps must be specified"
            )
        if args.ramp_up_start_rps < 0 or args.ramp_up_end_rps < 0:
            raise ValueError("Ramp-up start and end RPS must be non-negative")
        if args.ramp_up_start_rps > args.ramp_up_end_rps:
            raise ValueError("Ramp-up start RPS must be less than end RPS")
        if args.ramp_up_strategy == "exponential" and args.ramp_up_start_rps == 0:
            raise ValueError("For exponential ramp-up, the start RPS cannot be 0.")

    label = args.label
    model_id = args.model
    model_name = args.served_model_name
    tokenizer_id = args.tokenizer if args.tokenizer is not None else args.model
    tokenizer_mode = args.tokenizer_mode

    if args.base_url is not None:
        api_url = f"{args.base_url}{args.endpoint}"
        base_url = f"{args.base_url}"
    else:
        host_port = join_host_port(args.host, args.port)
        api_url = f"http://{host_port}{args.endpoint}"
        base_url = f"http://{host_port}"

    # Headers
    headers = None
    if args.header:
        headers = {}
        for item in args.header:
            if "=" in item:
                kvstring = item.split("=", 1)
                headers[kvstring[0].strip()] = kvstring[1].strip()
            else:
                raise ValueError("Invalid header format. Please use KEY=VALUE format.")

    tokenizer = get_tokenizer(
        tokenizer_id,
        tokenizer_mode=tokenizer_mode,
        trust_remote_code=args.trust_remote_code,
    )

    if args.dataset_name is None:
        raise ValueError(
            "Please specify '--dataset-name' and the corresponding "
            "'--dataset-path' if required."
        )

    # when using random datasets, default to ignoring EOS
    # so generation runs to the requested length
    if (
        args.dataset_name in ("random", "random-mm")
        and args.backend in OPENAI_COMPATIBLE_BACKENDS
    ):
        args.ignore_eos = True

    # Load the dataset.
    input_requests = get_samples(args, tokenizer)
    goodput_config_dict = check_goodput_args(args)

    backend = args.backend
    task_type = (
        TaskType.POOLING
        if "embeddings" in backend or "rerank" in backend
        else TaskType.GENERATION
    )

    # Collect the sampling parameters.
    if task_type == TaskType.GENERATION:
        # Default to beam search config suggested by paper:
        # use_beam_search=True, best_of=5, temperature=0, top_p=1, top_k=-1, early_stopping=True
        # CLI-provided values (if any) override these defaults for temperature/top_p/top_k.
        sampling_params = {
            "use_beam_search": True,
            "best_of": 5,
            "temperature": 0.0 if args.temperature is None else args.temperature,
            "top_p": 1.0 if args.top_p is None else args.top_p,
            "top_k": -1 if args.top_k is None else args.top_k,
            "min_p": args.min_p,
            "frequency_penalty": args.frequency_penalty,
            "presence_penalty": args.presence_penalty,
            "repetition_penalty": args.repetition_penalty,
            "early_stopping": True,
        }

        # Sampling parameters are only supported by openai-compatible backend.
        if sampling_params and args.backend not in OPENAI_COMPATIBLE_BACKENDS:
            raise ValueError(
                "Sampling parameters are only supported by openai-compatible backends."
            )

        if "temperature" not in sampling_params:
            sampling_params["temperature"] = 0.0  # Default to greedy decoding.

        default_percentile_metrics = "ttft,tpot,itl,e2el"
    else:
        sampling_params = {}
        default_percentile_metrics = "e2el"

    extra_body = args.extra_body or {}
    extra_body = {**sampling_params, **extra_body}

    percentile_metrics: str = args.percentile_metrics or default_percentile_metrics

    # Avoid GC processing "static" data - reduce pause times.
    freeze_gc_heap()

    benchmark_result = await benchmark(
        task_type=task_type,
        endpoint_type=backend,
        api_url=api_url,
        base_url=base_url,
        model_id=model_id,
        model_name=model_name,
        tokenizer=tokenizer,
        input_requests=input_requests,
        logprobs=args.logprobs,
        request_rate=args.request_rate,
        burstiness=args.burstiness,
        disable_tqdm=args.disable_tqdm,
        num_warmups=args.num_warmups,
        profile=args.profile,
        selected_percentile_metrics=percentile_metrics.split(","),
        selected_percentiles=[float(p) for p in args.metric_percentiles.split(",")],
        ignore_eos=args.ignore_eos,
        goodput_config_dict=goodput_config_dict,
        max_concurrency=args.max_concurrency,
        lora_modules=args.lora_modules,
        extra_headers=headers,
        extra_body=extra_body,
        ramp_up_strategy=args.ramp_up_strategy,
        ramp_up_start_rps=args.ramp_up_start_rps,
        ramp_up_end_rps=args.ramp_up_end_rps,
        ready_check_timeout_sec=args.ready_check_timeout_sec,
    )

    # Save config and results to json
    result_json: dict[str, Any] = {}

    # Setup
    current_dt = datetime.now().strftime("%Y%m%d-%H%M%S")
    result_json["date"] = current_dt
    result_json["endpoint_type"] = args.backend  # for backward compatibility
    result_json["backend"] = args.backend
    result_json["label"] = label
    result_json["model_id"] = model_id
    result_json["tokenizer_id"] = tokenizer_id
    result_json["num_prompts"] = args.num_prompts

    # Metadata
    if args.metadata:
        for item in args.metadata:
            if "=" in item:
                kvstring = item.split("=", 1)
                result_json[kvstring[0].strip()] = kvstring[1].strip()
            else:
                raise ValueError(
                    "Invalid metadata format. Please use KEY=VALUE format."
                )

    # Traffic
    result_json["request_rate"] = (
        args.request_rate if args.request_rate < float("inf") else "inf"
    )
    result_json["burstiness"] = args.burstiness
    result_json["max_concurrency"] = args.max_concurrency

    if args.ramp_up_strategy is not None:
        result_json["ramp_up_strategy"] = args.ramp_up_strategy
        result_json["ramp_up_start_rps"] = args.ramp_up_start_rps
        result_json["ramp_up_end_rps"] = args.ramp_up_end_rps

    # Merge with benchmark result
    result_json = {**result_json, **benchmark_result}

    # If --save-detailed is not set, remove the response_details field
    if not args.save_detailed:
        if "response_details" in result_json:
            del result_json["response_details"]

    # Save to file
    # Also save when --save-detailed is specified (implies saving results)
    if args.save_result or args.append_result or args.save_detailed:
        base_model_id = model_id.split("/")[-1]
        max_concurrency_str = (
            f"-concurrency{args.max_concurrency}"
            if args.max_concurrency is not None
            else ""
        )
        label = label or args.backend
        # Include dataset name and size in filename for better tracking
        # For custom datasets, extract name from path
        if args.dataset_name == "custom" and args.dataset_path:
            # Get the last part of the path
            dataset_basename = args.dataset_path.rstrip('/').split('/')[-1]
            # Remove file extension if present
            dataset_name_for_file = os.path.splitext(dataset_basename)[0]
        elif args.dataset_name:
            dataset_name_for_file = args.dataset_name
        else:
            dataset_name_for_file = ""

        dataset_str = f"{dataset_name_for_file}-{args.num_prompts}" if dataset_name_for_file else ""
        dataset_prefix = f"{dataset_str}-" if dataset_str else ""

        if args.ramp_up_strategy is not None:
            file_name = f"{label}-{dataset_prefix}ramp-up-{args.ramp_up_strategy}-{args.ramp_up_start_rps}qps-{args.ramp_up_end_rps}qps{max_concurrency_str}-{base_model_id}-{current_dt}.json"  # noqa
        else:
            file_name = f"{label}-{dataset_prefix}{args.request_rate}qps{max_concurrency_str}-{base_model_id}-{current_dt}.json"  # noqa
        if args.result_filename:
            file_name = args.result_filename
        # Expand and absolutize result_dir and filename for copy-paste friendliness
        if args.result_dir:
            result_dir = os.path.abspath(os.path.expanduser(args.result_dir))
            os.makedirs(result_dir, exist_ok=True)
            file_name = os.path.join(result_dir, file_name)
        # Always store and print an absolute path
        file_name = os.path.abspath(os.path.expanduser(file_name))
        with open(
            file_name, mode="a+" if args.append_result else "w", encoding="utf-8"
        ) as outfile:
            # Append a newline.
            if args.append_result and outfile.tell() != 0:
                outfile.write("\n")
            json.dump(result_json, outfile)
        save_to_pytorch_benchmark_format(args, result_json, file_name)

        # Print the saved filename for easy tracking
        print("\n" + "=" * 50)
        print(f"Results saved to: {file_name}")
        # Print companion PyTorch benchmark file as well
        pt_file = f"{os.path.splitext(file_name)[0]}.pytorch.json"
        if os.path.exists(pt_file):
            print(f"PyTorch benchmark saved to: {pt_file}")
        # Quick copy-paste commands for convenience
        print(f"Open in vim: vim {file_name}")
        if os.path.exists(pt_file):
            print(f"Open PT file in vim: vim {pt_file}")
        print("=" * 50 + "\n")

    return result_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark CARA serving throughput and latency."
    )
    add_cli_args(parser)
    args = parser.parse_args()

    # Auto-detect and apply length constraints from dataset filename if not explicitly set
    apply_filename_constraints_to_args(args)

    result = main(args)
    print("\nBenchmark completed successfully!")
