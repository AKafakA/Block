"""Generate requests and send them through the CARA scheduler to collect
latency training data for the XGBoost latency predictor.

Instance state is captured per-request by the sidecar predictor running on
each vLLM node (cara_predictor_api_server). cara_serve must be started with
--enable-predictor-feedback for data collection to work.

Supports two length-sampling modes:
  1. Synthetic: lognormal/uniform/fixed distributions (default)
  2. Real-data: sample prompts from preprocessed training data

Usage:
    # Synthetic lengths
    python -m block.predictor.cara.offline_training.generate_latency_benchmark \
        --host 127.0.0.1 --port 8200 \
        --num-prompts 20000 --request-rate 18 \
        --output latency_data/qps_18.jsonl

    # Real-data prompts from preprocessed training data
    python -m block.predictor.cara.offline_training.generate_latency_benchmark \
        --host 127.0.0.1 --port 8200 \
        --num-prompts 20000 --request-rate 18 \
        --real-data data/cara/training_data/cara_v3_all_training.json \
        --output latency_data/qps_18.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import aiohttp
import numpy as np
from tqdm.asyncio import tqdm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Real-data length sampling
# ---------------------------------------------------------------------------


def sample_real_requests(
    data_path: str,
    num: int,
    rng: np.random.Generator,
    max_tokens: int = 1024,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Sample real prompts and input_lens from training data.

    max_tokens is passed through as the output cap for all requests,
    matching the broadcasting cap so models generate natural responses.

    Returns:
        prompts (list[str]), input_lens, output_lens (numpy arrays, all max_tokens)
    """
    with open(data_path) as f:
        if data_path.endswith(".jsonl"):
            requests = [json.loads(line) for line in f]
        else:
            data = json.load(f)
            requests = data["requests"]
    n_total = len(requests)

    input_lens_all = np.array([req["input_len"] for req in requests])
    prompts_all = [req["prompt"] for req in requests]

    indices = rng.choice(n_total, size=num, replace=True)
    sampled_prompts = [prompts_all[i] for i in indices]
    output_lens = np.full(num, max_tokens, dtype=int)
    return sampled_prompts, input_lens_all[indices], output_lens


def sample_real_lengths(
    data_path: str,
    num: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Legacy: sample only lengths (for synthetic prompt mode)."""
    _, input_lens, output_lens = sample_real_requests(data_path, num, rng)
    return input_lens, output_lens


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

BASE_SENTENCE = (
    "The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs. "
    "How vexingly quick daft zebras jump. "
)


def sample_lengths(
    num: int,
    dist: str,
    mean: float,
    std: float,
    lo: int,
    hi: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return an int array of *num* lengths from the chosen distribution,
    clamped to [lo, hi]."""
    if dist == "fixed":
        lengths = np.full(num, int(mean))
    elif dist == "uniform":
        lengths = rng.integers(lo, hi, endpoint=True, size=num)
    elif dist == "lognormal":
        # For log-normal: if the user specifies mean=256, std=2.0 we
        # interpret mean as the *desired median* of the distribution and
        # std as the sigma of the underlying normal.
        mu = np.log(mean)
        sigma = std
        lengths = np.round(rng.lognormal(mu, sigma, size=num)).astype(int)
    else:
        raise ValueError(f"Unknown distribution: {dist}")

    return np.clip(lengths, lo, hi).astype(int)


def build_dummy_prompt(target_tokens: int, tokenizer: Any) -> str:
    """Create a prompt string that tokenizes to exactly *target_tokens* tokens.

    Strategy: repeat the base sentence enough times to overshoot, tokenize,
    truncate to the exact count, then decode back to text.
    """
    if target_tokens <= 0:
        return ""

    # Estimate ~4 chars per token; overshoot by 2x for safety.
    repeats = max(1, (target_tokens * 8) // len(BASE_SENTENCE) + 2)
    long_text = BASE_SENTENCE * repeats

    token_ids = tokenizer.encode(long_text, add_special_tokens=False)
    if len(token_ids) < target_tokens:
        # Extremely unlikely with 2x overshoot, but handle it.
        extra_repeats = (target_tokens // len(token_ids) + 2)
        long_text = long_text * extra_repeats
        token_ids = tokenizer.encode(long_text, add_special_tokens=False)

    trimmed_ids = token_ids[:target_tokens]
    return tokenizer.decode(trimmed_ids)


# ---------------------------------------------------------------------------
# Request / response data
# ---------------------------------------------------------------------------

@dataclass
class LatencyRecord:
    request_id: str = ""
    input_len: int = 0
    output_len: int = 0          # actual completion tokens
    max_tokens: int = 0          # requested max_tokens
    ttft: float = 0.0            # time to first token (server-reported)
    tpot: float = 0.0            # time per output token (from ITL mean)
    e2el: float = 0.0            # client-side end-to-end latency
    server_latency: float = 0.0  # server-reported E2E
    scheduling_overhead: float = 0.0
    model: str = ""
    host: str = ""
    instance_id: str = ""
    success: bool = False
    error: str = ""
    timestamp: float = 0.0       # request start time (epoch)
    request_rate: float = 0.0    # QPS level for this run


# ---------------------------------------------------------------------------
# Async request sender
# ---------------------------------------------------------------------------

async def send_request(
    session: aiohttp.ClientSession,
    api_url: str,
    prompt: str,
    input_len: int,
    max_tokens: int,
    request_id: str,
    model_name: str = "cara",
) -> LatencyRecord:
    """Send a single completion request and return a LatencyRecord."""

    payload = {
        "request_id": request_id,
        "model": model_name,
        "prompt": prompt,
        "prompt_len": input_len,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "repetition_penalty": 1.0,
        "stream": False,
    }
    headers = {"Content-Type": "application/json"}

    record = LatencyRecord(
        request_id=request_id,
        input_len=input_len,
        max_tokens=max_tokens,
    )

    st = time.perf_counter()
    record.timestamp = time.time()
    try:
        async with session.post(api_url, json=payload, headers=headers) as resp:
            if resp.status == 200:
                data = await resp.json()
                record.e2el = time.perf_counter() - st

                # Detect response format: CARA coordinator vs raw vLLM (OpenAI-compatible)
                if "choices" in data:
                    # Raw vLLM OpenAI-compatible format
                    choices = data.get("choices", [])
                    usage = data.get("usage", {})
                    if choices:
                        record.success = True
                        record.output_len = usage.get("completion_tokens", 0)
                        record.model = data.get("model", "")
                        record.server_latency = record.e2el  # no separate server_latency
                        record.scheduling_overhead = 0.0
                        # vLLM doesn't return TTFT/ITL in non-streaming mode
                        # TPOT estimate: (e2el - estimated_prefill) / output_tokens
                        if record.output_len > 0:
                            record.tpot = record.e2el / record.output_len
                    else:
                        record.error = "Empty choices in vLLM response"

                elif data.get("success", False):
                    # CARA coordinator format
                    record.success = True
                    record.output_len = data.get("output_tokens", 0)
                    record.ttft = data.get("ttft", 0.0)
                    record.model = data.get("model", "")
                    record.instance_id = data.get("instance_id", "")
                    record.host = data.get("host", "")
                    record.server_latency = data.get("server_latency", 0.0)
                    record.scheduling_overhead = record.e2el - record.server_latency
                    # Compute TPOT from ITL if available
                    itl = data.get("itl", [])
                    if itl and len(itl) > 0:
                        record.tpot = sum(itl) / len(itl)  # already in seconds

                else:
                    record.error = data.get("error", "Server returned success=False")
                    record.model = data.get("model", "")
                    record.instance_id = data.get("instance_id", "")
                    record.host = data.get("host", "")
            else:
                record.e2el = time.perf_counter() - st
                try:
                    err_data = await resp.json()
                    record.error = err_data.get("error", f"HTTP {resp.status}")
                except Exception:
                    record.error = f"HTTP {resp.status}: {resp.reason or 'Unknown'}"
    except Exception as exc:
        record.e2el = time.perf_counter() - st
        record.error = f"{type(exc).__name__}: {exc}"

    return record


async def run_benchmark(
    api_url: str,
    prompts: list[str],
    input_lens: np.ndarray,
    output_lens: np.ndarray,
    request_rate: float,
    model_name: str = "cara",
) -> list[LatencyRecord]:
    """Send all requests with Poisson rate limiting. No concurrency cap."""

    num_prompts = len(prompts)
    records: list[LatencyRecord] = []
    lock = asyncio.Lock()

    pbar = tqdm(total=num_prompts, desc="Sending requests")

    async def _task(idx: int) -> None:
        rid = str(uuid.uuid4())
        rec = await send_request(
            session, api_url, prompts[idx],
            int(input_lens[idx]), int(output_lens[idx]), rid,
            model_name=model_name,
        )
        async with lock:
            records.append(rec)
        pbar.update(1)

    timeout = aiohttp.ClientTimeout(total=600)  # 10 min per request
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks: list[asyncio.Task] = []
        start = time.perf_counter()

        for i in range(num_prompts):
            # Rate limiting via Poisson inter-arrival times
            if request_rate < float("inf") and i > 0:
                interval = 1.0 / request_rate
                elapsed = time.perf_counter() - start
                expected = i * interval
                sleep_time = expected - elapsed
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

            task = asyncio.create_task(_task(i))
            tasks.append(task)

        await asyncio.gather(*tasks)

    pbar.close()
    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def print_summary(records: list[LatencyRecord]) -> None:
    successes = [r for r in records if r.success]
    failures = [r for r in records if not r.success]

    print(f"\n{'=' * 60}")
    print(f"  Latency Benchmark Summary")
    print(f"{'=' * 60}")
    print(f"  Total requests:    {len(records)}")
    print(f"  Successful:        {len(successes)}")
    print(f"  Failed:            {len(failures)}")

    if successes:
        e2els = np.array([r.e2el for r in successes])
        ttfts = np.array([r.ttft for r in successes])
        in_lens = np.array([r.input_len for r in successes])
        out_lens = np.array([r.output_len for r in successes])

        print(f"\n  E2E Latency (s):")
        print(f"    mean={e2els.mean():.3f}  p50={np.median(e2els):.3f}  "
              f"p95={np.percentile(e2els, 95):.3f}  p99={np.percentile(e2els, 99):.3f}")
        print(f"  TTFT (s):")
        print(f"    mean={ttfts.mean():.4f}  p50={np.median(ttfts):.4f}  "
              f"p95={np.percentile(ttfts, 95):.4f}")
        print(f"  Input length:")
        print(f"    mean={in_lens.mean():.0f}  min={in_lens.min()}  max={in_lens.max()}")
        print(f"  Output length:")
        print(f"    mean={out_lens.mean():.0f}  min={out_lens.min()}  max={out_lens.max()}")

        # Per-model breakdown
        models = set(r.model for r in successes)
        if len(models) > 1:
            print(f"\n  Per-model breakdown:")
            for m in sorted(models):
                model_recs = [r for r in successes if r.model == m]
                me = np.array([r.e2el for r in model_recs])
                print(f"    {m}: n={len(model_recs)}, "
                      f"e2el mean={me.mean():.3f}s p95={np.percentile(me, 95):.3f}s")

    if failures:
        error_counts: dict[str, int] = {}
        for r in failures:
            key = str(r.error)[:80] if r.error else "unknown"
            error_counts[key] = error_counts.get(key, 0) + 1
        print(f"\n  Error breakdown (top 5):")
        for err, cnt in sorted(error_counts.items(), key=lambda x: -x[1])[:5]:
            print(f"    [{cnt}x] {err}")

    print(f"{'=' * 60}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic latency benchmark data for CARA XGBoost predictor.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Server
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8200)

    # Workload
    parser.add_argument("--num-prompts", type=int, default=20000)
    parser.add_argument("--request-rate", type=float, default=float("inf"),
                        help="Requests per second (inf = no rate limit)")

    # Input length distribution
    parser.add_argument("--input-len-dist", type=str, default="lognormal",
                        choices=["lognormal", "uniform", "fixed"])
    parser.add_argument("--input-len-mean", type=float, default=256)
    parser.add_argument("--input-len-std", type=float, default=2.0)
    parser.add_argument("--input-len-min", type=int, default=32)
    parser.add_argument("--input-len-max", type=int, default=512)

    # Output length distribution
    parser.add_argument("--output-len-dist", type=str, default="lognormal",
                        choices=["lognormal", "uniform", "fixed"])
    parser.add_argument("--output-len-mean", type=float, default=128)
    parser.add_argument("--output-len-std", type=float, default=1.5)
    parser.add_argument("--output-len-min", type=int, default=16)
    parser.add_argument("--output-len-max", type=int, default=512)

    # Real-data mode (overrides synthetic distributions)
    parser.add_argument("--real-data", type=str, default=None,
                        help="Path to preprocessed training JSON; sample (input_len, output_len) "
                             "from actual data distribution instead of synthetic")

    # Model name for the /v1/completions payload (use actual model name for direct vLLM)
    parser.add_argument("--model", type=str, default="cara",
                        help="Model name in API payload. Use 'cara' for coordinator, "
                             "or actual model name (e.g. 'Qwen/Qwen2.5-7B') for direct vLLM")

    # Tokenizer
    parser.add_argument("--tokenizer", type=str, default="Qwen/Qwen2.5-3B")

    # Max tokens cap (should match broadcasting config)
    parser.add_argument("--max-tokens", type=int, default=1024,
                        help="Max output tokens per request (should match broadcasting cap)")

    # Output
    parser.add_argument("--output", type=str, required=True,
                        help="Path to output JSONL file")

    # Misc
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    rng = np.random.default_rng(args.seed)

    # ---- 1. Sample requests ----
    real_prompts: list[str] | None = None
    if args.real_data:
        logger.info("Sampling %d real requests from: %s",
                     args.num_prompts, args.real_data)
        real_prompts, input_lens, output_lens = sample_real_requests(
            args.real_data, args.num_prompts, rng, max_tokens=args.max_tokens
        )
    else:
        logger.info("Sampling %d (input_len, output_len) from synthetic distributions ...",
                     args.num_prompts)
        input_lens = sample_lengths(
            args.num_prompts,
            args.input_len_dist,
            args.input_len_mean,
            args.input_len_std,
            args.input_len_min,
            args.input_len_max,
            rng,
        )
        output_lens = sample_lengths(
            args.num_prompts,
            args.output_len_dist,
            args.output_len_mean,
            args.output_len_std,
            args.output_len_min,
            args.output_len_max,
            rng,
        )

    logger.info(
        "Input lengths: mean=%.0f, min=%d, max=%d | Output lengths: mean=%.0f, min=%d, max=%d",
        input_lens.mean(), input_lens.min(), input_lens.max(),
        output_lens.mean(), output_lens.min(), output_lens.max(),
    )

    # ---- 2. Build prompts ----
    if real_prompts is not None:
        logger.info("Using %d real prompts from training data.", len(real_prompts))
        prompts = real_prompts
    else:
        logger.info("Loading tokenizer %s ...", args.tokenizer)
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

        logger.info("Generating %d dummy prompts ...", args.num_prompts)
        prompt_cache: dict[int, str] = {}
        prompts: list[str] = []
        for ilen in input_lens:
            ilen = int(ilen)
            if ilen not in prompt_cache:
                prompt_cache[ilen] = build_dummy_prompt(ilen, tokenizer)
            prompts.append(prompt_cache[ilen])
        logger.info("Cached %d unique prompt lengths.", len(prompt_cache))

    # ---- 3. Send requests ----
    api_url = f"http://{args.host}:{args.port}/v1/completions"
    logger.info(
        "Sending %d requests to %s (rate=%.1f rps) ...",
        args.num_prompts, api_url, args.request_rate,
    )

    # ---- 4. Run benchmark ----
    records = asyncio.run(run_benchmark(
        api_url, prompts, input_lens, output_lens,
        args.request_rate,
        model_name=args.model,
    ))

    # Tag each record with request_rate
    for rec in records:
        rec.request_rate = args.request_rate

    # ---- 5. Save results ----
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        for rec in records:
            f.write(json.dumps(asdict(rec)) + "\n")

    logger.info("Saved %d latency records to %s", len(records), out_path)

    # ---- 6. Print summary ----
    print_summary(records)


if __name__ == "__main__":
    main()
