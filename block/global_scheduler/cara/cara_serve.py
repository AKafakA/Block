import argparse
import asyncio
import json
import random
import ssl
import time
from argparse import Namespace
import aiohttp
from typing import Any, Optional, List
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from block.global_scheduler.cara.cara_instance.Instance import Instance
from block.server_utils_lite import serve_http
from block.global_scheduler.cara.utils import set_ulimit
import resource
import logging
import traceback
import math

STOP_WORD_MAPS = {
    "Qwen": ["<|im_start|>", "<|im_end|>"]
}
TIMEOUT_KEEP_ALIVE = 5  # seconds.
app = FastAPI()
instances = []
num_requests = 0
start_time = 0
scheduling = "random"
served_requests = []
logging.basicConfig(level=logging.INFO,
                    filemode='a+',
                    filename='experiment_output/logs/cara_serve.log')
logger = logging.getLogger(__name__)
chat = False
model_family = "Qwen"
repetition_penalty = 1.0
frequency_penalty = 1.2
temperature = 0.0
broadcasting_enabled = False
broadcast_model_list: list[str] = []
enable_predictor_feedback = False
# Learned predictor instances (instance_id -> CARALearnedPredictor)
learned_predictors: dict = {}
# Scheduling config for multi-objective (loaded from predictor config)
scoring_weights: dict = {"w_latency": 0.3, "w_cost": 0.3, "w_quality": 0.4}
slo_defaults: dict = {
    "ttft_slo_ms": 5000, "tpot_slo_ms": 200,
    "budget_tokens": 256, "quality_min": 0.3,
    "budget_confidence_threshold": 0.5,
}
# Instance metadata (instance_id -> {model_name, gpu_type, cost_per_token, instance_type})
instance_meta: dict = {}


def to_ollama_tag(hf_name: str) -> str:
    """Convert HuggingFace model name to Ollama tag format.
    Example: 'Qwen/Qwen2.5-3B' -> 'qwen2.5:3b'
    """
    name = hf_name.lower()
    if "/" in name:
        name = name.split("/")[-1]
    name = name.replace("-", ":")
    return name



# align with vllm bench so we can directly leverage existing tools
@app.post("/v1/completions")
async def completion(request: Request) -> Response:
    global num_requests
    request_json = await request.json()
    num_requests += 1
    request_id = request_json.get("request_id", str(num_requests))
    request_json["request_id"] = request_id
    served_requests.append(request_id)
    selected_instance = None
    # Inject server-side sampling parameters into the request payload.
    # These override any client-provided values to ensure consistent generation
    # across all experiments (broadcasting, latency collection, online serving).
    request_json["repetition_penalty"] = float(repetition_penalty)
    request_json["frequency_penalty"] = float(frequency_penalty)
    request_json["temperature"] = float(temperature)

    if chat:
        # Use /v1/chat/completions endpoint with messages format
        # This is cleaner and lets vLLM handle chat template automatically
        # Signal to vllm_instance to use chat endpoint
        request_json["use_chat_endpoint"] = True
    try:
        # CARA: Query predictors before scheduling (for training data collection)
        # Only query predictors if feedback is enabled
        num_prompt_tokens = request_json.get("prompt_len", 0)
        max_output_tokens = request_json.get("max_tokens", 256)

        if enable_predictor_feedback:
            # Build predicted_num_context_tokens dict for all models
            predicted_num_context_tokens = {}
            for instance in instances:
                predicted_num_context_tokens[instance._model_name] = max_output_tokens

            # Query all instance predictors (for training data collection)
            prediction_tasks = []
            for instance in instances:
                prediction_task = instance.query_predictor(
                    request_id=request_id,
                    num_context_tokens=num_prompt_tokens,
                    predicted_num_context_tokens=predicted_num_context_tokens
                )
                prediction_tasks.append(prediction_task)

            # Wait for all predictions (run in parallel)
            try:
                predictions = await asyncio.gather(*prediction_tasks, return_exceptions=True)
                logger.debug(f"Request {request_id}: Collected {len(predictions)} predictions")
            except Exception as e:
                logger.warning(f"Request {request_id}: Predictor query failed: {e}")
                predictions = []
        else:
            logger.debug(f"Request {request_id}: Skipping predictor queries (feedback disabled)")

        # Extract optional RSO (Request Service Objectives) from request
        rso = request_json.get("rso", {})
        budget_tokens = rso.get("budget", slo_defaults.get("budget_tokens", 256))
        ttft_slo_ms = rso.get("ttft_slo_ms", slo_defaults.get("ttft_slo_ms", 5000))
        tpot_slo_ms = rso.get("tpot_slo_ms", slo_defaults.get("tpot_slo_ms", 200))
        quality_min = rso.get("quality_min", slo_defaults.get("quality_min", 0.0))
        prompt_text = request_json.get("prompt", "")

        # Broadcasting mode: query one instance per selected model, pick one as main response
        # Non-broadcasting mode: use scheduling strategy selection
        if broadcasting_enabled and broadcast_model_list:
            # Normalize model names for matching (support HF name or Ollama tag)
            def _norm(name: str) -> str:
                return name.strip().lower()

            # Build a set of normalized targets including possible tag forms
            target_norms = set()
            for m in broadcast_model_list:
                try:
                    tag = to_ollama_tag(m)
                except Exception:
                    tag = m
                target_norms.add(_norm(m))
                target_norms.add(_norm(tag))

            # Pick at most one instance per requested model, rotating to spread load
            chosen: dict[str, Instance] = {}
            shuffled = list(instances)
            random.shuffle(shuffled)
            for inst in shuffled:
                model_key = _norm(inst._model_name)
                if model_key in target_norms and inst._model_name not in chosen:
                    chosen[inst._model_name] = inst

            # Launch queries to all chosen instances in parallel
            tasks = []
            for model_name, inst in chosen.items():
                # Avoid mutating original payload across concurrent requests
                payload_copy = json.loads(json.dumps(request_json))
                tasks.append(inst.query_instance(
                    payload_copy,
                    predicted_num_decode_tokens=max_output_tokens
                ))

            if tasks:
                try:
                    broadcast_results = await asyncio.gather(*tasks, return_exceptions=True)
                    # Filter out exceptions
                    broadcast_results = [res for res in broadcast_results if not isinstance(res, Exception)]

                    if broadcast_results:
                        # Randomly pick one as the main response (make a copy to avoid circular reference)
                        selected_response = random.choice(broadcast_results)
                        response_dict = dict(selected_response)
                        # Include all results in broadcast_results
                        response_dict["broadcast_results"] = broadcast_results
                    else:
                        # All broadcast queries failed
                        response_dict = {
                            "success": False,
                            "error": "All broadcast queries failed",
                            "request_id": request_id,
                        }
                except Exception as e:
                    response_dict = {
                        "success": False,
                        "error": f"Broadcasting failed: {str(e)}",
                        "request_id": request_id,
                    }
            else:
                # No instances matched the broadcast model list
                response_dict = {
                    "success": False,
                    "error": f"No instances found for broadcast models: {broadcast_model_list}",
                    "request_id": request_id,
                }
        else:
            # Non-broadcasting mode: use scheduling strategy
            selected_instance = await _select_instance(
                scheduling, instances, num_requests, request_json,
                prompt_text, num_prompt_tokens, max_output_tokens,
                budget_tokens, ttft_slo_ms, tpot_slo_ms, quality_min,
                request_id,
            )

            response_dict = await selected_instance.query_instance(
                request_json,
                predicted_num_decode_tokens=max_output_tokens
            )

        return JSONResponse(content=response_dict)
    except Exception as e:
        logger.error(f"Error processing request {request_id}: {e}")
        logger.error(f"Instance: {selected_instance._instance_id if selected_instance else 'None'}")
        logger.error(f"Host: {selected_instance._hostname if selected_instance else 'Unknown'}")
        logger.error(f"Model: {selected_instance._model_name if selected_instance else 'Unknown'}")
        logger.error(traceback.format_exc())

        # Include debugging info in the error response
        error_response = {
            "success": False,
            "error": str(e),
            "error_traceback": traceback.format_exc(),
            "request_id": request_id,
            "instance_id": selected_instance._instance_id if selected_instance else "Unknown",
            "host": selected_instance._hostname if selected_instance else "Unknown",
            "model": selected_instance._model_name if selected_instance else "Unknown",
        }
        return JSONResponse(content=error_response, status_code=500)


async def _get_instance_stats(instance: Instance) -> dict:
    """Query /instance_stats from a vLLM instance. Returns empty dict on failure."""
    try:
        url = f"http://{instance._ip_address}:{instance._backend_port}/instance_stats"
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=3)
        ) as session:
            async with session.get(url, ssl=False) as resp:
                if resp.status == 200:
                    return await resp.json()
    except Exception:
        pass
    return {}


async def _select_instance(
    strategy: str,
    all_instances: list,
    req_count: int,
    request_json: dict,
    prompt_text: str,
    num_prompt_tokens: int,
    max_output_tokens: int,
    budget_tokens: int,
    ttft_slo_ms: float,
    tpot_slo_ms: float,
    quality_min: float,
    request_id: str,
) -> Instance:
    """Select an instance based on the scheduling strategy."""

    if strategy == "random":
        return random.choice(all_instances)

    elif strategy == "round_robin":
        return all_instances[req_count % len(all_instances)]

    elif strategy == "shortest_queue":
        return await _schedule_shortest_queue(all_instances)

    elif strategy == "quality_greedy":
        return await _schedule_quality_greedy(all_instances)

    elif strategy == "cost_greedy":
        return await _schedule_cost_greedy(all_instances)

    elif strategy == "length_aware":
        return await _schedule_length_aware(
            all_instances, prompt_text, num_prompt_tokens, max_output_tokens
        )

    elif strategy == "cara":
        return await _schedule_cara(
            all_instances, prompt_text, num_prompt_tokens, max_output_tokens,
            budget_tokens, ttft_slo_ms, tpot_slo_ms, quality_min, request_id,
        )

    else:
        logger.warning(f"Unknown scheduling strategy '{strategy}', falling back to random")
        return random.choice(all_instances)


async def _schedule_shortest_queue(all_instances: list) -> Instance:
    """Pick instance with fewest pending requests."""
    stats_tasks = [_get_instance_stats(inst) for inst in all_instances]
    stats_list = await asyncio.gather(*stats_tasks, return_exceptions=True)

    best_inst = all_instances[0]
    best_queue = float("inf")

    for inst, stats in zip(all_instances, stats_list):
        if isinstance(stats, Exception) or not stats:
            queue_len = inst.total_request  # fallback to total requests served
        else:
            queue_len = stats.get("num_running", 0) + stats.get("num_waiting", 0)
        if queue_len < best_queue:
            best_queue = queue_len
            best_inst = inst

    return best_inst


def _model_size_key(model_name: str) -> int:
    """Extract numeric model size from name for ordering (larger = higher)."""
    import re
    # Match patterns like "3B", "7B", "14B", "72B"
    match = re.search(r'(\d+)[Bb]', model_name)
    if match:
        return int(match.group(1))
    return 0


async def _schedule_quality_greedy(all_instances: list) -> Instance:
    """Pick largest model with shortest queue (quality-maximizing)."""
    # Group by model size, pick largest group
    by_size: dict[int, list] = {}
    for inst in all_instances:
        size = _model_size_key(inst._model_name)
        by_size.setdefault(size, []).append(inst)

    # Try from largest to smallest
    for size in sorted(by_size.keys(), reverse=True):
        group = by_size[size]
        if group:
            # Shortest queue within the group
            stats_tasks = [_get_instance_stats(inst) for inst in group]
            stats_list = await asyncio.gather(*stats_tasks, return_exceptions=True)

            best_inst = group[0]
            best_queue = float("inf")
            for inst, stats in zip(group, stats_list):
                if isinstance(stats, Exception) or not stats:
                    queue_len = 999
                else:
                    queue_len = stats.get("num_running", 0) + stats.get("num_waiting", 0)
                if queue_len < best_queue:
                    best_queue = queue_len
                    best_inst = inst
            return best_inst

    return random.choice(all_instances)


async def _schedule_cost_greedy(all_instances: list) -> Instance:
    """Pick smallest model with shortest queue (cost-minimizing)."""
    by_size: dict[int, list] = {}
    for inst in all_instances:
        size = _model_size_key(inst._model_name)
        by_size.setdefault(size, []).append(inst)

    # Try from smallest to largest
    for size in sorted(by_size.keys()):
        group = by_size[size]
        if group:
            stats_tasks = [_get_instance_stats(inst) for inst in group]
            stats_list = await asyncio.gather(*stats_tasks, return_exceptions=True)

            best_inst = group[0]
            best_queue = float("inf")
            for inst, stats in zip(group, stats_list):
                if isinstance(stats, Exception) or not stats:
                    queue_len = 999
                else:
                    queue_len = stats.get("num_running", 0) + stats.get("num_waiting", 0)
                if queue_len < best_queue:
                    best_queue = queue_len
                    best_inst = inst
            return best_inst

    return random.choice(all_instances)


async def _schedule_length_aware(
    all_instances: list, prompt_text: str,
    num_prompt_tokens: int, max_output_tokens: int,
) -> Instance:
    """Use length prediction to route to instance that minimizes estimated
    completion time (like L4/S3 baselines)."""
    # Get predictions from learned predictors if available
    stats_tasks = [_get_instance_stats(inst) for inst in all_instances]
    stats_list = await asyncio.gather(*stats_tasks, return_exceptions=True)

    best_inst = all_instances[0]
    best_time = float("inf")

    for inst, stats in zip(all_instances, stats_list):
        if isinstance(stats, Exception) or not stats:
            stats = {}

        inst_id = inst._instance_id
        predictor = learned_predictors.get(inst_id)

        if predictor and prompt_text:
            # Use learned predictor for length estimate
            pred = predictor.predict_full(
                prompt_text, num_prompt_tokens, max_output_tokens,
                schedule_state=stats,
            )
            expected_length = pred["expected_length"]
            tpot = pred["tpot"]
            ttft = pred["ttft"]
        else:
            # Fallback: assume max output tokens
            expected_length = max_output_tokens
            tpot = 0.05
            ttft = 1.0

        # Estimated completion = TTFT + expected_length * TPOT + queue wait
        queue_len = stats.get("num_running", 0) + stats.get("num_waiting", 0)
        queue_wait = queue_len * expected_length * tpot  # rough estimate
        est_time = ttft + expected_length * tpot + queue_wait

        if est_time < best_time:
            best_time = est_time
            best_inst = inst

    return best_inst


async def _schedule_cara(
    all_instances: list, prompt_text: str,
    num_prompt_tokens: int, max_output_tokens: int,
    budget_tokens: int, ttft_slo_ms: float, tpot_slo_ms: float,
    quality_min: float, request_id: str,
) -> Instance:
    """Full multi-objective CARA scheduling.

    1. Filter: P(tokens<=budget) < threshold → reject
               predicted_ttft > SLO → reject
               tpot > SLO → reject
    2. Rank: score = w_lat * ttft_norm + w_cost * E[length] * cost_per_tok - w_qual * quality
       Lower score is better.
    """
    w_lat = scoring_weights.get("w_latency", 0.3)
    w_cost = scoring_weights.get("w_cost", 0.3)
    w_qual = scoring_weights.get("w_quality", 0.4)
    budget_threshold = slo_defaults.get("budget_confidence_threshold", 0.5)

    # Get instance stats in parallel
    stats_tasks = [_get_instance_stats(inst) for inst in all_instances]
    stats_list = await asyncio.gather(*stats_tasks, return_exceptions=True)

    candidates = []

    for inst, stats in zip(all_instances, stats_list):
        if isinstance(stats, Exception) or not stats:
            stats = {}

        inst_id = inst._instance_id
        predictor = learned_predictors.get(inst_id)

        if predictor and prompt_text:
            pred = predictor.predict_full(
                prompt_text, num_prompt_tokens, max_output_tokens,
                schedule_state=stats, budget_tokens=budget_tokens,
            )
        else:
            # Fallback for instances without learned predictors
            meta = instance_meta.get(inst_id, {})
            pred = {
                "p_under_budget": 0.5,
                "ttft": 1.0,
                "tpot": 0.05,
                "quality": 0.5,
                "expected_length": max_output_tokens,
                "cost_per_token": meta.get("cost_per_token", 0.01),
            }

        # --- Filter stage ---
        # 1. Budget compliance
        if pred["p_under_budget"] < budget_threshold:
            logger.debug(
                f"Request {request_id}: {inst_id} filtered (budget compliance "
                f"{pred['p_under_budget']:.2f} < {budget_threshold})"
            )
            continue

        # 2. TTFT SLO
        predicted_ttft_ms = pred["ttft"] * 1000
        if predicted_ttft_ms > ttft_slo_ms:
            logger.debug(
                f"Request {request_id}: {inst_id} filtered (TTFT "
                f"{predicted_ttft_ms:.0f}ms > {ttft_slo_ms}ms)"
            )
            continue

        # 3. TPOT SLO
        predicted_tpot_ms = pred["tpot"] * 1000
        if predicted_tpot_ms > tpot_slo_ms:
            logger.debug(
                f"Request {request_id}: {inst_id} filtered (TPOT "
                f"{predicted_tpot_ms:.0f}ms > {tpot_slo_ms}ms)"
            )
            continue

        # 4. Quality minimum
        if pred["quality"] < quality_min:
            logger.debug(
                f"Request {request_id}: {inst_id} filtered (quality "
                f"{pred['quality']:.2f} < {quality_min})"
            )
            continue

        # --- Scoring ---
        # Normalize TTFT to [0, 1] range (max 10s)
        ttft_norm = min(pred["ttft"] / 10.0, 1.0)
        # Cost component
        cost = pred["expected_length"] * pred["cost_per_token"]
        # Normalize cost (assume max ~$0.05 per request)
        cost_norm = min(cost / 0.05, 1.0)
        # Quality is already in [0, 1]
        quality_norm = pred["quality"]

        # Lower is better: latency and cost are penalties, quality is reward
        score = w_lat * ttft_norm + w_cost * cost_norm - w_qual * quality_norm

        candidates.append((score, inst, pred))

    if not candidates:
        # All filtered out — fall back to shortest queue
        logger.warning(
            f"Request {request_id}: all instances filtered by CARA, "
            f"falling back to shortest_queue"
        )
        return await _schedule_shortest_queue(all_instances)

    # Pick the candidate with the lowest score
    candidates.sort(key=lambda x: x[0])
    best_score, best_inst, best_pred = candidates[0]

    logger.debug(
        f"Request {request_id}: CARA selected {best_inst._instance_id} "
        f"(score={best_score:.3f}, quality={best_pred['quality']:.2f}, "
        f"ttft={best_pred['ttft']:.2f}s, E[len]={best_pred['expected_length']:.0f})"
    )

    return best_inst


def _load_learned_predictors(config_path: str, all_instances: list):
    """Load learned predictors for CARA multi-objective scheduling."""
    global learned_predictors, scoring_weights, slo_defaults, instance_meta

    try:
        with open(config_path) as f:
            config_dict = json.load(f)

        from block.predictor.cara.cara_predictor_config import CARABasePredictorConfig
        config = CARABasePredictorConfig.from_dict(config_dict)

        # Update global scoring weights and SLO defaults
        scoring_weights.update(config.scoring_weights)
        slo_defaults.update(config.slo_defaults)

        # Build instance_type mapping from instance metadata
        inst_metadata = config.instance_metadata

        for inst in all_instances:
            # Try to determine instance type from model name + metadata
            for inst_type, meta in inst_metadata.items():
                if meta.get("model_name") == inst._model_name:
                    instance_meta[inst._instance_id] = {
                        **meta, "instance_type": inst_type
                    }

                    # Create learned predictor for this instance
                    try:
                        from block.predictor.cara.cara_learned_predictor import CARALearnedPredictor
                        predictor = CARALearnedPredictor(
                            config=config,
                            port=inst._backend_port,
                            hostname=inst._hostname,
                            instance_type=inst_type,
                        )
                        learned_predictors[inst._instance_id] = predictor
                        logger.info(
                            f"Loaded learned predictor for {inst._instance_id} "
                            f"(type={inst_type})"
                        )
                    except Exception as e:
                        logger.warning(
                            f"Failed to load learned predictor for {inst._instance_id}: {e}"
                        )
                    break

        logger.info(
            f"Loaded {len(learned_predictors)} learned predictors, "
            f"scoring_weights={scoring_weights}"
        )

    except Exception as e:
        logger.error(f"Failed to load learned predictor config: {e}")


def build_app(args: Namespace) -> FastAPI:
    global app
    app.root_path = args.root_path
    return app


async def init_app(
        args: Namespace,
        instances_list: Optional[List[Instance]] = None,
) -> FastAPI:
    app = build_app(args)
    global instances, start_time, scheduling, chat, model_family, repetition_penalty, frequency_penalty, temperature, broadcasting_enabled, broadcast_model_list, enable_predictor_feedback
    chat = args.chat
    model_family = args.model_family
    repetition_penalty = args.repetition_penalty
    frequency_penalty = args.frequency_penalty
    temperature = args.temperature
    model_config_path = args.model_config_path
    broadcasting_enabled = bool(getattr(args, "broadcasting", False))
    broadcast_model_list = list(getattr(args, "selected_broadcasted_models", []) or [])
    enable_predictor_feedback = args.enable_predictor_feedback

    model_dict = json.load(open(model_config_path))
    host_config = json.load(open(args.host_config))
    if instances_list is not None:
        instances.extend(instances_list)
    else:
        for model, model_config in model_dict.items():
            node_hosts = model_config["node_hosts"]
            backend_type = model_config["backend"]
            hf_model_name = model_config["hf_model_name"]
            for idx, host in enumerate(node_hosts):
                # Extract hostname from "user@hostname" format
                hostname = host.split("@")[-1] if "@" in host else host
                ip_address = host_config[hostname]["ip_address"]
                predictor_ports = host_config[hostname]["predictor_ports"]
                backend_port = host_config[hostname]["backend_port"]
                # Avoid including the backend port in predictor ports
                predictor_ports = [p for p in predictor_ports if p != backend_port]
                instance_id = f"{model}_{idx}"
                if backend_type == "ollama":
                    from block.global_scheduler.cara.cara_instance.ollama_instance import OllamaInstance
                    # Convert HF model name to Ollama tag format, not supposed to be used besides testing
                    ollama_model_name = to_ollama_tag(hf_model_name)
                    instance = OllamaInstance(
                        instance_id=instance_id,
                        hostname=hostname,
                        ip_address=ip_address,
                        predictor_ports=predictor_ports,
                        model_name=ollama_model_name,
                        backend_port=backend_port,
                        enable_predictor_feedback=args.enable_predictor_feedback,
                        feedback_sample_rate=args.feedback_sample_rate
                    )
                elif backend_type == "vllm":
                    from block.global_scheduler.cara.cara_instance.vllm_instance import VllmInstance
                    instance = VllmInstance(
                        instance_id=instance_id,
                        hostname=hostname,
                        ip_address=ip_address,
                        predictor_ports=predictor_ports,
                        model_name=hf_model_name,
                        backend_port=backend_port,
                        enable_predictor_feedback=args.enable_predictor_feedback,
                        feedback_sample_rate=args.feedback_sample_rate
                    )
                else:
                    raise ValueError(f"Unsupported backend type: {backend_type}")
                instances.append(instance)

    start_time = time.time()
    scheduling = args.scheduling

    # Load learned predictors if using cara/length_aware scheduling
    if scheduling in ("cara", "length_aware") and getattr(args, "predictor_config", None):
        _load_learned_predictors(args.predictor_config, instances)

    # Log registered instances and routes
    logger.info(f"CARA Scheduler initialized with {len(instances)} instances:")
    for inst in instances:
        logger.info(f"  - {inst._instance_id} ({inst._model_name}) @ {inst._ip_address}")
    logger.info(f"Scheduling strategy: {scheduling}")
    logger.info("Registered routes:")
    for route in app.routes:
        if hasattr(route, 'path') and hasattr(route, 'methods'):
            logger.info(f"  {list(route.methods)[0] if route.methods else 'GET'} {route.path}")

    return app


async def run_server(args: Namespace,
                     instances_list: Optional[List[Instance]] = None,
                     **uvicorn_kwargs: Any) -> None:
    app = await init_app(args, instances_list)
    print(str(set_ulimit()) + " set limits file ")
    assert len(instances) > 0

    if args.debugging_logs:
        logger.setLevel(logging.DEBUG)

    try:
        shutdown_task = await serve_http(
            app,
            host=args.host,
            port=args.port,
            timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
            ssl_keyfile=args.ssl_keyfile,
            ssl_certfile=args.ssl_certfile,
            ssl_ca_certs=args.ssl_ca_certs,
            ssl_cert_reqs=args.ssl_cert_reqs,
            workers=args.workers,
            **uvicorn_kwargs,
        )

        await shutdown_task
    finally:
        logger.info("Server shutdown.")
        # Flush training data from all predictors across instances (only if feedback is enabled)
        if not args.enable_predictor_feedback:
            logger.info("Predictor feedback disabled, skipping flush on shutdown.")
            return

        try:
            flush_urls = []
            for inst in instances:
                flush_urls.extend(inst.get_predictor_flush_urls())

            if flush_urls:
                timeout = aiohttp.ClientTimeout(total=15)

                async def _post_flush(session, url):
                    try:
                        async with session.post(url, ssl=False) as resp:
                            await resp.text()
                            return resp.status
                    except Exception:
                        return None

                async with aiohttp.ClientSession(timeout=timeout) as session:
                    results = await asyncio.gather(*[_post_flush(session, u) for u in flush_urls], return_exceptions=True)
                success = sum(1 for r in results if isinstance(r, int) and 200 <= r < 300)
                failed = len(flush_urls) - success
                logger.info(f"Flushed predictors: {success} ok, {failed} failed")
        except Exception as e:
            logger.warning(f"Error flushing predictors on shutdown: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=8200)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--ssl-keyfile", type=str, default=None)
    parser.add_argument("--ssl-certfile", type=str, default=None)
    parser.add_argument("--ssl-ca-certs",
                        type=str,
                        default=None,
                        help="The CA certificates file")
    parser.add_argument(
        "--ssl-cert-reqs",
        type=int,
        default=int(ssl.CERT_NONE),
        help="Whether client certificate is required (see stdlib ssl module's)"
    )
    parser.add_argument(
        "--root-path",
        type=str,
        default=None,
        help="FastAPI root_path when app is behind a path based routing proxy")
    parser.add_argument("--model_config_path", type=str,
                        default="block/config/cara/model_deployment.json")
    parser.add_argument("--host_config", type=str,
                        default="block/config/host_configs.json")
    parser.add_argument("--scheduling", type=str, default="random",
                        help="Scheduling strategy: random, round_robin, shortest_queue, "
                             "quality_greedy, cost_greedy, length_aware, cara")
    parser.add_argument("--debugging_logs", action="store_true",
                        help="Enable debug level logging")
    parser.add_argument("--chat", action="store_true",
                        help="Whether the model is a chat model to decide if stop words are appended")
    parser.add_argument("--model-family", type=str, default="Qwen",
                        help="Model family, used for append the stop words for chat models")
    parser.add_argument("--repetition-penalty", type=float, default=1.0,
                        help="Repetition penalty (multiplicative, 1.0 = no penalty). "
                             "Applied to previously generated tokens regardless of frequency.")
    parser.add_argument("--frequency-penalty", type=float, default=1.2,
                        help="Frequency penalty (additive, proportional to token count, default=1.2). "
                             "Penalizes tokens based on how many times they appear in the output so far. "
                             "Prevents degenerate repetition and keeps output_tokens under max_tokens. "
                             "Tuned via sweep: best survival rate at 1.2 (see sweep_broadcasting.sh).")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (0.0 = greedy deterministic, >0 adds randomness)")
    parser.add_argument("--enable-predictor-feedback", action="store_true",
                        help="Enable sending actual metrics back to predictor for training data collection")
    parser.add_argument("--feedback-sample-rate", type=float, default=1.0,
                        help="Sampling rate for predictor feedback (0.0 to 1.0). Only applies when --enable-predictor-feedback is set")
    parser.add_argument("--broadcasting", action="store_true",
                        help="Enable broadcasting to one instance per selected model for data collection")
    parser.add_argument("--selected-broadcasted-models", nargs='+', default=[],
                        help="List of model names/tags to broadcast to when broadcasting is enabled")
    parser.add_argument("--predictor-config", type=str, default=None,
                        help="Path to learned predictor config JSON (for cara/length_aware scheduling)")
    args = parser.parse_args()
    logger.info("Starting server with args: %s", str(args))

    asyncio.run(run_server(args))
