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
from block.server_utils import serve_http
from block.global_scheduler.cara.utils import set_ulimit
import resource
import logging
import traceback

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
broadcasting_enabled = False
broadcast_model_list: list[str] = []
enable_predictor_feedback = False

# Lightweight health endpoint for readiness probes
@app.get("/health")
async def health() -> Response:
    return JSONResponse(content={"status": "ok"})


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
    if chat:
        # Use /v1/chat/completions endpoint with messages format
        # This is cleaner and lets vLLM handle chat template automatically
        # Signal to vllm_instance to use chat endpoint
        request_json["use_chat_endpoint"] = True
        # Remove prompt since we're using messages

        # Respect client-provided repetition_penalty; only apply server default if absent
        if "repetition_penalty" not in request_json or request_json["repetition_penalty"] is None:
            request_json["repetition_penalty"] = float(repetition_penalty)
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

        # Broadcasting mode: query one instance per selected model, pick one as main response
        # Non-broadcasting mode: use random/round-robin selection
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

            # Pick at most one instance per requested model
            chosen: dict[str, Instance] = {}
            for inst in instances:
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
            # Non-broadcasting mode: use existing random/round-robin selection
            if scheduling == "random":
                selected_instance = random.choice(instances)
            elif scheduling == "round_robin":
                selected_instance = instances[num_requests % len(instances)]
            else:
                selected_instance = random.choice(instances)

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


def build_app(args: Namespace) -> FastAPI:
    global app
    app.root_path = args.root_path
    return app


async def init_app(
        args: Namespace,
        instances_list: Optional[List[Instance]] = None,
) -> FastAPI:
    app = build_app(args)
    global instances, start_time, scheduling, chat, model_family, repetition_penalty, broadcasting_enabled, broadcast_model_list, enable_predictor_feedback
    chat = args.chat
    model_family = args.model_family
    repetition_penalty = args.repetition_penalty
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
                        help="Scheduling strategy among instances: random, round_robin")
    parser.add_argument("--debugging_logs", action="store_true",
                        help="Enable debug level logging")
    parser.add_argument("--chat", action="store_true",
                        help="Whether the model is a chat model to decide if stop words are appended")
    parser.add_argument("--model-family", type=str, default="Qwen",
                        help="Model family, used for append the stop words for chat models")
    parser.add_argument("--repetition-penalty", type=float, default=1.1,
                        help="Repetition penalty to use for generation to avoid repetition")
    parser.add_argument("--enable-predictor-feedback", action="store_true",
                        help="Enable sending actual metrics back to predictor for training data collection")
    parser.add_argument("--feedback-sample-rate", type=float, default=1.0,
                        help="Sampling rate for predictor feedback (0.0 to 1.0). Only applies when --enable-predictor-feedback is set")
    parser.add_argument("--broadcasting", action="store_true",
                        help="Enable broadcasting to one instance per selected model for data collection")
    parser.add_argument("--selected-broadcasted-models", nargs='+', default=[],
                        help="List of model names/tags to broadcast to when broadcasting is enabled")
    args = parser.parse_args()
    logger.info("Starting server with args: %s", str(args))

    asyncio.run(run_server(args))
