import argparse
import asyncio
import json
import random
import ssl
import time
from argparse import Namespace
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
                    filename='experiment_output/logs/predictor_output.log')
logger = logging.getLogger(__name__)
backend_port_map = {
    "ollama": 11434,
    "vllm": 8000
}
chat = False
model_family = "Qwen"


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
    request_id = request_json.get("request_id")
    served_requests.append(request_id)
    selected_instance = None
    if chat:
        # Append stop words based on model family
        start_word = STOP_WORD_MAPS[model_family][0]
        stop_word = STOP_WORD_MAPS[model_family][1]

        system_prompt = f"{start_word}system\nYou are a helpful assistant.{stop_word}\n"
        user_part = f"{start_word}user\n{request_json['prompt']}{stop_word}\n"
        assistant_trigger = f"{start_word}assistant\n"

        # Combine them
        request_json["prompt"] = system_prompt + user_part + assistant_trigger

        request_json["stop"] = [stop_word, "<|endoftext|>", start_word]
        request_json["repetition_penalty"] = float(request_json.get("repetition_penalty", 1.0))
    try:
        if scheduling == "random":
            selected_instance = random.choice(instances)
        elif scheduling == "round_robin":
            selected_instance = instances[num_requests % len(instances)]
        else:
            selected_instance = random.choice(instances)

        response_dict = await selected_instance.query_instance(
            request_json,
            # useless for now, leave for future extension
            predicted_num_decode_tokens=0
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
    global instances, start_time, scheduling, chat, model_family
    chat = args.chat
    model_family = args.model_family
    model_config_path = args.model_config_path

    model_dict = json.load(open(model_config_path))
    host_config = json.load(open(args.host_config))
    if instances_list is not None:
        instances.extend(instances_list)
    else:
        for model, model_config in model_dict.items():
            node_hosts = model_config["node_hosts"]
            backend_type = model_config["backend"]
            hf_model_name = model_config["hf_model_name"]
            # use the hard-coded port here and ignore the one in host_config as it was generated for Block
            # with homogeneous backend
            # TODO: rewrite the config generation script for Cara later and merge the two configs
            backend_port = backend_port_map.get(backend_type, 8000)
            for idx, host in enumerate(node_hosts):
                # Extract hostname from "user@hostname" format
                hostname = host.split("@")[-1] if "@" in host else host
                ip_address = host_config[hostname]["ip_address"]
                predictor_ports = host_config[hostname]["predictor_ports"]
                instance_id = f"{model}_{idx}"
                if backend_type == "ollama":
                    from block.global_scheduler.cara.cara_instance.ollama_instance import OllamaInstance
                    # Convert HF model name to Ollama tag format
                    ollama_model_name = to_ollama_tag(hf_model_name)
                    instance = OllamaInstance(
                        instance_id=instance_id,
                        hostname=hostname,
                        ip_address=ip_address,
                        predictor_ports=predictor_ports,
                        model_name=ollama_model_name,
                        backend_port=backend_port
                    )
                elif backend_type == "vllm":
                    from block.global_scheduler.cara.cara_instance.vllm_instance import VllmInstance
                    instance = VllmInstance(
                        instance_id=instance_id,
                        hostname=hostname,
                        ip_address=ip_address,
                        predictor_ports=predictor_ports,
                        model_name=hf_model_name,
                        backend_port=backend_port
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
    print(set_ulimit() + " set limits file ")
    assert len(instances) > 0

    if args.debugging_logs:
        logger.setLevel(logging.DEBUG)

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
    parser.add_argument("--repetition-penalty", type=float, default=1.0,
                        help="Repetition penalty to use for generation to avoid repetition")
    args = parser.parse_args()
    logger.info("Starting server with args: %s", str(args))

    asyncio.run(run_server(args))
