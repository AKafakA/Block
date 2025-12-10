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
import resource
import logging
import traceback

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



# align with vllm bench so we can directly leverage existing tools
@app.post("/v1/completions")
async def completion(request: Request) -> Response:
    global num_requests
    request_json = await request.json()
    num_requests += 1
    request_id = request_json.get("request_id")
    served_requests.append(request_id)
    try:
        if scheduling == "random":
            instance = random.choice(instances)
        elif scheduling == "round_robin":
            instance = instances[num_requests % len(instances)]
        else:
            instance = random.choice(instances)
        response_dict = await instance.query_instance(
            request_json,
            # useless for now, leave for future extension
            predicted_num_decode_tokens=0
        )
        return JSONResponse(content=response_dict)
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        logger.error(traceback.format_exc())
        error_response = {
            "error": str(e)
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
    global instances, start_time, scheduling
    model_config_path = args.model_config_path

    model_dict = json.load(open(model_config_path))
    host_config = json.load(open(args.host_config))
    if instances_list is not None:
        instances.extend(instances_list)
    else:
        for model, model_config in model_dict.items():
            node_hosts = model_config["node_hosts"]
            backend_type = model_config["backend"]
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
                    instance = OllamaInstance(
                        instance_id=instance_id,
                        ip_address=ip_address,
                        predictor_ports=predictor_ports,
                        model_name=model,
                        backend_port=backend_port
                    )
                elif backend_type == "vllm":
                    from block.global_scheduler.cara.cara_instance.vllm_instance import VllmInstance
                    instance = VllmInstance(
                        instance_id=instance_id,
                        ip_address=ip_address,
                        predictor_ports=predictor_ports,
                        model_name=model,
                        backend_port=backend_port
                    )
                else:
                    raise ValueError(f"Unsupported backend type: {backend_type}")
                instances.append(instance)

    start_time = time.time()
    scheduling = args.scheduling
    return app


async def run_server(args: Namespace,
                     instances_list: Optional[List[Instance]] = None,
                     **uvicorn_kwargs: Any) -> None:
    app = await init_app(args, instances_list)
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
    args = parser.parse_args()
    logger.info("Starting server with args: %s", str(args))
    # in case the limited by the number of files
    resource.setrlimit(resource.RLIMIT_NOFILE, (65536, 65536))

    asyncio.run(run_server(args))
