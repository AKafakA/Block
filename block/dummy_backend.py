"""
NOTE: This API server is used only for demonstrating usage of AsyncEngine
and simple performance benchmarks. It is not intended for production use.
For production use, we recommend using our OpenAI compatible server.
We are also not going to accept PRs modifying this file, please
change `vllm/entrypoints/openai/api_server.py` instead.
"""
import argparse
import asyncio
import ssl
from argparse import Namespace
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response

from block.server_utils import serve_http

app = FastAPI()
counter = 0
TIMEOUT_KEEP_ALIVE = 5  # seconds.


@app.get("/schedule_trace")
async def status() -> Response:
    """Status check."""
    dummy_status = {}
    free_gpu_blocks = 1000
    dummy_statuses = {"1": dummy_status}
    running_request_info = {"request_id": 0, "arrival_time": 0, "seq_prompts_length": 100, "seq_total_output_length": 70,
                            "n_blocks": 35}
    waiting_request_info = {"request_id": 1, "arrival_time": 3, "seq_prompts_length": 100, "seq_total_output_length": 60,
                            "n_blocks": 70}

    global counter
    if counter == 0:
        dummy_status["running"] = []
        dummy_status["swap"] = []
        dummy_status["waiting"] = []
        dummy_status["gpu_blocks"] = free_gpu_blocks
    elif counter == 1:
        dummy_status["running"] = [running_request_info]
        dummy_status["swap"] = []
        dummy_status["waiting"] = []
        dummy_status["gpu_blocks"] = free_gpu_blocks - running_request_info["n_blocks"]
    elif counter == 2:
        dummy_status["running"] = [running_request_info, waiting_request_info]
        dummy_status["swap"] = []
        dummy_status["waiting"] = []
        dummy_status["gpu_blocks"] = (free_gpu_blocks - running_request_info["n_blocks"]
                                      - waiting_request_info["n_blocks"])
    counter += 1
    return JSONResponse(dummy_statuses)


def build_app(args: Namespace) -> FastAPI:
    global app
    app.root_path = args.root_path
    return app


async def run_server(args: Namespace,
                     **uvicorn_kwargs: Any) -> None:
    app = build_app(args)
    app.root_path = args.root_path

    shutdown_task = await serve_http(
        app,
        host=args.host,
        port=args.port,
        timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
        ssl_keyfile=args.ssl_keyfile,
        ssl_certfile=args.ssl_certfile,
        ssl_ca_certs=args.ssl_ca_certs,
        ssl_cert_reqs=args.ssl_cert_reqs,
        **uvicorn_kwargs,
    )

    await shutdown_task

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=8080)
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
    args = parser.parse_args()
    asyncio.run(run_server(args))

