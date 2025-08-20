import asyncio
import json
import logging
import signal
from typing import Optional, Dict, Any

import psutil
import requests
import uvicorn
from fastapi import FastAPI
import aiohttp

from vidur.entities import Request as VidurRequest, Request
from block.predictor.predictor_config import PredictorConfig
from block.predictor.dummy_predictor import DummyPredictor
from block.predictor.simulate_predictor import SimulatePredictor

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)


def find_process_using_port(port: int) -> Optional[psutil.Process]:
    for conn in psutil.net_connections():
        if conn.laddr.port == port:
            try:
                return psutil.Process(conn.pid)
            except psutil.NoSuchProcess:
                return None
    return None


def convert_request(request: Dict) -> VidurRequest:
    request_id = request["id"]
    arrival_time = request["arrival_time"]
    num_context_tokens = request["num_context_tokens"]
    num_decode_tokens = request["num_decode_tokens"]
    vidur_request = VidurRequest(arrival_time, num_context_tokens, num_decode_tokens)
    vidur_request.set_id(request_id)
    return vidur_request


async def serve_http(app: FastAPI, **uvicorn_kwargs: Any):
    for route in app.routes:
        methods = getattr(route, "methods", None)
        path = getattr(route, "path", None)

        if methods is None or path is None:
            continue

    config = uvicorn.Config(app, **uvicorn_kwargs)
    server = uvicorn.Server(config)
    loop = asyncio.get_running_loop()
    server_task = loop.create_task(server.serve())

    def signal_handler() -> None:
        # prevents the uvicorn signal handler to exit early
        server_task.cancel()

    async def dummy_shutdown() -> None:
        pass

    loop.add_signal_handler(signal.SIGINT, signal_handler)
    loop.add_signal_handler(signal.SIGTERM, signal_handler)

    try:
        await server_task
        return dummy_shutdown()
    except asyncio.CancelledError:
        port = uvicorn_kwargs["port"]
        process = find_process_using_port(port)
        if process is not None:
            logging.debug(
                "port %s is used by process %s launched with command:\n%s",
                port, process, " ".join(process.cmdline()))
        logging.info("Shutting down FastAPI HTTP server.")
        return server.shutdown()


def get_predictor(type_str: str, predictor_config: PredictorConfig, instance_port: int == -1):
    if type_str == "dummy":
        return DummyPredictor(predictor_config, instance_port)
    elif type_str == "simulate":
        return SimulatePredictor(predictor_config, instance_port)


def get_http_request(query_url: str) -> requests.Response:
    headers = {"User-Agent": "Test Client"}
    response = requests.get(query_url, headers=headers)
    return response


def post_predicting_request(api_url: str,
                            request_id: int, num_context_tokens: int, num_decode_tokens: int,
                            arrived_at: float,
                            stream: bool = False) -> requests.Response:
    headers = {"User-Agent": "Test Client"}
    pload = {
        "id": request_id,
        "arrival_time": arrived_at,
        "num_context_tokens": num_context_tokens,
        "num_decode_tokens": num_decode_tokens,
    }
    response = requests.post(api_url,
                             headers=headers,
                             json=pload,
                             stream=stream)
    return response


def get_predicting_response(response: requests.Response):
    data = json.loads(response.content)
    output = data["metric"]
    return output


def post_serving_request(api_url: str,
                         prompt: str,
                         n: int = 1,
                         stream: bool = False) -> requests.Response:
    headers = {"User-Agent": "Test Client"}
    pload = {
        "prompt": prompt,
        "n": n,
        "temperature": 0.0,
        "max_tokens": 16,
        "stream": stream,
    }
    response = requests.post(api_url,
                             headers=headers,
                             json=pload,
                             stream=stream)
    return response
