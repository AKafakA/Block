import argparse
import asyncio
import json
import logging
import os
import ssl
import time
from argparse import Namespace
from typing import Any, Optional
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from block.predictor.predictor_config import PredictorConfig
from block.predictor.predictor import Predictor
from block.server_utils import convert_request, get_predictor, serve_http
import resource

TIMEOUT_KEEP_ALIVE = 5  # seconds.
app = FastAPI()
predictor: Optional[Predictor] = None
start_time = 0



@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)

@app.post("/predict")
async def predict(request: Request) -> Response:
    """Predict completion for the request. """
    assert predictor is not None
    start_time = time.time()
    request_dict = await request.json()
    vidur_request = convert_request(request_dict)
    metric = await predictor.predict(vidur_request)
    time_elapsed = (time.time() - start_time) * 1000
    logging.debug("Predicted metric: %s for request: %s", metric, str(vidur_request.id))
    metric["time_to_predict"] = time_elapsed
    return JSONResponse(metric)


def build_app(args: Namespace) -> FastAPI:
    global app
    app.root_path = args.root_path
    return app


async def init_app(
        args: Namespace,
        instance_predictor: Optional[Predictor] = None,
) -> FastAPI:
    app = build_app(args)
    instance_port = args.instance_port
    global predictor
    config_path = args.config_path
    config_dict = json.load(open(config_path))
    config: PredictorConfig = PredictorConfig.create_from_dict(config_dict, args.enable_chunked_prefill)
    if args.metric_type:
        config.target_metric = args.metric_type
    config.replica_scheduler_config.batch_size_cap = args.batch_size_cap
    config.enable_batch_time_estimation = args.enable_time_estimation
    config.threshold_batch_size_for_time_estimation = args.threshold_batch_size_for_time_estimation
    predictor = (instance_predictor if instance_predictor is not None else
                 get_predictor(args.predictor_type, config, instance_port))
    config.prediction_timeout = args.predictor_timeout
    return app


async def run_server(args: Namespace,
                     instance_predictor: Optional[Predictor] = None,
                     **uvicorn_kwargs: Any) -> None:
    global start_time, app
    start_time = time.time()
    app = await init_app(args, instance_predictor)
    assert predictor is not None
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
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--ssl-keyfile", type=str, default=None)
    parser.add_argument("--ssl-certfile", type=str, default=None)
    parser.add_argument("--workers", type=int, default=1)
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
    parser.add_argument("--instance-port", type=int, default=8000)
    parser.add_argument("--config_path", type=str, default="vidur/prediction/config/llama_config.json")
    parser.add_argument("--predictor_type", type=str, default="simulate")
    parser.add_argument("--metric_type", type=str, default="")
    parser.add_argument("--enable_time_estimation", type=bool, default=True)
    parser.add_argument("--batch_size_cap", type=int, default=48)
    parser.add_argument("--enable_chunked_prefill", action='store_true')
    parser.add_argument("--threshold_batch_size_for_time_estimation", type=int, default=36,
                        help="Threshold batch size for enabling time estimation. "
                             "Less that 0 means disable time estimation. 0 means always enable time estimation."
                             "And >0 means enable time estimation only when batch size > this")
    parser.add_argument("--predictor_timeout", type=int, default=10)
    parser.add_argument("--predictor_index", type=int, default=1)
    logging.log(logging.INFO, "Starting server with args: %s", str(parser.parse_args()))
    args = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_NOFILE, (65536, 65536))
    asyncio.run(run_server(args))
