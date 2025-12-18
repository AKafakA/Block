"""
API server for CARA predictors.

Separate from Block's predictor API server to avoid coupling with
Block-specific configs and Vidur request transformations.
"""
import argparse
import asyncio
import logging
import ssl
import time
from argparse import Namespace
from typing import Any, Optional
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from block.predictor.cara.cara_predictor_config import CARABasePredictorConfig
from block.predictor.cara.data_structures import PredictRequest
from block.server_utils import serve_http
from block.global_scheduler.cara.utils import set_ulimit

TIMEOUT_KEEP_ALIVE = 5  # seconds
app = FastAPI()
predictor: Optional[Any] = None  # CARA predictor instance
start_time = 0

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)


@app.post("/predict")
async def predict(request: Request) -> Response:
    """Predict metrics for a target request.

    Request JSON format:
    {
        "request_id": str,
        "num_prompt_tokens": int,
        "num_predicted_output_tokens": int
    }

    Response JSON format:
    {
        "target_metric": float,
        "gpu_blocks": int,
        "num_requests": int,
        "num_preempted": int,
        "predictor_type": str,
        "time_to_predict": float  # milliseconds
    }
    """
    assert predictor is not None
    pred_start_time = time.time()
    request_dict = await request.json()

    # Create PredictRequest
    target_request = PredictRequest(
        request_id=str(request_dict["request_id"]),
        num_prompt_tokens=int(request_dict["num_prompt_tokens"]),
        num_predicted_output_tokens=int(request_dict["num_predicted_output_tokens"])
    )

    metric = await predictor.predict(target_request)
    time_elapsed = (time.time() - pred_start_time) * 1000

    logger.debug(
        f"Predicted for request {target_request.request_id}: "
        f"metric={metric['target_metric']:.2f}, time={time_elapsed:.2f}ms"
    )

    metric["time_to_predict"] = time_elapsed
    return JSONResponse(metric)


@app.post("/log_actual")
async def log_actual(request: Request) -> Response:
    """Log actual metrics for training data collection.

    Request JSON format:
    {
        "request_id": str,
        "e2e_latency": float,
        "ttft": float (optional),
        "tpot": float (optional)
    }
    """
    assert predictor is not None
    request_dict = await request.json()

    # Only CARA predictors with data collection have this method
    if hasattr(predictor, 'log_actual_result'):
        await predictor.log_actual_result(
            request_id=request_dict['request_id'],
            e2e_latency=request_dict['e2e_latency'],
            ttft=request_dict.get('ttft'),
            tpot=request_dict.get('tpot')
        )
        logger.debug(f"Logged actual for request: {request_dict['request_id']}")
    else:
        logger.warning("Predictor does not support log_actual_result")

    return Response(status_code=200)


@app.get("/stats")
async def stats() -> Response:
    """Get collection statistics (for data collection predictors)."""
    if hasattr(predictor, 'data_collector') and predictor.data_collector:
        stats_dict = predictor.data_collector.get_stats()
        return JSONResponse(stats_dict)
    else:
        return JSONResponse({"error": "Data collection not enabled"}, status_code=400)


@app.post("/flush")
async def flush() -> Response:
    """Force flush buffered training data to disk (for testing)."""
    if hasattr(predictor, 'data_collector') and predictor.data_collector:
        await predictor.data_collector.flush()
        stats_dict = predictor.data_collector.get_stats()
        return JSONResponse({"status": "flushed", **stats_dict})
    else:
        return JSONResponse({"error": "Data collection not enabled"}, status_code=400)


def build_app(args: Namespace) -> FastAPI:
    global app
    app.root_path = args.root_path
    return app


async def init_app(
    args: Namespace,
    instance_predictor: Optional[Any] = None,
) -> FastAPI:
    """Initialize CARA predictor API server."""
    app = build_app(args)
    global predictor

    # Load CARA predictor config
    config = CARABasePredictorConfig.from_json_file(args.config_path)
    logger.info(f"Loaded CARA predictor config: type={config.predictor_type}")

    # Create appropriate predictor
    if instance_predictor is not None:
        predictor = instance_predictor
    elif config.predictor_type == "dummy":
        from block.predictor.cara.dummy_cara_predictor import DummyCARAPredictor
        predictor = DummyCARAPredictor(
            config=config,
            backend_port=args.backend_port,
            predictor_port=args.port,
            hostname=args.hostname
        )
        logger.info(
            f"Created DummyCARAPredictor: hostname={args.hostname}, "
            f"backend_port={args.backend_port}, predictor_port={args.port}"
        )
    elif config.predictor_type == "lstm":
        # TODO: Implement LSTM predictor
        raise NotImplementedError("LSTM predictor not yet implemented")
    else:
        raise ValueError(f"Unknown predictor type: {config.predictor_type}")

    return app


async def run_server(
    args: Namespace,
    instance_predictor: Optional[Any] = None,
    **uvicorn_kwargs: Any
) -> None:
    """Run CARA predictor API server."""
    global start_time
    start_time = time.time()

    app = await init_app(args, instance_predictor)
    assert predictor is not None

    logger.info(f"Starting CARA predictor server on {args.host}:{args.port}")
    logger.info(f"Monitoring backend at {args.hostname}:{args.backend_port}")

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
        # Cleanup on shutdown
        if hasattr(predictor, 'shutdown'):
            logger.info("Shutting down predictor...")
            await predictor.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CARA Predictor API Server"
    )
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host to bind to")
    parser.add_argument("--port", type=int, default=8100,
                        help="Port this predictor listens on")
    parser.add_argument("--backend-port", type=int, default=8000,
                        help="Port of backend instance (vLLM/Ollama)")
    parser.add_argument("--hostname", type=str, default="localhost",
                        help="Hostname of the backend instance being monitored")
    parser.add_argument("--config-path", type=str,
                        default="block/config/cara/predictor_config.json",
                        help="Path to CARA predictor config JSON")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of uvicorn workers")
    parser.add_argument("--ssl-keyfile", type=str, default=None)
    parser.add_argument("--ssl-certfile", type=str, default=None)
    parser.add_argument("--ssl-ca-certs", type=str, default=None,
                        help="The CA certificates file")
    parser.add_argument("--ssl-cert-reqs", type=int,
                        default=int(ssl.CERT_NONE),
                        help="Whether client certificate is required")
    parser.add_argument("--root-path", type=str, default=None,
                        help="FastAPI root_path when app is behind a proxy")

    args = parser.parse_args()

    # Set file descriptor limits
    set_ulimit()

    logger.info(f"Starting CARA predictor server with args: {args}")
    asyncio.run(run_server(args))
