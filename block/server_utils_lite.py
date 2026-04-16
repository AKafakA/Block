"""
Lightweight server utilities for CARA sidecar deployment.

Only includes serve_http — no vidur/Block dependencies.
Used on worker nodes that don't have the full Block package.
"""
import asyncio
import logging
import signal
from typing import Optional, Any

import psutil
import uvicorn
from fastapi import FastAPI


def find_process_using_port(port: int) -> Optional[psutil.Process]:
    for conn in psutil.net_connections():
        if conn.laddr.port == port:
            try:
                return psutil.Process(conn.pid)
            except psutil.NoSuchProcess:
                return None
    return None


async def serve_http(app: FastAPI, **uvicorn_kwargs: Any):
    config = uvicorn.Config(app, **uvicorn_kwargs)
    server = uvicorn.Server(config)
    loop = asyncio.get_running_loop()
    server_task = loop.create_task(server.serve())

    def signal_handler() -> None:
        server_task.cancel()

    async def dummy_shutdown() -> None:
        pass

    loop.add_signal_handler(signal.SIGINT, signal_handler)
    loop.add_signal_handler(signal.SIGTERM, signal_handler)

    try:
        await server_task
        return dummy_shutdown()
    except asyncio.CancelledError:
        port = uvicorn_kwargs.get("port")
        if port:
            process = find_process_using_port(port)
            if process is not None:
                logging.debug(
                    "port %s is used by process %s launched with command:\n%s",
                    port, process, " ".join(process.cmdline()))
        logging.info("Shutting down FastAPI HTTP server.")
        return server.shutdown()
