"""
Client for fetching schedule trace from vLLM /schedule_trace endpoint.
"""
import aiohttp
import asyncio
import logging
from typing import Optional
from block.predictor.cara.data_structures import ScheduleState

logger = logging.getLogger(__name__)


class ScheduleTraceClient:
    """Async client for querying vLLM /schedule_trace endpoint."""

    def __init__(self, backend_host: str, backend_port: int, timeout: int = 5):
        """
        Args:
            backend_host: IP address or hostname of vLLM instance
            backend_port: Port of vLLM instance (usually 8000)
            timeout: Timeout for HTTP request in seconds
        """
        self.backend_url = f"http://{backend_host}:{backend_port}/schedule_trace"
        self.timeout = aiohttp.ClientTimeout(total=timeout)

    async def fetch_schedule_trace(self) -> Optional[ScheduleState]:
        """Fetch current schedule state from vLLM.

        Returns:
            ScheduleState object if successful, None if error/timeout
        """
        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.get(self.backend_url) as response:
                    if response.status != 200:
                        logger.warning(
                            f"schedule_trace returned status {response.status}"
                        )
                        return None

                    response_dict = await response.json()
                    state = ScheduleState.from_response(response_dict)
                    logger.debug(
                        f"Fetched schedule_trace: {state.total_requests} requests, "
                        f"{state.free_gpu_blocks} free GPU blocks"
                    )
                    return state

        except asyncio.TimeoutError:
            logger.warning(
                f"Timeout fetching schedule_trace from {self.backend_url}"
            )
            return None
        except Exception as e:
            logger.error(
                f"Error fetching schedule_trace from {self.backend_url}: {e}"
            )
            return None