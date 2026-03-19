"""
Client for fetching instance state from vLLM endpoints.

Calls two endpoints in parallel:
- /instance_stats: aggregate features (XGBoost/Linear/Roofline)
- /schedule_trace: per-request lists (LSTM)
"""
import aiohttp
import asyncio
import logging
import time
from typing import Optional, Tuple

from block.predictor.cara.data_structures import ScheduleState

logger = logging.getLogger(__name__)


class ScheduleTraceClient:
    """Async client for querying vLLM instance state endpoints."""

    def __init__(self, backend_host: str, backend_port: int, timeout: int = 5):
        """
        Args:
            backend_host: IP address or hostname of vLLM instance
            backend_port: Port of vLLM instance (usually 8000)
            timeout: Timeout for HTTP request in seconds
        """
        self.base_url = f"http://{backend_host}:{backend_port}"
        self.schedule_trace_url = f"{self.base_url}/schedule_trace"
        self.instance_stats_url = f"{self.base_url}/instance_stats"
        # Keep backward compat
        self.backend_url = self.schedule_trace_url
        self.timeout = aiohttp.ClientTimeout(total=timeout)

    async def fetch_schedule_trace(self) -> Optional[ScheduleState]:
        """Fetch current schedule state from vLLM /schedule_trace.

        Returns:
            ScheduleState object if successful, None if error/timeout
        """
        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.get(self.schedule_trace_url) as response:
                    if response.status != 200:
                        logger.warning(
                            f"schedule_trace returned status {response.status}"
                        )
                        return None

                    response_dict = await response.json()
                    state = ScheduleState.from_schedule_trace(response_dict)
                    logger.debug(
                        f"Fetched schedule_trace: {state.total_requests} requests, "
                        f"{state.free_gpu_blocks} free GPU blocks"
                    )
                    return state

        except asyncio.TimeoutError:
            logger.warning(
                f"Timeout fetching schedule_trace from {self.schedule_trace_url}"
            )
            return None
        except Exception as e:
            logger.error(
                f"Error fetching schedule_trace from {self.schedule_trace_url}: {e}"
            )
            return None

    async def fetch_instance_stats(self) -> Optional[ScheduleState]:
        """Fetch aggregate instance stats from vLLM /instance_stats.

        Returns:
            ScheduleState with aggregate fields populated, None on error.
        """
        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.get(self.instance_stats_url) as response:
                    if response.status != 200:
                        logger.warning(
                            f"instance_stats returned status {response.status}"
                        )
                        return None

                    stats_dict = await response.json()
                    state = ScheduleState.from_instance_stats(stats_dict)
                    logger.debug(
                        f"Fetched instance_stats: {state.num_running} running, "
                        f"{state.num_waiting} waiting, "
                        f"kv_util={state.kv_cache_utilization:.2f}"
                    )
                    return state

        except asyncio.TimeoutError:
            logger.warning(
                f"Timeout fetching instance_stats from {self.instance_stats_url}"
            )
            return None
        except Exception as e:
            logger.error(
                f"Error fetching instance_stats from {self.instance_stats_url}: {e}"
            )
            return None

    async def fetch_both(self) -> Tuple[Optional[ScheduleState], float]:
        """Fetch both /instance_stats and /schedule_trace in parallel.

        Returns a single merged ScheduleState with both aggregate features
        and per-request lists. Also returns probe latency in milliseconds.

        Returns:
            (ScheduleState or None, probe_latency_ms)
        """
        probe_start = time.monotonic()

        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                # Fire both requests in parallel
                stats_task = session.get(self.instance_stats_url)
                trace_task = session.get(self.schedule_trace_url)

                stats_resp, trace_resp = await asyncio.gather(
                    stats_task, trace_task, return_exceptions=True
                )

                probe_ms = (time.monotonic() - probe_start) * 1000

                # Parse instance_stats (primary)
                if isinstance(stats_resp, Exception):
                    logger.warning(f"instance_stats failed: {stats_resp}")
                    stats_dict = None
                elif stats_resp.status != 200:
                    logger.warning(f"instance_stats returned {stats_resp.status}")
                    stats_dict = None
                else:
                    stats_dict = await stats_resp.json()

                # Parse schedule_trace (secondary, for LSTM)
                if isinstance(trace_resp, Exception):
                    logger.warning(f"schedule_trace failed: {trace_resp}")
                    trace_dict = None
                elif trace_resp.status != 200:
                    logger.warning(f"schedule_trace returned {trace_resp.status}")
                    trace_dict = None
                else:
                    trace_dict = await trace_resp.json()

                # Build merged state
                if stats_dict is not None:
                    state = ScheduleState.from_instance_stats(stats_dict)
                    if trace_dict is not None:
                        state.merge_schedule_trace(trace_dict)
                elif trace_dict is not None:
                    # Fallback: only schedule_trace available
                    state = ScheduleState.from_schedule_trace(trace_dict)
                else:
                    return None, probe_ms

                logger.debug(
                    f"Fetched both endpoints in {probe_ms:.1f}ms: "
                    f"{state.num_running} running, {state.num_waiting} waiting, "
                    f"{len(state.running)} trace entries"
                )
                return state, probe_ms

        except asyncio.TimeoutError:
            probe_ms = (time.monotonic() - probe_start) * 1000
            logger.warning(f"Timeout fetching both endpoints ({probe_ms:.1f}ms)")
            return None, probe_ms
        except Exception as e:
            probe_ms = (time.monotonic() - probe_start) * 1000
            logger.error(f"Error fetching both endpoints: {e}")
            return None, probe_ms
