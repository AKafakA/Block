"""
Dummy CARA predictor for data collection.

Returns simple heuristic-based predictions while collecting training data
for future model training. Tracks EMA service rates from completed requests.
"""
import random
import logging
import time
from typing import Dict, Optional

from block.predictor.cara.base_predictor import CARABasePredictor
from block.predictor.cara.cara_predictor_config import DummyPredictorConfig
from block.predictor.cara.data_structures import PredictRequest, ScheduleState
from block.predictor.cara.schedule_trace_client import ScheduleTraceClient
from block.predictor.cara.training_data_collector import TrainingDataCollector

logger = logging.getLogger(__name__)


class EMATracker:
    """Exponential moving average tracker for service rates."""

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.decode_tok_per_s: Optional[float] = None
        self.prefill_tok_per_s: Optional[float] = None
        self.decode_iter_ms: Optional[float] = None
        self._prev_num_preempted: Optional[int] = None
        self._prev_preempted_ts: Optional[float] = None
        self.kv_evictions_per_s: float = 0.0

    def update(
        self,
        num_prompt_tokens: int,
        e2e_latency: float,
        ttft: Optional[float],
        tpot: Optional[float],
        num_preempted: Optional[int] = None,
    ) -> None:
        """Update EMA from a completed request's metrics."""
        # Prefill rate from TTFT
        if ttft is not None and ttft > 0.001 and num_prompt_tokens > 0:
            instant_prefill = num_prompt_tokens / ttft
            if self.prefill_tok_per_s is None:
                self.prefill_tok_per_s = instant_prefill
            else:
                self.prefill_tok_per_s = (
                    self.alpha * instant_prefill
                    + (1 - self.alpha) * self.prefill_tok_per_s
                )

        # Decode rate from (e2el - ttft)
        decode_time = None
        if ttft is not None:
            decode_time = e2e_latency - ttft
        elif tpot is not None and tpot > 0:
            # Estimate decode_time from tpot if ttft unavailable
            pass

        if decode_time is not None and decode_time > 0.001:
            # Estimate output tokens from tpot if available
            if tpot is not None and tpot > 0:
                output_tokens = decode_time / tpot
            else:
                # Can't compute without knowing output tokens; skip
                output_tokens = None

            if output_tokens is not None and output_tokens > 0.5:
                instant_decode = output_tokens / decode_time
                instant_iter_ms = decode_time / output_tokens * 1000

                if self.decode_tok_per_s is None:
                    self.decode_tok_per_s = instant_decode
                    self.decode_iter_ms = instant_iter_ms
                else:
                    self.decode_tok_per_s = (
                        self.alpha * instant_decode
                        + (1 - self.alpha) * self.decode_tok_per_s
                    )
                    self.decode_iter_ms = (
                        self.alpha * instant_iter_ms
                        + (1 - self.alpha) * self.decode_iter_ms
                    )

        # KV eviction rate from num_preempted delta
        if num_preempted is not None:
            now = time.time()
            if self._prev_num_preempted is not None and self._prev_preempted_ts is not None:
                delta_p = num_preempted - self._prev_num_preempted
                delta_t = now - self._prev_preempted_ts
                if delta_t > 0 and delta_p >= 0:
                    self.kv_evictions_per_s = delta_p / delta_t
            self._prev_num_preempted = num_preempted
            self._prev_preempted_ts = now

    def inject_into_state(self, state: ScheduleState) -> None:
        """Inject current EMA values into a ScheduleState."""
        state.ema_decode_tok_per_s = self.decode_tok_per_s or 0.0
        state.ema_prefill_tok_per_s = self.prefill_tok_per_s or 0.0
        state.ema_decode_iter_ms = self.decode_iter_ms or 0.0
        state.kv_evictions_per_s = self.kv_evictions_per_s


class DummyCARAPredictor(CARABasePredictor):
    """Dummy predictor that collects training data while using simple heuristics.

    Uses CARA-specific PredictRequest interface (not Vidur Request).
    """

    def __init__(self, config: DummyPredictorConfig, backend_port: int,
                 predictor_port: int, hostname: str = "localhost"):
        super().__init__(config, backend_port)
        self._predictor_port = predictor_port
        self._hostname = hostname

        # Schedule trace client
        self.schedule_client = ScheduleTraceClient(
            backend_host=hostname,
            backend_port=backend_port,
            timeout=config.schedule_trace_timeout
        )

        # EMA tracker for service rates
        self.ema_tracker = EMATracker(alpha=0.1)

        # Training data collector (if enabled)
        self.data_collector = None
        if config.enable_data_collection:
            self.data_collector = TrainingDataCollector(
                output_dir=config.data_output_dir,
                hostname=hostname,
                predictor_port=predictor_port,
                sample_rate=config.data_collection_sample_rate,
                save_batch_size=config.save_batch_size
            )
            logger.info(
                f"Data collection enabled: output_dir={config.data_output_dir}, "
                f"hostname={hostname}, port={predictor_port}, "
                f"sample_rate={config.data_collection_sample_rate}"
            )

        self.heuristic_mode = config.heuristic_mode
        logger.info(
            f"DummyCARAPredictor initialized: hostname={hostname}, "
            f"backend_port={backend_port}, predictor_port={predictor_port}, "
            f"heuristic={self.heuristic_mode}"
        )

    async def predict(self, target_request: PredictRequest) -> Dict:
        """Make prediction using simple heuristics.

        Fetches both /instance_stats and /schedule_trace in parallel,
        injects EMA rates, and logs for training data collection.
        """
        predict_start = time.monotonic()

        # Fetch both endpoints in parallel
        schedule_state, probe_latency_ms = await self.schedule_client.fetch_both()

        # Handle fetch failure
        if schedule_state is None:
            logger.warning(
                f"Failed to fetch instance state for request {target_request.request_id}, "
                "proceeding with minimal state for data collection"
            )
            schedule_state = ScheduleState()

        # Inject current EMA rates into the state
        self.ema_tracker.inject_into_state(schedule_state)

        prediction_latency_ms = (time.monotonic() - predict_start) * 1000

        # Log prediction context for training data collection
        if self.data_collector:
            will_collect = await self.data_collector.log_prediction(
                request_id=target_request.request_id,
                num_prompt_tokens=target_request.num_prompt_tokens,
                num_predicted_output_tokens=target_request.num_predicted_output_tokens,
                schedule_state=schedule_state,
                probe_latency_ms=probe_latency_ms,
                prediction_latency_ms=prediction_latency_ms,
            )
            if will_collect:
                logger.debug(
                    f"Will collect training data for request {target_request.request_id}"
                )

        # Compute heuristic-based metric
        target_metric = self._compute_heuristic(schedule_state)

        return {
            "target_metric": target_metric,
            "gpu_blocks": schedule_state.kv_free_blocks,
            "num_requests": schedule_state.total_requests,
            "num_preempted": schedule_state.num_preempted,
            "predictor_type": "dummy_cara",
            "probe_latency_ms": probe_latency_ms,
            "prediction_latency_ms": prediction_latency_ms,
        }

    def _compute_heuristic(self, schedule_state: ScheduleState) -> float:
        """Compute heuristic metric based on schedule state."""
        if self.heuristic_mode == "min_requests":
            return float(schedule_state.total_requests)

        elif self.heuristic_mode == "max_gpu_blocks":
            return -float(schedule_state.kv_free_blocks)

        elif self.heuristic_mode == "combined":
            num_requests = schedule_state.total_requests
            free_blocks = schedule_state.kv_free_blocks
            return num_requests / max(free_blocks, 1)

        else:  # "random" or unknown
            return random.random() * 1000

    async def log_actual_result(
        self,
        request_id: str,
        e2e_latency: float,
        ttft: float = None,
        tpot: float = None,
        num_prompt_tokens: int = None,
    ):
        """Log actual metrics for a completed request.

        Updates EMA tracker and logs to data collector.
        """
        # Update EMA from completed request
        prompt_tokens = num_prompt_tokens or 0
        # Try to get prompt tokens from pending prediction if not provided
        if prompt_tokens == 0 and self.data_collector:
            pending = self.data_collector.pending_predictions.get(request_id)
            if pending is not None:
                prompt_tokens = pending.num_prompt_tokens

        if prompt_tokens > 0:
            self.ema_tracker.update(
                num_prompt_tokens=prompt_tokens,
                e2e_latency=e2e_latency,
                ttft=ttft,
                tpot=tpot,
            )

        if self.data_collector:
            await self.data_collector.log_actual_result(
                request_id=request_id,
                e2e_latency=e2e_latency,
                ttft=ttft,
                tpot=tpot
            )

    async def shutdown(self):
        """Cleanup on shutdown - save any remaining data."""
        if self.data_collector:
            logger.info("Flushing training data collector on shutdown...")
            await self.data_collector.flush()

            stats = self.data_collector.get_stats()
            logger.info(f"Final collection stats: {stats}")
