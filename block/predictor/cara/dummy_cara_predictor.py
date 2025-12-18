"""
Dummy CARA predictor for data collection.

Returns simple heuristic-based predictions while collecting training data
for future LSTM model training.
"""
import random
import logging
from typing import Dict

from block.predictor.cara.base_predictor import CARABasePredictor
from block.predictor.cara.cara_predictor_config import DummyPredictorConfig
from block.predictor.cara.data_structures import PredictRequest
from block.predictor.cara.schedule_trace_client import ScheduleTraceClient
from block.predictor.cara.training_data_collector import TrainingDataCollector

logger = logging.getLogger(__name__)


class DummyCARAPredictor(CARABasePredictor):
    """Dummy predictor that collects training data while using simple heuristics.

    Uses CARA-specific PredictRequest interface (not Vidur Request).
    """

    def __init__(self, config: DummyPredictorConfig, backend_port: int,
                 predictor_port: int, hostname: str = "localhost"):
        """
        Args:
            config: Dummy predictor configuration
            backend_port: Backend instance port (vLLM)
            predictor_port: This predictor's port
            hostname: Hostname of the instance being monitored
        """
        super().__init__(config, backend_port)
        self._predictor_port = predictor_port
        self._hostname = hostname

        # Schedule trace client
        self.schedule_client = ScheduleTraceClient(
            backend_host=hostname,
            backend_port=backend_port,
            timeout=config.schedule_trace_timeout
        )

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

        Also logs prediction context for training data collection.

        Args:
            target_request: PredictRequest with request info

        Returns:
            Dict with prediction metrics
        """
        # Fetch current schedule state
        schedule_state = await self.schedule_client.fetch_schedule_trace()

        # Handle fetch failure
        if schedule_state is None:
            logger.warning(
                f"Failed to fetch schedule_trace for request {target_request.request_id}, "
                "using fallback values"
            )
            return {
                "target_metric": 999999.0,  # High penalty
                "gpu_blocks": -1,
                "num_requests": -1,
                "num_preempted": -1,
                "predictor_type": "dummy_cara"
            }

        # Log prediction context for training data collection
        if self.data_collector:
            will_collect = await self.data_collector.log_prediction(
                request_id=target_request.request_id,
                num_prompt_tokens=target_request.num_prompt_tokens,
                num_predicted_output_tokens=target_request.num_predicted_output_tokens,
                schedule_state=schedule_state
            )
            if will_collect:
                logger.debug(
                    f"Will collect training data for request {target_request.request_id}"
                )

        # Compute heuristic-based metric
        target_metric = self._compute_heuristic(schedule_state)

        return {
            "target_metric": target_metric,
            "gpu_blocks": schedule_state.free_gpu_blocks,
            "num_requests": schedule_state.total_requests,
            "num_preempted": schedule_state.num_preempted,
            "predictor_type": "dummy_cara"
        }

    def _compute_heuristic(self, schedule_state) -> float:
        """Compute heuristic metric based on schedule state.

        Args:
            schedule_state: Current schedule state

        Returns:
            Metric value (lower is better for scheduling)
        """
        if self.heuristic_mode == "min_requests":
            # Prefer instances with fewer requests
            return float(schedule_state.total_requests)

        elif self.heuristic_mode == "max_gpu_blocks":
            # Prefer instances with more free GPU blocks
            # Negative so lower is better
            return -float(schedule_state.free_gpu_blocks)

        elif self.heuristic_mode == "combined":
            # Simple combined heuristic
            # Normalized score considering both requests and GPU blocks
            num_requests = schedule_state.total_requests
            free_blocks = schedule_state.free_gpu_blocks
            # Avoid division by zero
            return num_requests / max(free_blocks, 1)

        else:  # "random" or unknown
            # Random assignment
            return random.random() * 1000

    async def log_actual_result(
        self,
        request_id: str,
        e2e_latency: float,
        ttft: float = None,
        tpot: float = None
    ):
        """Log actual metrics for a completed request.

        Called by the instance after request completion.

        Args:
            request_id: Request identifier
            e2e_latency: Actual end-to-end latency
            ttft: Actual time to first token
            tpot: Actual time per output token
        """
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