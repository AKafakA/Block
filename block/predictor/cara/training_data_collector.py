"""
Training data collector for CARA predictor.

Collects prediction contexts and actual metrics for offline training.
"""
import json
import os
import time
import logging
import random
from typing import Dict, Optional
from pathlib import Path
from block.predictor.cara.data_structures import TrainingExample, ScheduleState

logger = logging.getLogger(__name__)


class TrainingDataCollector:
    """Collects training data by matching predictions with actual results."""

    def __init__(
        self,
        output_dir: str,
        hostname: str,
        predictor_port: int,
        sample_rate: float = 1.0,
        save_batch_size: int = 100
    ):
        """
        Args:
            output_dir: Directory to save training data
            hostname: Hostname of the instance being monitored
            predictor_port: Port of this predictor (for unique identification)
            sample_rate: Fraction of requests to collect (0.0-1.0)
            save_batch_size: Save to disk every N complete examples
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.hostname = hostname
        self.predictor_port = predictor_port
        self.sample_rate = sample_rate
        self.save_batch_size = save_batch_size

        # Pending predictions waiting for actual results
        self.pending_predictions: Dict[str, TrainingExample] = {}

        # Completed examples ready to save
        self.completed_examples = []

        # Statistics
        self.total_predictions = 0
        self.total_completed = 0
        self.total_saved = 0

        logger.info(
            f"TrainingDataCollector initialized: "
            f"output_dir={output_dir}, hostname={hostname}, port={predictor_port}, "
            f"sample_rate={sample_rate}"
        )

    def should_collect(self) -> bool:
        """Determine if this request should be collected based on sample rate."""
        return random.random() < self.sample_rate

    async def log_prediction(
        self,
        request_id: str,
        num_prompt_tokens: int,
        num_predicted_output_tokens: int,
        schedule_state: ScheduleState,
        probe_latency_ms: float = None,
        prediction_latency_ms: float = None,
    ) -> bool:
        """Log a prediction context.

        Args:
            request_id: Unique request identifier
            num_prompt_tokens: Number of prompt tokens
            num_predicted_output_tokens: Predicted output tokens
            schedule_state: Current schedule state
            probe_latency_ms: Time to fetch vLLM endpoints (ms)
            prediction_latency_ms: Total /predict call time (ms)

        Returns:
            True if this request will be collected, False otherwise
        """
        if not self.should_collect():
            return False

        # Create instance_id from hostname and port for tracking
        instance_id = f"{self.hostname}_port{self.predictor_port}"

        example = TrainingExample(
            request_id=request_id,
            num_prompt_tokens=num_prompt_tokens,
            num_predicted_output_tokens=num_predicted_output_tokens,
            schedule_state=schedule_state,
            instance_id=instance_id,
            prediction_timestamp=time.time(),
            probe_latency_ms=probe_latency_ms,
            prediction_latency_ms=prediction_latency_ms,
        )

        self.pending_predictions[request_id] = example
        self.total_predictions += 1

        logger.debug(
            f"Logged prediction for request {request_id} "
            f"({self.total_predictions} total predictions)"
        )

        return True

    async def log_actual_result(
        self,
        request_id: str,
        e2e_latency: float,
        ttft: Optional[float] = None,
        tpot: Optional[float] = None
    ):
        """Log actual metrics for a completed request.

        Args:
            request_id: Request identifier
            e2e_latency: Actual end-to-end latency
            ttft: Actual time to first token
            tpot: Actual time per output token
        """
        if request_id not in self.pending_predictions:
            # Not collected or already processed
            return

        example = self.pending_predictions.pop(request_id)
        example.actual_e2e_latency = e2e_latency
        example.actual_ttft = ttft
        example.actual_tpot = tpot
        example.completion_timestamp = time.time()

        self.completed_examples.append(example)
        self.total_completed += 1

        logger.debug(
            f"Logged actual result for request {request_id}: "
            f"e2e={e2e_latency:.3f}s, ttft={ttft:.3f}s, tpot={tpot:.4f}s "
            f"({self.total_completed} completed)"
        )

        # Auto-save when batch is full
        if len(self.completed_examples) >= self.save_batch_size:
            await self.save_batch()

    async def save_batch(self):
        """Save completed examples to disk."""
        if not self.completed_examples:
            return

        # Generate filename with hostname, port, and timestamp for uniqueness
        # Format: training_data_{hostname}_port{port}_{timestamp}.jsonl
        # This ensures concurrent predictors on the same host don't conflict
        timestamp = int(time.time())
        filename = f"training_data_{self.hostname}_port{self.predictor_port}_{timestamp}.jsonl"
        filepath = self.output_dir / filename

        try:
            with open(filepath, 'w') as f:
                for example in self.completed_examples:
                    json_line = json.dumps(example.to_dict())
                    f.write(json_line + '\n')

            self.total_saved += len(self.completed_examples)

            logger.info(
                f"Saved {len(self.completed_examples)} training examples to {filepath} "
                f"(total saved: {self.total_saved})"
            )

            self.completed_examples = []

        except Exception as e:
            logger.error(f"Error saving training data to {filepath}: {e}")

    async def flush(self):
        """Force save any remaining completed examples."""
        if self.completed_examples:
            await self.save_batch()

        # Clean up old pending predictions (timeout after 5 minutes)
        current_time = time.time()
        timeout_requests = [
            req_id for req_id, example in self.pending_predictions.items()
            if current_time - example.prediction_timestamp > 300
        ]

        for req_id in timeout_requests:
            self.pending_predictions.pop(req_id)
            logger.warning(f"Dropped timed-out pending prediction: {req_id}")

    def get_stats(self) -> Dict:
        """Get collection statistics."""
        return {
            "total_predictions": self.total_predictions,
            "total_completed": self.total_completed,
            "total_saved": self.total_saved,
            "pending": len(self.pending_predictions),
            "buffered": len(self.completed_examples)
        }