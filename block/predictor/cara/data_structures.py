"""
Data structures for CARA predictor.
Parses vLLM /schedule_trace endpoint with 4-field format:
[request_id, num_prompt_tokens, num_computed_tokens, num_predicted_output_tokens, ...]
"""
from dataclasses import dataclass, field
from typing import List, Optional
import time


@dataclass
class PredictRequest:
    """Request information for CARA predictor prediction.

    Simple dataclass independent from Vidur Request.
    """
    request_id: str
    num_prompt_tokens: int
    num_predicted_output_tokens: int


@dataclass
class RequestInfo:
    """Single request information from schedule trace."""
    request_id: str
    num_prompt_tokens: int
    num_computed_tokens: int
    num_predicted_output_tokens: int

    @classmethod
    def from_list(cls, raw_list: List, offset: int) -> 'RequestInfo':
        """Parse from flat list at given offset.
        Format: [request_id, num_prompt_tokens, num_computed_tokens, num_predicted_output_tokens]
        """
        return cls(
            request_id=str(raw_list[offset]),
            num_prompt_tokens=int(raw_list[offset + 1]),
            num_computed_tokens=int(raw_list[offset + 2]),
            num_predicted_output_tokens=int(raw_list[offset + 3])
        )


@dataclass
class ScheduleState:
    """Current scheduling state from vLLM instance."""
    running: List[RequestInfo] = field(default_factory=list)
    waiting: List[RequestInfo] = field(default_factory=list)
    free_gpu_blocks: int = 0
    num_preempted: int = 0
    timestamp: float = field(default_factory=time.time)

    @classmethod
    def from_response(cls, response_dict: dict) -> 'ScheduleState':
        """Parse from /schedule_trace response.

        Expected format:
        {
            "running": [req_id, n_prompt, n_computed, n_predicted, ...],
            "waiting": [req_id, n_prompt, n_computed, n_predicted, ...],
            "free_gpu_blocks": int,
            "num_preempted": int
        }
        """
        running_raw = response_dict.get("running", [])
        waiting_raw = response_dict.get("waiting", [])

        # Each request takes 4 fields
        FIELDS_PER_REQUEST = 4

        running = []
        for i in range(0, len(running_raw), FIELDS_PER_REQUEST):
            if i + FIELDS_PER_REQUEST <= len(running_raw):
                running.append(RequestInfo.from_list(running_raw, i))

        waiting = []
        for i in range(0, len(waiting_raw), FIELDS_PER_REQUEST):
            if i + FIELDS_PER_REQUEST <= len(waiting_raw):
                waiting.append(RequestInfo.from_list(waiting_raw, i))

        return cls(
            running=running,
            waiting=waiting,
            free_gpu_blocks=response_dict.get("free_gpu_blocks", 0),
            num_preempted=response_dict.get("num_preempted", 0),
            timestamp=time.time()
        )

    @property
    def total_requests(self) -> int:
        """Total number of requests in system."""
        return len(self.running) + len(self.waiting)


@dataclass
class TrainingExample:
    """Training data point for CARA predictor.

    Stores the prediction context and actual observed metrics.
    """
    # Prediction inputs
    request_id: str
    num_prompt_tokens: int
    num_predicted_output_tokens: int
    schedule_state: ScheduleState
    instance_id: str
    prediction_timestamp: float

    # Ground truth labels (filled after request completion)
    actual_e2e_latency: Optional[float] = None
    actual_ttft: Optional[float] = None
    actual_tpot: Optional[float] = None
    completion_timestamp: Optional[float] = None

    def is_complete(self) -> bool:
        """Check if actual metrics have been collected."""
        return self.actual_e2e_latency is not None

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict for logging."""
        return {
            "request_id": self.request_id,
            "num_prompt_tokens": self.num_prompt_tokens,
            "num_predicted_output_tokens": self.num_predicted_output_tokens,
            "schedule_state": {
                "num_running": len(self.schedule_state.running),
                "num_waiting": len(self.schedule_state.waiting),
                "free_gpu_blocks": self.schedule_state.free_gpu_blocks,
                "num_preempted": self.schedule_state.num_preempted,
                # Store detailed request info for training
                "running_requests": [
                    {
                        "request_id": r.request_id,
                        "num_prompt_tokens": r.num_prompt_tokens,
                        "num_computed_tokens": r.num_computed_tokens,
                        "num_predicted_output_tokens": r.num_predicted_output_tokens
                    } for r in self.schedule_state.running
                ],
                "waiting_requests": [
                    {
                        "request_id": r.request_id,
                        "num_prompt_tokens": r.num_prompt_tokens,
                        "num_computed_tokens": r.num_computed_tokens,
                        "num_predicted_output_tokens": r.num_predicted_output_tokens
                    } for r in self.schedule_state.waiting
                ]
            },
            "instance_id": self.instance_id,
            "prediction_timestamp": self.prediction_timestamp,
            "actual_e2e_latency": self.actual_e2e_latency,
            "actual_ttft": self.actual_ttft,
            "actual_tpot": self.actual_tpot,
            "completion_timestamp": self.completion_timestamp
        }