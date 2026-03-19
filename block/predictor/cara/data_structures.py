"""
Data structures for CARA predictor.

Combines two vLLM endpoints:
- /schedule_trace: per-request lists (running/waiting) for LSTM
- /instance_stats: aggregate features for XGBoost/Linear/Roofline

Plus client-side EMA service rates and probing overhead tracking.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
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
    total_num_tokens: int

    @property
    def num_output_tokens(self) -> int:
        """Derived output tokens = total_num_tokens - num_prompt_tokens."""
        return max(0, self.total_num_tokens - self.num_prompt_tokens)

    @classmethod
    def from_list(cls, raw_list: List, offset: int) -> 'RequestInfo':
        """Parse from flat list at given offset.
        Format: [request_id, num_prompt_tokens, num_computed_tokens, total_num_tokens]
        """
        return cls(
            request_id=str(raw_list[offset]),
            num_prompt_tokens=int(raw_list[offset + 1]),
            num_computed_tokens=int(raw_list[offset + 2]),
            total_num_tokens=int(raw_list[offset + 3])
        )


@dataclass
class ScheduleState:
    """Current scheduling state from vLLM instance.

    Combines data from two sources:
    - /schedule_trace: per-request running/waiting lists (for LSTM)
    - /instance_stats: aggregate features (for XGBoost/Linear/Roofline)
    Plus client-side EMA service rates.
    """
    # Per-request lists from /schedule_trace (LSTM features)
    running: List[RequestInfo] = field(default_factory=list)
    waiting: List[RequestInfo] = field(default_factory=list)

    # Aggregate features from /instance_stats (XGBoost features)
    num_running: int = 0
    num_waiting: int = 0
    num_active_decode_seqs: int = 0
    decode_ctx_p50: float = 0.0
    decode_ctx_p95: float = 0.0
    decode_ctx_max: float = 0.0
    pending_prefill_tokens: int = 0
    pending_decode_tokens: int = 0
    kv_cache_utilization: float = 0.0
    kv_free_blocks: int = 0
    token_budget_per_iter: int = 0
    prefill_chunk_size: int = 0
    max_num_seqs: int = 0
    num_preempted: int = 0

    # EMA service rates (computed client-side in sidecar)
    ema_decode_tok_per_s: float = 0.0
    ema_prefill_tok_per_s: float = 0.0
    ema_decode_iter_ms: float = 0.0
    kv_evictions_per_s: float = 0.0

    # Legacy field (kept for backward compat with old data)
    free_gpu_blocks: int = 0

    timestamp: float = field(default_factory=time.time)

    @classmethod
    def from_schedule_trace(cls, response_dict: dict) -> 'ScheduleState':
        """Parse from /schedule_trace response.

        Expected format:
        {
            "running": [req_id, n_prompt, n_computed, total_num_tokens, ...],
            "waiting": [req_id, n_prompt, n_computed, total_num_tokens, ...],
            "free_gpu_blocks": int,
            "num_preempted": int
        }
        """
        running_raw = response_dict.get("running", [])
        waiting_raw = response_dict.get("waiting", [])

        FIELDS_PER_REQUEST = 4

        running = []
        for i in range(0, len(running_raw), FIELDS_PER_REQUEST):
            if i + FIELDS_PER_REQUEST <= len(running_raw):
                running.append(RequestInfo.from_list(running_raw, i))

        waiting = []
        for i in range(0, len(waiting_raw), FIELDS_PER_REQUEST):
            if i + FIELDS_PER_REQUEST <= len(waiting_raw):
                waiting.append(RequestInfo.from_list(waiting_raw, i))

        free_blocks = response_dict.get("free_gpu_blocks", 0)
        num_preempted = response_dict.get("num_preempted", 0)

        return cls(
            running=running,
            waiting=waiting,
            num_running=len(running),
            num_waiting=len(waiting),
            free_gpu_blocks=free_blocks,
            kv_free_blocks=free_blocks,
            num_preempted=num_preempted,
            timestamp=time.time()
        )

    @classmethod
    def from_response(cls, response_dict: dict) -> 'ScheduleState':
        """Backward-compatible alias for from_schedule_trace."""
        return cls.from_schedule_trace(response_dict)

    @classmethod
    def from_instance_stats(cls, stats_dict: dict) -> 'ScheduleState':
        """Parse from /instance_stats response.

        Expected format from vLLM /instance_stats endpoint:
        {
            "num_running": 8, "num_waiting": 3, "num_active_decode_seqs": 6,
            "decode_ctx_p50": 480, "decode_ctx_p95": 1024, "decode_ctx_max": 1200,
            "pending_prefill_tokens": 1500, "pending_decode_tokens": 2400,
            "kv_cache_utilization": 0.65, "kv_free_blocks": 1287,
            "token_budget_per_iter": 2048, "max_num_seqs": 256,
            "num_preempted": 0, ...
        }
        """
        free_blocks = stats_dict.get("kv_free_blocks", 0)
        return cls(
            num_running=stats_dict.get("num_running", 0),
            num_waiting=stats_dict.get("num_waiting", 0),
            num_active_decode_seqs=stats_dict.get("num_active_decode_seqs", 0),
            decode_ctx_p50=stats_dict.get("decode_ctx_p50", 0.0),
            decode_ctx_p95=stats_dict.get("decode_ctx_p95", 0.0),
            decode_ctx_max=stats_dict.get("decode_ctx_max", 0.0),
            pending_prefill_tokens=stats_dict.get("pending_prefill_tokens", 0),
            pending_decode_tokens=stats_dict.get("pending_decode_tokens", 0),
            kv_cache_utilization=stats_dict.get("kv_cache_utilization", 0.0),
            kv_free_blocks=free_blocks,
            free_gpu_blocks=free_blocks,
            token_budget_per_iter=stats_dict.get("token_budget_per_iter", 0),
            prefill_chunk_size=stats_dict.get("prefill_chunk_size", 0),
            max_num_seqs=stats_dict.get("max_num_seqs", 0),
            num_preempted=stats_dict.get("num_preempted", 0),
            timestamp=time.time()
        )

    def merge_schedule_trace(self, trace_dict: dict) -> None:
        """Merge per-request lists from /schedule_trace into this state.

        Call after from_instance_stats() to add LSTM-needed request lists.
        """
        running_raw = trace_dict.get("running", [])
        waiting_raw = trace_dict.get("waiting", [])
        FIELDS_PER_REQUEST = 4

        self.running = []
        for i in range(0, len(running_raw), FIELDS_PER_REQUEST):
            if i + FIELDS_PER_REQUEST <= len(running_raw):
                self.running.append(RequestInfo.from_list(running_raw, i))

        self.waiting = []
        for i in range(0, len(waiting_raw), FIELDS_PER_REQUEST):
            if i + FIELDS_PER_REQUEST <= len(waiting_raw):
                self.waiting.append(RequestInfo.from_list(waiting_raw, i))

    def to_dict(self) -> Dict:
        """Serialize to dict for JSON output (training data)."""
        return {
            # Aggregate features (XGBoost/Linear/Roofline)
            "num_running": self.num_running,
            "num_waiting": self.num_waiting,
            "num_active_decode_seqs": self.num_active_decode_seqs,
            "decode_ctx_p50": self.decode_ctx_p50,
            "decode_ctx_p95": self.decode_ctx_p95,
            "decode_ctx_max": self.decode_ctx_max,
            "pending_prefill_tokens": self.pending_prefill_tokens,
            "pending_decode_tokens": self.pending_decode_tokens,
            "kv_cache_utilization": self.kv_cache_utilization,
            "kv_free_blocks": self.kv_free_blocks,
            "token_budget_per_iter": self.token_budget_per_iter,
            "prefill_chunk_size": self.prefill_chunk_size,
            "max_num_seqs": self.max_num_seqs,
            "num_preempted": self.num_preempted,
            # EMA rates (client-side computed)
            "ema_decode_tok_per_s": self.ema_decode_tok_per_s,
            "ema_prefill_tok_per_s": self.ema_prefill_tok_per_s,
            "ema_decode_iter_ms": self.ema_decode_iter_ms,
            "kv_evictions_per_s": self.kv_evictions_per_s,
            # Per-request lists (LSTM)
            "running_requests": [
                {
                    "request_id": r.request_id,
                    "num_prompt_tokens": r.num_prompt_tokens,
                    "num_computed_tokens": r.num_computed_tokens,
                    "total_num_tokens": r.total_num_tokens,
                    "num_output_tokens": r.num_output_tokens,
                } for r in self.running
            ],
            "waiting_requests": [
                {
                    "request_id": r.request_id,
                    "num_prompt_tokens": r.num_prompt_tokens,
                    "num_computed_tokens": r.num_computed_tokens,
                    "total_num_tokens": r.total_num_tokens,
                    "num_output_tokens": r.num_output_tokens,
                } for r in self.waiting
            ],
        }

    @property
    def total_requests(self) -> int:
        """Total number of requests in system."""
        # Prefer aggregate counts; fall back to list lengths
        if self.num_running or self.num_waiting:
            return self.num_running + self.num_waiting
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

    # Overhead tracking (milliseconds)
    probe_latency_ms: Optional[float] = None
    prediction_latency_ms: Optional[float] = None

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
            "schedule_state": self.schedule_state.to_dict(),
            "instance_id": self.instance_id,
            "probe_latency_ms": self.probe_latency_ms,
            "prediction_latency_ms": self.prediction_latency_ms,
            "prediction_timestamp": self.prediction_timestamp,
            "actual_e2e_latency": self.actual_e2e_latency,
            "actual_ttft": self.actual_ttft,
            "actual_tpot": self.actual_tpot,
            "completion_timestamp": self.completion_timestamp
        }