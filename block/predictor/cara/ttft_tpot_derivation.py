"""
TTFT and TPOT derivation from service rates.

Derives Time-to-First-Token (TTFT) and Time-per-Output-Token (TPOT)
from service rate estimates when direct predictions are not available.

This is the default approach for CARA scheduling. Future work includes
training multi-output models that directly predict (E2E, TTFT, TPOT).
"""
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def derive_ttft_tpot(
    num_prompt_tokens: int,
    num_output_tokens: int,
    e2e_latency_ms: float,
    prefill_rate: Optional[float] = None,
    decode_rate: Optional[float] = None,
    default_prefill_rate: float = 2000.0,
    default_decode_rate: float = 500.0,
) -> Tuple[float, float]:
    """Derive TTFT and TPOT from service rates.

    TTFT (Time-to-First-Token):
        Approximated as prefill time = prompt_tokens / prefill_rate

    TPOT (Time-per-Output-Token):
        Approximated as average decode time per token = decode_time / output_tokens
        where decode_time = E2E_latency - TTFT

    This derivation assumes:
    - TTFT is dominated by prefill (prompt processing)
    - TPOT is average decode time, which can vary during generation
    - Service rates are reasonably accurate (from EMA tracking)

    Future Work:
    ------------
    Train multi-output model that directly predicts (E2E, TTFT, TPOT)
    for better accuracy. The current derivation can be inaccurate when:
    - Queuing delays are significant
    - Prefill is chunked or interleaved with decode
    - Service rates are stale or inaccurate

    See: cara_paper/codex/cara-data-and-experiments.md

    Args:
        num_prompt_tokens: Number of input/prompt tokens
        num_output_tokens: Number of output tokens (predicted or actual)
        e2e_latency_ms: Predicted or actual E2E latency in milliseconds
        prefill_rate: Prefill throughput in tokens/second (from ServiceRateTracker)
        decode_rate: Decode throughput in tokens/second (from ServiceRateTracker)
        default_prefill_rate: Default prefill rate if not provided
        default_decode_rate: Default decode rate if not provided

    Returns:
        Tuple of (ttft_ms, tpot_ms):
        - ttft_ms: Estimated time to first token in milliseconds
        - tpot_ms: Estimated time per output token in milliseconds
    """
    # Use defaults if rates not provided
    prefill_rate = prefill_rate if prefill_rate and prefill_rate > 0 else default_prefill_rate
    decode_rate = decode_rate if decode_rate and decode_rate > 0 else default_decode_rate

    # TTFT from prefill rate
    # TTFT = prompt_tokens / prefill_rate (in seconds) * 1000 (to ms)
    ttft_ms = (num_prompt_tokens / prefill_rate) * 1000 if prefill_rate > 0 else 0

    # Decode time = E2E - TTFT
    decode_time_ms = max(0, e2e_latency_ms - ttft_ms)

    # TPOT from decode time
    # TPOT = decode_time / output_tokens
    if num_output_tokens > 0:
        tpot_ms = decode_time_ms / num_output_tokens
    else:
        # No output tokens, estimate from decode rate
        tpot_ms = 1000 / decode_rate if decode_rate > 0 else 0

    # Sanity checks - clamp to reasonable values
    ttft_ms = max(0, ttft_ms)
    tpot_ms = max(0, tpot_ms)

    # Upper bounds (prevent unreasonable estimates)
    ttft_ms = min(ttft_ms, e2e_latency_ms)
    tpot_ms = min(tpot_ms, 1000)  # Max 1 second per token

    return (ttft_ms, tpot_ms)


def derive_ttft_from_rate(
    num_prompt_tokens: int,
    prefill_rate: Optional[float] = None,
    default_prefill_rate: float = 2000.0,
) -> float:
    """Derive TTFT from prefill rate only.

    Simpler version when E2E latency is not available.

    Args:
        num_prompt_tokens: Number of prompt tokens
        prefill_rate: Prefill throughput in tokens/second
        default_prefill_rate: Default if not provided

    Returns:
        Estimated TTFT in milliseconds
    """
    rate = prefill_rate if prefill_rate and prefill_rate > 0 else default_prefill_rate
    return (num_prompt_tokens / rate) * 1000


def derive_tpot_from_rate(
    decode_rate: Optional[float] = None,
    default_decode_rate: float = 500.0,
) -> float:
    """Derive TPOT from decode rate only.

    Simpler version when E2E latency is not available.

    Args:
        decode_rate: Decode throughput in tokens/second
        default_decode_rate: Default if not provided

    Returns:
        Estimated TPOT in milliseconds
    """
    rate = decode_rate if decode_rate and decode_rate > 0 else default_decode_rate
    return 1000 / rate


class TTFTTPOTDeriver:
    """Class-based interface for TTFT/TPOT derivation.

    Maintains default rates and provides a consistent interface.
    """

    def __init__(
        self,
        default_prefill_rate: float = 2000.0,
        default_decode_rate: float = 500.0,
    ):
        """Initialize deriver with default rates.

        Args:
            default_prefill_rate: Default prefill rate (tokens/sec)
            default_decode_rate: Default decode rate (tokens/sec)
        """
        self.default_prefill_rate = default_prefill_rate
        self.default_decode_rate = default_decode_rate

    def derive(
        self,
        num_prompt_tokens: int,
        num_output_tokens: int,
        e2e_latency_ms: float,
        prefill_rate: Optional[float] = None,
        decode_rate: Optional[float] = None,
    ) -> Tuple[float, float]:
        """Derive TTFT and TPOT.

        Args:
            num_prompt_tokens: Number of prompt tokens
            num_output_tokens: Number of output tokens
            e2e_latency_ms: E2E latency in milliseconds
            prefill_rate: Optional prefill rate override
            decode_rate: Optional decode rate override

        Returns:
            Tuple of (ttft_ms, tpot_ms)
        """
        return derive_ttft_tpot(
            num_prompt_tokens=num_prompt_tokens,
            num_output_tokens=num_output_tokens,
            e2e_latency_ms=e2e_latency_ms,
            prefill_rate=prefill_rate,
            decode_rate=decode_rate,
            default_prefill_rate=self.default_prefill_rate,
            default_decode_rate=self.default_decode_rate,
        )
