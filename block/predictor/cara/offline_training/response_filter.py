"""
Response filtering for CARA training data preparation.

Filters out low-quality, incomplete, or erroneous responses.
"""

import logging
import zlib
from typing import Tuple
from dataclasses import dataclass
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class ModelResponse:
    """Single model's response to a request."""
    model_name: str
    instance_id: str
    host: str
    generated_text: str
    output_tokens: int
    ttft: float
    server_latency: float
    success: bool
    error: str
    itl: List[float]


class ResponseFilter:
    """Filters out bad/invalid responses.

    Filtering policy (per documented conclusions):
    - Filter: empty, too short (<3 tokens), truncated (hit max_tokens),
              failures/errors
    - Keep: repetitive outputs (model should learn to predict this pattern)
    - Store: compression_ratio as metadata for quality labeling
    """

    def __init__(self,
                 min_output_tokens: int = 3,
                 max_output_tokens: int = 1024):
        """
        Args:
            min_output_tokens: Minimum valid output length
            max_output_tokens: Max output length (responses at this length are truncated)
        """
        self.min_output_tokens = min_output_tokens
        self.max_output_tokens = max_output_tokens

        logger.info(
            f"ResponseFilter initialized: "
            f"min_tokens={min_output_tokens}, max_tokens={max_output_tokens}"
        )

    def is_valid(self, response: ModelResponse) -> Tuple[bool, str]:
        """Check if response is valid.

        Returns:
            (is_valid, reason_if_invalid)
        """
        # Check success flag
        if not response.success:
            return False, f"Response marked as failed: {response.error}"

        # Check for error messages
        if response.error:
            return False, f"Error present: {response.error}"

        # Check minimum length
        if response.output_tokens < self.min_output_tokens:
            return False, f"Too short: {response.output_tokens} tokens"

        # Check if response was truncated (hit max length limit)
        # Truncated responses are incomplete and not suitable for training
        if response.output_tokens >= self.max_output_tokens:
            return False, (
                f"Truncated: output_tokens={response.output_tokens} "
                f"(hit max_tokens={self.max_output_tokens})"
            )

        # NOTE: Repetitive outputs are intentionally NOT filtered.
        # Per documented analysis, repetition is model-inherent and
        # length-dependent (34.5% at 800-1024 tokens). The model should
        # learn to predict this pattern. compression_ratio is stored as
        # metadata instead.

        return True, ""

    def compute_compression_ratio(self, text: str) -> float:
        """Compute compression ratio using zlib.

        Repetitive text has low compression ratio (compresses well).
        This is exposed as a public method for storing as metadata.

        Returns:
            Compression ratio in [0, 1] where:
            - 1.0 = no compression (random/diverse text)
            - 0.0 = perfect compression (highly repetitive)
        """
        return self._compute_compression_ratio(text)

    def _compute_compression_ratio(self, text: str) -> float:
        """Compute compression ratio using zlib.

        Repetitive text has low compression ratio (compresses well).

        Args:
            text: Text to analyze

        Returns:
            Compression ratio in [0, 1] where:
            - 1.0 = no compression (random/diverse text)
            - 0.0 = perfect compression (highly repetitive)
        """
        if not text:
            return 1.0

        # Encode to bytes
        text_bytes = text.encode('utf-8')
        original_size = len(text_bytes)

        if original_size == 0:
            return 1.0

        # Compress with zlib
        compressed = zlib.compress(text_bytes)
        compressed_size = len(compressed)

        # Compute ratio
        ratio = compressed_size / original_size

        return ratio