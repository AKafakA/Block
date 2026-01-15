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

    Checks for:
    - Success/error flags
    - Minimum output length
    - Truncated responses (hitting max_length)
    - High repetition using compression ratio
    """

    def __init__(self,
                 min_output_tokens: int = 3,
                 max_output_tokens: int = 1024,
                 min_compression_ratio: float = 0.2):
        """
        Args:
            min_output_tokens: Minimum valid output length
            max_output_tokens: Max output length (responses at this length are truncated)
            min_compression_ratio: Minimum compression ratio.
                                  Lower ratio = more repetition.
                                  Typical values: 0.2-0.3 threshold
                                  (repetitive text compresses to <20% of original)
        """
        self.min_output_tokens = min_output_tokens
        self.max_output_tokens = max_output_tokens
        self.min_compression_ratio = min_compression_ratio

        logger.info(
            f"ResponseFilter initialized: "
            f"min_tokens={min_output_tokens}, max_tokens={max_output_tokens}, "
            f"min_compression_ratio={min_compression_ratio}"
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
        if response.output_tokens >= self.max_output_tokens:
            return False, (
                f"Truncated: output_tokens={response.output_tokens} "
                f"(hit max_length={self.max_output_tokens})"
            )

        # Check for excessive repetition using compression ratio
        if response.generated_text:
            compression_ratio = self._compute_compression_ratio(response.generated_text)
            if compression_ratio < self.min_compression_ratio:
                return False, (
                    f"High repetition detected: "
                    f"compression_ratio={compression_ratio:.3f} "
                    f"(threshold={self.min_compression_ratio})"
                )
            logger.debug(
                f"{response.model_name}: compression_ratio={compression_ratio:.3f}"
            )

        return True, ""

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