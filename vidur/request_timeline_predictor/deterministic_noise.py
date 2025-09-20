"""Deterministic noise support for parallel what-if simulations.

Provides stable multiplicative noise factors keyed by
(request_id, replica_id, metric, element_index) so that enabling
parallel evaluation does not change the behaviour relative to the
sequential random draw implementation.
"""

from __future__ import annotations

import hashlib
from typing import Optional


class DeterministicNoiseProvider:
    def __init__(self, noise_fraction: float, seed: int) -> None:
        self._noise_fraction = max(noise_fraction, 0.0)
        self._seed = seed if seed is not None else 0

    def get_multiplier(
        self,
        request_id: int,
        replica_id: int,
        metric_name: str,
        element_index: Optional[int] = None,
    ) -> float:
        """Return a deterministic multiplier in [1-f, 1+f]."""

        if self._noise_fraction == 0.0:
            return 1.0

        key = f"{self._seed}:{request_id}:{replica_id}:{metric_name}:{element_index or 0}".encode(
            "utf-8"
        )
        digest = hashlib.sha256(key).digest()
        rand = int.from_bytes(digest[:8], "big") / (1 << 64)
        delta = (rand * 2.0) - 1.0  # in [-1, 1]
        multiplier = 1.0 + delta * self._noise_fraction
        return multiplier if multiplier > 0.0 else 0.0
