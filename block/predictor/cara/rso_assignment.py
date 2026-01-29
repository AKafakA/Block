"""
RSO assignment for benchmarking.

Provides random RSO (Request-Specific Objectives) assignment to requests
for evaluating CARA scheduling under various constraint scenarios.
"""
import random
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List

from block.predictor.cara.rso import RequestSpecificObjective

logger = logging.getLogger(__name__)


@dataclass
class RSOAssignmentConfig:
    """Configuration for random RSO assignment in benchmarks.

    Controls what fraction of requests receive each type of RSO constraint,
    and the ranges for random budget values.

    Example:
        config = RSOAssignmentConfig(
            latency_fraction=0.1,    # 10% get latency RSO
            quality_fraction=0.1,    # 10% get quality RSO
            budget_fraction=0.1,     # 10% get budget RSO
            allow_overlap=True,      # Same request can have multiple RSOs
        )
    """

    # Fraction of requests that get each RSO type (0.0 to 1.0)
    latency_fraction: float = 0.0
    ttft_fraction: float = 0.0
    tpot_fraction: float = 0.0
    quality_fraction: float = 0.0
    budget_fraction: float = 0.0

    # Allow overlapping RSOs on same request
    allow_overlap: bool = True

    # Budget ranges for random assignment
    latency_range_ms: Tuple[float, float] = (200.0, 2000.0)
    ttft_range_ms: Tuple[float, float] = (50.0, 500.0)
    tpot_range_ms: Tuple[float, float] = (10.0, 100.0)
    quality_range: Tuple[float, float] = (0.5, 0.95)
    token_budget_range: Tuple[int, int] = (500, 5000)

    # Distribution for budget values
    # "uniform": uniform random in range
    # "normal": truncated normal centered in range
    distribution: str = "uniform"

    # Random seed for reproducibility (None = non-deterministic)
    seed: Optional[int] = None

    def has_any_rso(self) -> bool:
        """Check if any RSO fraction is set."""
        return any([
            self.latency_fraction > 0,
            self.ttft_fraction > 0,
            self.tpot_fraction > 0,
            self.quality_fraction > 0,
            self.budget_fraction > 0,
        ])

    @classmethod
    def from_dict(cls, data: dict) -> 'RSOAssignmentConfig':
        """Create from dictionary."""
        return cls(
            latency_fraction=data.get('latency_fraction', 0.0),
            ttft_fraction=data.get('ttft_fraction', 0.0),
            tpot_fraction=data.get('tpot_fraction', 0.0),
            quality_fraction=data.get('quality_fraction', 0.0),
            budget_fraction=data.get('budget_fraction', 0.0),
            allow_overlap=data.get('allow_overlap', True),
            latency_range_ms=tuple(data.get('latency_range_ms', (200.0, 2000.0))),
            ttft_range_ms=tuple(data.get('ttft_range_ms', (50.0, 500.0))),
            tpot_range_ms=tuple(data.get('tpot_range_ms', (10.0, 100.0))),
            quality_range=tuple(data.get('quality_range', (0.5, 0.95))),
            token_budget_range=tuple(data.get('token_budget_range', (500, 5000))),
            distribution=data.get('distribution', 'uniform'),
            seed=data.get('seed'),
        )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'latency_fraction': self.latency_fraction,
            'ttft_fraction': self.ttft_fraction,
            'tpot_fraction': self.tpot_fraction,
            'quality_fraction': self.quality_fraction,
            'budget_fraction': self.budget_fraction,
            'allow_overlap': self.allow_overlap,
            'latency_range_ms': list(self.latency_range_ms),
            'ttft_range_ms': list(self.ttft_range_ms),
            'tpot_range_ms': list(self.tpot_range_ms),
            'quality_range': list(self.quality_range),
            'token_budget_range': list(self.token_budget_range),
            'distribution': self.distribution,
            'seed': self.seed,
        }


class RSOAssigner:
    """Assigns random RSOs to requests for benchmarking.

    Randomly assigns RSO constraints to a fraction of requests based on
    the configuration. Used for evaluating CARA scheduling under various
    constraint scenarios.

    Usage:
        config = RSOAssignmentConfig(latency_fraction=0.1, quality_fraction=0.1)
        assigner = RSOAssigner(config)

        for request in requests:
            rso = assigner.assign(request)
            if rso:
                request['request_specific_objective'] = rso.to_dict()

    The assigner can also be used to batch-assign RSOs:
        requests = assigner.assign_batch(requests)
    """

    def __init__(self, config: Optional[RSOAssignmentConfig] = None):
        """Initialize RSO assigner.

        Args:
            config: RSO assignment configuration. If None, no RSOs are assigned.
        """
        self.config = config
        self._rng = random.Random(config.seed if config else None)

        # Statistics
        self._total_requests = 0
        self._rso_assigned = 0
        self._type_counts: Dict[str, int] = {
            'latency': 0,
            'ttft': 0,
            'tpot': 0,
            'quality': 0,
            'budget': 0,
        }

        if config and config.has_any_rso():
            logger.info(
                f"RSOAssigner initialized: "
                f"latency={config.latency_fraction:.1%}, "
                f"ttft={config.ttft_fraction:.1%}, "
                f"tpot={config.tpot_fraction:.1%}, "
                f"quality={config.quality_fraction:.1%}, "
                f"budget={config.budget_fraction:.1%}, "
                f"overlap={config.allow_overlap}"
            )

    def assign(self, request: dict) -> Optional[RequestSpecificObjective]:
        """Assign RSO to a single request based on configuration.

        Args:
            request: Request dictionary (can contain 'prompt', 'prompt_len', etc.)
                     Not modified by this method.

        Returns:
            RequestSpecificObjective if any RSO assigned, None otherwise.
        """
        self._total_requests += 1

        if self.config is None or not self.config.has_any_rso():
            return None

        rso = RequestSpecificObjective()
        any_assigned = False
        cfg = self.config

        # Latency RSO
        if self._rng.random() < cfg.latency_fraction:
            rso.latency_ms = self._sample_value(cfg.latency_range_ms)
            self._type_counts['latency'] += 1
            any_assigned = True
            if not cfg.allow_overlap:
                self._rso_assigned += 1
                return rso

        # TTFT RSO
        if self._rng.random() < cfg.ttft_fraction:
            rso.ttft_ms = self._sample_value(cfg.ttft_range_ms)
            self._type_counts['ttft'] += 1
            any_assigned = True
            if not cfg.allow_overlap:
                self._rso_assigned += 1
                return rso

        # TPOT RSO
        if self._rng.random() < cfg.tpot_fraction:
            rso.tpot_ms = self._sample_value(cfg.tpot_range_ms)
            self._type_counts['tpot'] += 1
            any_assigned = True
            if not cfg.allow_overlap:
                self._rso_assigned += 1
                return rso

        # Quality RSO
        if self._rng.random() < cfg.quality_fraction:
            rso.quality_min = self._sample_value(cfg.quality_range)
            self._type_counts['quality'] += 1
            any_assigned = True
            if not cfg.allow_overlap:
                self._rso_assigned += 1
                return rso

        # Budget RSO
        if self._rng.random() < cfg.budget_fraction:
            rso.token_budget = int(self._sample_value(
                (float(cfg.token_budget_range[0]), float(cfg.token_budget_range[1]))
            ))
            self._type_counts['budget'] += 1
            any_assigned = True

        if any_assigned:
            self._rso_assigned += 1
            return rso

        return None

    def assign_batch(
        self,
        requests: List[dict],
        rso_key: str = 'request_specific_objective'
    ) -> List[dict]:
        """Assign RSOs to a batch of requests.

        Modifies requests in-place by adding RSO field if assigned.

        Args:
            requests: List of request dictionaries
            rso_key: Key name for RSO in request dict

        Returns:
            Same list of requests (modified in-place)
        """
        for request in requests:
            rso = self.assign(request)
            if rso:
                request[rso_key] = rso.to_dict()

        return requests

    def _sample_value(self, range_tuple: Tuple[float, float]) -> float:
        """Sample a value from the given range using configured distribution."""
        low, high = range_tuple

        if self.config.distribution == 'uniform':
            return self._rng.uniform(low, high)
        elif self.config.distribution == 'normal':
            # Truncated normal: mean at center, std = range/4
            mean = (low + high) / 2
            std = (high - low) / 4
            while True:
                val = self._rng.gauss(mean, std)
                if low <= val <= high:
                    return val
        else:
            return self._rng.uniform(low, high)

    def get_stats(self) -> Dict:
        """Get assignment statistics."""
        return {
            'total_requests': self._total_requests,
            'rso_assigned': self._rso_assigned,
            'assignment_rate': self._rso_assigned / self._total_requests if self._total_requests > 0 else 0,
            'type_counts': dict(self._type_counts),
        }

    def reset_stats(self) -> None:
        """Reset statistics counters."""
        self._total_requests = 0
        self._rso_assigned = 0
        self._type_counts = {k: 0 for k in self._type_counts}
