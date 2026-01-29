"""
Request-Specific Objectives (RSO) and constraint handling for CARA.

RSOs allow per-request specification of:
- Latency constraints (E2E, TTFT, TPOT)
- Quality requirements (minimum acceptable quality)
- Budget limits (token cost)

These are checked during instance selection and used in tiered scoring.
"""
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum


class ConstraintMode(Enum):
    """Modes for handling constraint violations during instance selection."""
    STRICT = "strict"    # Reject request if no tier-0 candidate
    RELAXED = "relaxed"  # Accept best available candidate
    TIERED = "tiered"    # Try tier-0, then tier-1, etc. Reject only if all empty


@dataclass
class RequestSpecificObjective:
    """Per-request constraints and objectives.

    All fields are optional - if not provided, no constraint is checked
    for that dimension.

    Attributes:
        latency_ms: Maximum acceptable end-to-end latency in milliseconds
        ttft_ms: Maximum acceptable time-to-first-token in milliseconds
        tpot_ms: Maximum acceptable time-per-output-token in milliseconds
        quality_min: Minimum acceptable quality score in [0, 1]
        token_budget: Maximum cost in token units
    """
    latency_ms: Optional[float] = None
    ttft_ms: Optional[float] = None
    tpot_ms: Optional[float] = None
    quality_min: Optional[float] = None
    token_budget: Optional[int] = None

    def has_constraints(self) -> bool:
        """Check if any constraint is specified."""
        return any([
            self.latency_ms is not None,
            self.ttft_ms is not None,
            self.tpot_ms is not None,
            self.quality_min is not None,
            self.token_budget is not None,
        ])

    @classmethod
    def from_dict(cls, data: dict) -> 'RequestSpecificObjective':
        """Create RSO from dictionary."""
        return cls(
            latency_ms=data.get('latency_ms'),
            ttft_ms=data.get('ttft_ms'),
            tpot_ms=data.get('tpot_ms'),
            quality_min=data.get('quality_min') or data.get('model_quality_min'),
            token_budget=data.get('token_budget'),
        )

    def to_dict(self) -> dict:
        """Convert to dictionary (only non-None fields)."""
        result = {}
        if self.latency_ms is not None:
            result['latency_ms'] = self.latency_ms
        if self.ttft_ms is not None:
            result['ttft_ms'] = self.ttft_ms
        if self.tpot_ms is not None:
            result['tpot_ms'] = self.tpot_ms
        if self.quality_min is not None:
            result['quality_min'] = self.quality_min
        if self.token_budget is not None:
            result['token_budget'] = self.token_budget
        return result


@dataclass
class SLOConfig:
    """Cluster-wide Service Level Objectives.

    SLOs apply to all requests and are typically stricter than RSOs.
    When both RSO and SLO exist for a metric, the stricter one applies.

    Attributes:
        latency_ms: Maximum E2E latency SLO in milliseconds
        ttft_ms: Maximum TTFT SLO in milliseconds
        tpot_ms: Maximum TPOT SLO in milliseconds
        tier_weight: Weight for SLO violations in tier calculation.
            Default 1.0 = same as RSO violations.
            Set to 2.0 to make SLO violations count double.
    """
    latency_ms: Optional[float] = None
    ttft_ms: Optional[float] = None
    tpot_ms: Optional[float] = None
    tier_weight: float = 1.0

    @classmethod
    def from_dict(cls, data: dict) -> 'SLOConfig':
        """Create SLOConfig from dictionary."""
        return cls(
            latency_ms=data.get('slo_latency_ms'),
            ttft_ms=data.get('slo_ttft_ms'),
            tpot_ms=data.get('slo_tpot_ms'),
            tier_weight=data.get('slo_tier_weight', 1.0),
        )


@dataclass
class ConstraintViolation:
    """A single constraint violation.

    Attributes:
        constraint_type: Type of constraint violated (e.g., "rso_latency", "slo_ttft")
        violation_amount: How much the constraint is violated by (absolute)
        tier_weight: Weight for tier calculation (RSO=1.0, SLO=configurable)
    """
    constraint_type: str
    violation_amount: float
    tier_weight: float = 1.0


def check_constraints(
    latency_ms: float,
    ttft_ms: float,
    tpot_ms: float,
    quality: float,
    cost_tokens: float,
    rso: Optional[RequestSpecificObjective],
    slo: Optional[SLOConfig],
) -> List[ConstraintViolation]:
    """Check all RSO and SLO constraints.

    Args:
        latency_ms: Predicted E2E latency
        ttft_ms: Predicted/derived TTFT
        tpot_ms: Predicted/derived TPOT
        quality: Estimated quality score
        cost_tokens: Estimated cost in token units
        rso: Request-specific objectives (optional)
        slo: Service-level objectives (optional)

    Returns:
        List of constraint violations (empty list = all constraints satisfied)
    """
    violations = []

    # Check RSO constraints (tier_weight = 1.0)
    if rso is not None:
        if rso.latency_ms is not None and latency_ms > rso.latency_ms:
            violations.append(ConstraintViolation(
                constraint_type="rso_latency",
                violation_amount=latency_ms - rso.latency_ms,
                tier_weight=1.0
            ))

        if rso.ttft_ms is not None and ttft_ms > rso.ttft_ms:
            violations.append(ConstraintViolation(
                constraint_type="rso_ttft",
                violation_amount=ttft_ms - rso.ttft_ms,
                tier_weight=1.0
            ))

        if rso.tpot_ms is not None and tpot_ms > rso.tpot_ms:
            violations.append(ConstraintViolation(
                constraint_type="rso_tpot",
                violation_amount=tpot_ms - rso.tpot_ms,
                tier_weight=1.0
            ))

        if rso.quality_min is not None and quality < rso.quality_min:
            violations.append(ConstraintViolation(
                constraint_type="rso_quality",
                violation_amount=rso.quality_min - quality,
                tier_weight=1.0
            ))

        if rso.token_budget is not None and cost_tokens > rso.token_budget:
            violations.append(ConstraintViolation(
                constraint_type="rso_budget",
                violation_amount=cost_tokens - rso.token_budget,
                tier_weight=1.0
            ))

    # Check SLO constraints (tier_weight = configurable)
    if slo is not None:
        slo_weight = slo.tier_weight

        if slo.latency_ms is not None and latency_ms > slo.latency_ms:
            violations.append(ConstraintViolation(
                constraint_type="slo_latency",
                violation_amount=latency_ms - slo.latency_ms,
                tier_weight=slo_weight
            ))

        if slo.ttft_ms is not None and ttft_ms > slo.ttft_ms:
            violations.append(ConstraintViolation(
                constraint_type="slo_ttft",
                violation_amount=ttft_ms - slo.ttft_ms,
                tier_weight=slo_weight
            ))

        if slo.tpot_ms is not None and tpot_ms > slo.tpot_ms:
            violations.append(ConstraintViolation(
                constraint_type="slo_tpot",
                violation_amount=tpot_ms - slo.tpot_ms,
                tier_weight=slo_weight
            ))

    return violations


def compute_effective_tier(violations: List[ConstraintViolation]) -> float:
    """Compute effective tier from violations.

    Tier = sum of tier_weight for each violation.
    This allows SLO violations to count more than RSO violations.

    Examples:
        - No violations → tier = 0
        - 1 RSO violation (weight=1) → tier = 1
        - 1 SLO violation (weight=2) → tier = 2
        - 1 RSO + 1 SLO (weights 1+2) → tier = 3

    Args:
        violations: List of constraint violations

    Returns:
        Effective tier value (0 = fully feasible)
    """
    return sum(v.tier_weight for v in violations)
