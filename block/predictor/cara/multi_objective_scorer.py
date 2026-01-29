"""
Multi-objective scoring for CARA instance selection.

Implements tiered constraint handling with weighted scoring:
1. Check RSO and SLO constraints
2. Compute effective tier (sum of weighted violations)
3. Normalize metrics (min-max or regret)
4. Compute weighted score
5. Sort by (tier, score) and select best

This replaces simple penalty-based scoring with proper tiered selection.
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum

from block.predictor.cara.rso import (
    RequestSpecificObjective,
    SLOConfig,
    ConstraintMode,
    ConstraintViolation,
    check_constraints,
    compute_effective_tier,
)

logger = logging.getLogger(__name__)


class NormalizationMethod(Enum):
    """Normalization methods for multi-objective scoring."""
    MINMAX = "minmax"      # Per-batch min-max scaling to [0, 1]
    REGRET = "regret"      # Regret vs best candidate


@dataclass
class ScoringConfig:
    """Configuration for multi-objective scoring."""

    # Objective weights (all should be >= 0, lower score is better)
    w_latency: float = 1.0
    w_cost: float = 0.1
    w_quality: float = 0.0   # 0 = disabled (quality not in scoring)
    w_memory: float = 0.1    # Memory utilization balance

    # Normalization method
    normalization: NormalizationMethod = NormalizationMethod.MINMAX

    # Constraint handling mode
    constraint_mode: ConstraintMode = ConstraintMode.RELAXED

    # SLO configuration (cluster-wide)
    slo: Optional[SLOConfig] = None

    @classmethod
    def from_dict(cls, data: dict) -> 'ScoringConfig':
        """Create from dictionary."""
        slo_data = {
            'slo_latency_ms': data.get('slo_latency_ms'),
            'slo_ttft_ms': data.get('slo_ttft_ms'),
            'slo_tpot_ms': data.get('slo_tpot_ms'),
            'slo_tier_weight': data.get('slo_tier_weight', 1.0),
        }
        slo = SLOConfig.from_dict(slo_data) if any(v for v in slo_data.values() if v is not None) else None

        norm_str = data.get('normalization', 'minmax')
        try:
            normalization = NormalizationMethod(norm_str)
        except ValueError:
            normalization = NormalizationMethod.MINMAX

        mode_str = data.get('constraint_mode', 'relaxed')
        try:
            constraint_mode = ConstraintMode(mode_str)
        except ValueError:
            constraint_mode = ConstraintMode.RELAXED

        return cls(
            w_latency=data.get('w_latency', 1.0),
            w_cost=data.get('w_cost', 0.1),
            w_quality=data.get('w_quality', 0.0),
            w_memory=data.get('w_memory', 0.1),
            normalization=normalization,
            constraint_mode=constraint_mode,
            slo=slo,
        )


@dataclass
class CandidateScore:
    """Score for a candidate instance.

    Contains all metrics, constraint violations, and computed scores
    for a single (request, instance) pair.
    """
    instance_id: str
    model_id: str

    # Raw predictions/estimates
    latency_ms: float = 0.0
    ttft_ms: float = 0.0
    tpot_ms: float = 0.0
    quality: float = 1.0
    cost_tokens: float = 0.0
    memory_util: float = 0.0

    # Constraint handling
    violations: List[ConstraintViolation] = field(default_factory=list)
    effective_tier: float = 0.0

    # Normalized scores (computed during scoring)
    lat_norm: float = 0.0
    cost_norm: float = 0.0
    qual_norm: float = 0.0
    mem_norm: float = 0.0

    # Final scores
    base_score: float = 0.0
    final_score: float = 0.0


class MultiObjectiveScorer:
    """Scores instances for request assignment using multiple objectives.

    Features:
    - Tiered constraint handling (RSO + SLO)
    - Configurable weights for latency, cost, quality, memory
    - Multiple normalization methods (min-max, regret)
    - Selection modes (STRICT, RELAXED, TIERED)

    Usage:
        config = ScoringConfig(w_latency=1.0, w_cost=0.1)
        scorer = MultiObjectiveScorer(config)

        # Build candidates from predictions
        candidates = [
            CandidateScore(instance_id="inst-0", latency_ms=100, ...),
            CandidateScore(instance_id="inst-1", latency_ms=150, ...),
        ]

        # Score with RSO constraints
        rso = RequestSpecificObjective(latency_ms=120)
        scorer.score_candidates(candidates, rso)

        # Select best
        best_id = scorer.select_best(candidates)
    """

    def __init__(self, config: ScoringConfig):
        """Initialize scorer.

        Args:
            config: Scoring configuration
        """
        self.config = config
        logger.info(
            f"MultiObjectiveScorer initialized: "
            f"weights=(lat={config.w_latency}, cost={config.w_cost}, "
            f"qual={config.w_quality}, mem={config.w_memory}), "
            f"norm={config.normalization.value}, "
            f"mode={config.constraint_mode.value}"
        )

    def score_candidates(
        self,
        candidates: List[CandidateScore],
        rso: Optional[RequestSpecificObjective] = None,
    ) -> None:
        """Score all candidate instances.

        Modifies candidates in-place with:
        - violations: List of constraint violations
        - effective_tier: Tier based on weighted violations
        - *_norm: Normalized metric values
        - base_score: Weighted sum of normalized metrics
        - final_score: Same as base_score (tier handled in selection)

        Args:
            candidates: List of CandidateScore to score
            rso: Request-specific objectives (optional)
        """
        if not candidates:
            return

        # Step 1: Check constraints for each candidate
        for c in candidates:
            c.violations = check_constraints(
                latency_ms=c.latency_ms,
                ttft_ms=c.ttft_ms,
                tpot_ms=c.tpot_ms,
                quality=c.quality,
                cost_tokens=c.cost_tokens,
                rso=rso,
                slo=self.config.slo,
            )
            c.effective_tier = compute_effective_tier(c.violations)

        # Step 2: Normalize metrics
        self._normalize(candidates)

        # Step 3: Compute weighted scores
        for c in candidates:
            c.base_score = (
                self.config.w_latency * c.lat_norm +
                self.config.w_cost * c.cost_norm +
                self.config.w_quality * c.qual_norm +
                self.config.w_memory * c.mem_norm
            )
            c.final_score = c.base_score

    def select_best(self, candidates: List[CandidateScore]) -> Optional[str]:
        """Select best instance using tiered constraint handling.

        Selection is based on (effective_tier, final_score) - lower is better.
        The constraint_mode determines how to handle cases with no tier-0 candidates.

        Args:
            candidates: Scored candidates

        Returns:
            Best instance ID, or None if no valid candidate
        """
        if not candidates:
            return None

        # Sort by tier first, then by score
        sorted_candidates = sorted(
            candidates,
            key=lambda c: (c.effective_tier, c.final_score)
        )

        mode = self.config.constraint_mode

        if mode == ConstraintMode.STRICT:
            # Only accept tier 0 (all constraints satisfied)
            tier0 = [c for c in sorted_candidates if c.effective_tier == 0]
            if not tier0:
                logger.warning("STRICT mode: No tier-0 candidates, rejecting request")
                return None
            return tier0[0].instance_id

        elif mode == ConstraintMode.RELAXED:
            # Accept best available (lowest tier, then lowest score)
            return sorted_candidates[0].instance_id

        elif mode == ConstraintMode.TIERED:
            # Try each tier in order, reject only if all tiers empty
            if not sorted_candidates:
                return None

            max_tier = int(max(c.effective_tier for c in sorted_candidates))
            for target_tier in range(max_tier + 1):
                tier_candidates = [c for c in sorted_candidates
                                  if int(c.effective_tier) == target_tier]
                if tier_candidates:
                    return tier_candidates[0].instance_id

            # All tiers checked, no candidates
            logger.warning("TIERED mode: All tiers empty, rejecting request")
            return None

        else:
            # Default to relaxed
            return sorted_candidates[0].instance_id

    def _normalize(self, candidates: List[CandidateScore]) -> None:
        """Normalize metrics using configured method."""
        if self.config.normalization == NormalizationMethod.MINMAX:
            self._normalize_minmax(candidates)
        elif self.config.normalization == NormalizationMethod.REGRET:
            self._normalize_regret(candidates)

    def _normalize_minmax(self, candidates: List[CandidateScore]) -> None:
        """Per-batch min-max normalization to [0, 1].

        Lower values are better for latency, cost, memory.
        Higher values are better for quality (inverted in normalization).
        """
        eps = 1e-9

        # Extract ranges
        lat_vals = [c.latency_ms for c in candidates]
        cost_vals = [c.cost_tokens for c in candidates]
        qual_vals = [c.quality for c in candidates]
        mem_vals = [c.memory_util for c in candidates]

        lat_min, lat_max = min(lat_vals), max(lat_vals)
        cost_min, cost_max = min(cost_vals), max(cost_vals)
        qual_min, qual_max = min(qual_vals), max(qual_vals)
        mem_min, mem_max = min(mem_vals), max(mem_vals)

        lat_range = lat_max - lat_min + eps
        cost_range = cost_max - cost_min + eps
        qual_range = qual_max - qual_min + eps
        mem_range = mem_max - mem_min + eps

        for c in candidates:
            # Lower is better (0 = best, 1 = worst)
            c.lat_norm = (c.latency_ms - lat_min) / lat_range
            c.cost_norm = (c.cost_tokens - cost_min) / cost_range
            c.mem_norm = (c.memory_util - mem_min) / mem_range

            # Quality: higher is better, so invert (0 = best quality)
            c.qual_norm = 1.0 - (c.quality - qual_min) / qual_range

    def _normalize_regret(self, candidates: List[CandidateScore]) -> None:
        """Regret normalization vs best candidate.

        Score = how much worse than the best candidate.
        """
        eps = 1e-9

        # Find best values
        best_lat = min(c.latency_ms for c in candidates)
        best_cost = min(c.cost_tokens for c in candidates)
        best_qual = max(c.quality for c in candidates)  # Higher is better
        best_mem = min(c.memory_util for c in candidates)

        for c in candidates:
            # Regret = (value - best) / best
            c.lat_norm = (c.latency_ms - best_lat) / (best_lat + eps)
            c.cost_norm = (c.cost_tokens - best_cost) / (best_cost + eps)
            c.mem_norm = (c.memory_util - best_mem) / (best_mem + eps)

            # Quality: regret for not being best
            c.qual_norm = (best_qual - c.quality) / (best_qual + eps)

    def get_stats(self) -> Dict:
        """Get scorer statistics."""
        return {
            "weights": {
                "latency": self.config.w_latency,
                "cost": self.config.w_cost,
                "quality": self.config.w_quality,
                "memory": self.config.w_memory,
            },
            "normalization": self.config.normalization.value,
            "constraint_mode": self.config.constraint_mode.value,
            "slo": {
                "latency_ms": self.config.slo.latency_ms if self.config.slo else None,
                "ttft_ms": self.config.slo.ttft_ms if self.config.slo else None,
                "tpot_ms": self.config.slo.tpot_ms if self.config.slo else None,
                "tier_weight": self.config.slo.tier_weight if self.config.slo else 1.0,
            }
        }
