#!/usr/bin/env python3
"""
Roofline (analytical) latency predictor for CARA.

Simple formula-based baseline:
    latency = backlog_time + prefill_time + decode_time + overhead

Uses static rates calibrated per (model, GPU-type) from observed throughput data.
Falls back to default rates if calibration data is unavailable.

Optionally wraps NVIDIA AIConfigurator for more principled roofline analysis.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Default rates per (model_key, gpu_type) — tokens/sec
# These are rough estimates; calibrate with actual measurements
DEFAULT_RATES = {
    # (prefill_tok_per_s, decode_tok_per_s, iteration_overhead_ms)
    "qwen2.5-3b_p100": (1500, 80, 5),
    "qwen2.5-3b_a30": (3000, 150, 3),
    "qwen2.5-7b_a30": (2000, 100, 4),
    "qwen2.5-14b_v100": (1200, 60, 6),
    "qwen2.5-72b_a100": (800, 30, 8),
}


class RooflineLatencyPredictor:
    """Analytical latency predictor using simple roofline model.

    latency = queue_wait + prefill_time + decode_time + overhead
    where:
        queue_wait = pending_tokens / throughput
        prefill_time = prompt_tokens / prefill_rate
        decode_time = output_tokens / decode_rate
        overhead = per-iteration overhead * estimated_iterations
    """

    def __init__(self, rates: Optional[Dict] = None):
        """
        Args:
            rates: Dict mapping instance_type to (prefill_tok_s, decode_tok_s, overhead_ms).
                   If None, uses DEFAULT_RATES.
        """
        self.rates = rates or dict(DEFAULT_RATES)

    def predict(
        self,
        instance_type: str,
        schedule_state: Dict,
        num_prompt_tokens: int,
        num_predicted_output_tokens: int,
    ) -> Dict[str, float]:
        """Predict latency using roofline model."""
        rates = self.rates.get(instance_type)
        if rates is None:
            # Use EMA rates from schedule state if available
            prefill_rate = schedule_state.get("ema_prefill_tok_per_s", 1000)
            decode_rate = schedule_state.get("ema_decode_tok_per_s", 50)
            overhead_ms = 5.0
        else:
            prefill_rate, decode_rate, overhead_ms = rates

        prefill_rate = max(prefill_rate, 1.0)
        decode_rate = max(decode_rate, 1.0)

        # Queue wait time
        pending_prefill = schedule_state.get("pending_prefill_tokens", 0)
        pending_decode = schedule_state.get("pending_decode_tokens", 0)
        queue_wait = pending_prefill / prefill_rate + pending_decode / decode_rate

        # Request processing time
        prefill_time = num_prompt_tokens / prefill_rate
        decode_time = num_predicted_output_tokens / decode_rate

        # Overhead (one per decode iteration)
        overhead = (overhead_ms / 1000.0) * num_predicted_output_tokens

        total = queue_wait + prefill_time + decode_time + overhead

        return {
            "e2e_latency": total,
            "queue_wait": queue_wait,
            "prefill_time": prefill_time,
            "decode_time": decode_time,
            "overhead": overhead,
        }

    def calibrate(self, instance_type: str, records: list, target: str = "e2el"):
        """Calibrate rates from observed latency data.

        Estimates prefill and decode rates from actual measurements.
        """
        import numpy as np

        prefill_rates = []
        decode_rates = []

        for rec in records:
            target_val = rec.get(target) or rec.get("actual_e2e_latency")
            if target_val is None or target_val <= 0:
                continue

            ttft = rec.get("ttft") or rec.get("actual_ttft")
            num_prompt = rec.get("num_prompt_tokens") or rec.get("input_len", 0)
            num_output = rec.get("output_len", 0)

            if ttft and ttft > 0 and num_prompt > 0:
                prefill_rates.append(num_prompt / ttft)

            if num_output > 0 and target_val > 0:
                decode_time = target_val - (ttft or 0)
                if decode_time > 0:
                    decode_rates.append(num_output / decode_time)

        if prefill_rates and decode_rates:
            self.rates[instance_type] = (
                float(np.median(prefill_rates)),
                float(np.median(decode_rates)),
                5.0,  # Default overhead
            )
            logger.info(
                f"Calibrated {instance_type}: "
                f"prefill={self.rates[instance_type][0]:.0f} tok/s, "
                f"decode={self.rates[instance_type][1]:.0f} tok/s"
            )

    def save(self, output_dir: str):
        """Save calibrated rates."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        serializable_rates = {
            k: {"prefill_tok_s": v[0], "decode_tok_s": v[1], "overhead_ms": v[2]}
            for k, v in self.rates.items()
        }
        with open(output_path / "roofline_rates.json", "w") as f:
            json.dump(serializable_rates, f, indent=2)

        logger.info(f"Roofline predictor saved to {output_path}")

    @classmethod
    def load(cls, model_dir: str) -> "RooflineLatencyPredictor":
        """Load calibrated rates."""
        with open(Path(model_dir) / "roofline_rates.json") as f:
            raw_rates = json.load(f)

        rates = {
            k: (v["prefill_tok_s"], v["decode_tok_s"], v["overhead_ms"])
            for k, v in raw_rates.items()
        }
        predictor = cls(rates=rates)
        logger.info(f"Roofline predictor loaded: {len(rates)} instance types")
        return predictor
