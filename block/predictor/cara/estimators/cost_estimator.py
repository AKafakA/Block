"""
Cost estimator for CARA scheduling.

Calculates request cost based on token counts and model-specific pricing.
"""
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class CostEstimator:
    """Token-based cost estimation for scheduling decisions.

    Calculates cost using the formula:
        cost = input_tokens * input_price + output_tokens * output_price

    For fair comparison across models with different pricing, costs are
    normalized to "equivalent tokens" using a reference price ratio.

    Pricing Configuration:
    ----------------------
    Prices are loaded from model_deployment.json:

    ```json
    {
        "Qwen-2.5-3B": {
            "hf_model_name": "Qwen/Qwen2.5-3B",
            "input_token_price": 0.0001,   // $/1K tokens
            "output_token_price": 0.0002,  // $/1K tokens
            ...
        }
    }
    ```

    If pricing is not specified for a model, defaults are used.

    Normalization:
    --------------
    To compare costs across models fairly, we normalize to "token equivalents":
        cost_tokens = input_tokens + output_tokens * (output_price / input_price)

    This represents the total "work" in units of input tokens.

    Usage:
        prices = {
            "Qwen/Qwen2.5-72B": {"input_token_price": 0.001, "output_token_price": 0.002},
            "Qwen/Qwen2.5-3B": {"input_token_price": 0.0001, "output_token_price": 0.0002},
        }
        estimator = CostEstimator(model_prices=prices)

        # 72B costs 10x more per token
        cost_72b = estimator.predict(100, 200, "Qwen/Qwen2.5-72B")  # Higher
        cost_3b = estimator.predict(100, 200, "Qwen/Qwen2.5-3B")    # Lower
    """

    # Default prices (used when model not in config)
    DEFAULT_INPUT_PRICE = 0.0001   # $/1K tokens
    DEFAULT_OUTPUT_PRICE = 0.0002  # $/1K tokens

    def __init__(
        self,
        model_prices: Optional[Dict[str, Dict[str, float]]] = None,
        default_input_price: float = DEFAULT_INPUT_PRICE,
        default_output_price: float = DEFAULT_OUTPUT_PRICE,
        normalize_to_tokens: bool = True
    ):
        """Initialize cost estimator.

        Args:
            model_prices: Dict mapping model_id to pricing dict with keys:
                - "input_token_price": Price per 1K input tokens
                - "output_token_price": Price per 1K output tokens
            default_input_price: Default input token price
            default_output_price: Default output token price
            normalize_to_tokens: If True, normalize cost to token equivalents
                for fair cross-model comparison. If False, return raw cost.
        """
        self.model_prices = model_prices or {}
        self.default_input_price = default_input_price
        self.default_output_price = default_output_price
        self.normalize_to_tokens = normalize_to_tokens

        logger.info(
            f"CostEstimator initialized: "
            f"models={list(self.model_prices.keys())}, "
            f"defaults=(input={default_input_price}, output={default_output_price})"
        )

    def get_prices(self, model_id: str) -> tuple:
        """Get input and output prices for a model.

        Args:
            model_id: Model identifier

        Returns:
            Tuple of (input_price, output_price)
        """
        # Try exact match
        if model_id in self.model_prices:
            prices = self.model_prices[model_id]
            return (
                prices.get("input_token_price", self.default_input_price),
                prices.get("output_token_price", self.default_output_price)
            )

        # Try partial match (e.g., "Qwen2.5-3B" matches "Qwen-2.5-3B")
        model_lower = model_id.lower()
        for key, prices in self.model_prices.items():
            if key.lower() in model_lower or model_lower in key.lower():
                return (
                    prices.get("input_token_price", self.default_input_price),
                    prices.get("output_token_price", self.default_output_price)
                )

        return (self.default_input_price, self.default_output_price)

    def predict(
        self,
        num_input_tokens: int,
        num_output_tokens: int,
        model_id: str
    ) -> float:
        """Estimate cost for a request.

        Args:
            num_input_tokens: Number of input/prompt tokens
            num_output_tokens: Number of output/completion tokens
            model_id: Model identifier

        Returns:
            Cost estimate. If normalize_to_tokens=True, returns token equivalents.
            Otherwise returns raw cost in pricing units.
        """
        input_price, output_price = self.get_prices(model_id)

        if self.normalize_to_tokens:
            # Normalize to input token equivalents
            # This makes costs comparable across models with different pricing
            output_ratio = output_price / input_price if input_price > 0 else 1.0
            return num_input_tokens + num_output_tokens * output_ratio
        else:
            # Raw cost (in same units as prices, e.g., $/1K tokens)
            return (num_input_tokens * input_price + num_output_tokens * output_price) / 1000

    def predict_raw_cost(
        self,
        num_input_tokens: int,
        num_output_tokens: int,
        model_id: str
    ) -> float:
        """Estimate raw cost (not normalized).

        Args:
            num_input_tokens: Number of input tokens
            num_output_tokens: Number of output tokens
            model_id: Model identifier

        Returns:
            Raw cost in pricing units (e.g., dollars if prices are in $/1K tokens)
        """
        input_price, output_price = self.get_prices(model_id)
        return (num_input_tokens * input_price + num_output_tokens * output_price) / 1000

    @classmethod
    def from_model_deployment(cls, model_config: Dict) -> 'CostEstimator':
        """Create CostEstimator from model_deployment.json config.

        Args:
            model_config: Loaded model_deployment.json dict

        Returns:
            Configured CostEstimator
        """
        model_prices = {}

        for model_name, config in model_config.items():
            hf_name = config.get("hf_model_name", model_name)
            input_price = config.get("input_token_price")
            output_price = config.get("output_token_price")

            if input_price is not None or output_price is not None:
                model_prices[hf_name] = {
                    "input_token_price": input_price or cls.DEFAULT_INPUT_PRICE,
                    "output_token_price": output_price or cls.DEFAULT_OUTPUT_PRICE,
                }
                # Also add short name mapping
                model_prices[model_name] = model_prices[hf_name]

        return cls(model_prices=model_prices)
