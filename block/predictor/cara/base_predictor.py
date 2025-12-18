"""
Base predictor class for CARA predictors.

Separate from Block's Predictor to use CARA-specific PredictRequest.
"""
from abc import ABC, abstractmethod
from typing import Dict
from block.predictor.cara.data_structures import PredictRequest


class CARABasePredictor(ABC):
    """Base class for CARA predictors.

    Uses PredictRequest instead of Vidur Request to stay independent from Block/Vidur.
    """

    def __init__(self, config, port: int) -> None:
        """Initialize predictor.

        Args:
            config: Predictor configuration (CARABasePredictorConfig)
            port: Backend instance port
        """
        self._config = config
        self._instance_port = port

    @abstractmethod
    async def predict(self, target_request: PredictRequest) -> Dict:
        """Predict metrics for a target request.

        Args:
            target_request: Request information

        Returns:
            Dict with prediction metrics:
            {
                "target_metric": float,  # Lower is better for scheduling
                "gpu_blocks": int,
                "num_requests": int,
                "num_preempted": int,
                "predictor_type": str
            }
        """
        pass