from typing import Dict

from vidur.entities import Request
from block.predictor.predictor_config import PredictorConfig
from abc import ABC, abstractmethod


class Predictor(ABC):
    """Predictor class that extends the Simulator class which is used to predict the completion time of the request.
       for single model instances.
    """

    def __init__(self, config: PredictorConfig, port: int) -> None:
        self._config = config
        self._current_time = 0
        self._last_request = None
        self._instance_port = port

    @abstractmethod
    async def predict(self, target_request: Request) -> Dict:
        """Predict the completion time of the request.
        """
        pass
