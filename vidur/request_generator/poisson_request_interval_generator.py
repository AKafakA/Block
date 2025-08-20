import numpy as np

from vidur.config import PoissonRequestIntervalGeneratorConfig
from vidur.request_generator.base_request_interval_generator import (
    BaseRequestIntervalGenerator,
)


class PoissonRequestIntervalGenerator(BaseRequestIntervalGenerator):

    def __init__(self, config: PoissonRequestIntervalGeneratorConfig):
        super().__init__(config)

        self.qps = self.config.qps
        self.std = 1.0 / self.qps

    def get_next_inter_request_time(self) -> float:
        # next_interval = -math.log(1.0 - random.random()) / self.qps
        next_interval = np.random.gamma(shape=1, scale=self.std)

        return next_interval
