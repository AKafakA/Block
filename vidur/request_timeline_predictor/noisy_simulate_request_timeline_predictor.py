import random
from typing import Iterable, Optional, Sequence, Union

from vidur.request_timeline_predictor.simulate_request_timeline_predictor import (
    SimulateRequestTimelinePredictor,
)
from vidur.request_timeline_predictor.deterministic_noise import (
    DeterministicNoiseProvider,
)


Numeric = Union[float, int]


class NoisySimulateRequestTimelinePredictor(SimulateRequestTimelinePredictor):
    """Wraps the simulated predictor and injects multiplicative noise."""

    def __init__(
        self,
        noise_fraction: float = 0.1,
        noise_distribution: str = "uniform",
        random_seed: int = -1,
        noise_provider: Optional[DeterministicNoiseProvider] = None,
    ) -> None:
        super().__init__()
        self._noise_fraction = max(noise_fraction, 0.0)
        self._noise_distribution = noise_distribution
        self._rng = random.Random()
        if random_seed >= 0:
            self._rng.seed(random_seed)
        self._noise_provider = noise_provider

    def set_noise_provider(
        self, noise_provider: Optional[DeterministicNoiseProvider]
    ) -> None:
        self._noise_provider = noise_provider

    def configure(
        self,
        noise_fraction: float,
        noise_distribution: str = "uniform",
        random_seed: int = -1,
        noise_provider: Optional[DeterministicNoiseProvider] = None,
    ) -> None:
        self._noise_fraction = max(noise_fraction, 0.0)
        self._noise_distribution = noise_distribution
        if random_seed >= 0:
            self._rng.seed(random_seed)
        self._noise_provider = noise_provider

    def _noisify(self, value: Numeric) -> float:
        if self._noise_fraction == 0:
            return float(value)
        if self._noise_distribution != "uniform":
            raise ValueError(
                f"Unsupported noise distribution {self._noise_distribution}"
            )
        delta = self._rng.uniform(-self._noise_fraction, self._noise_fraction)
        return max(float(value) * (1 + delta), 0.0)

    def _apply_noise(
        self,
        value: Union[Numeric, Sequence[Numeric]],
        replica_scheduler,
        request,
        metric_name: str,
    ) -> Union[float, Sequence[float]]:
        if self._noise_provider is not None:
            replica_id = getattr(replica_scheduler, "replica_id", -1)
            if isinstance(value, Iterable) and not isinstance(value, (float, int)):
                return type(value)(
                    self._noise_provider.get_multiplier(
                        request.id,
                        replica_id,
                        metric_name,
                        idx,
                    )
                    * v
                    for idx, v in enumerate(value)
                )  # type: ignore[arg-type]
            multiplier = self._noise_provider.get_multiplier(
                request.id, replica_id, metric_name
            )
            return multiplier * float(value)  # type: ignore[arg-type]

        if isinstance(value, Iterable) and not isinstance(value, (float, int)):
            return type(value)(self._noisify(v) for v in value)  # type: ignore[arg-type]
        return self._noisify(value)  # type: ignore[arg-type]

    def predict_avg_block_size(self, replica_scheduler, request):
        value = super().predict_avg_block_size(replica_scheduler, request)
        return self._apply_noise(value, replica_scheduler, request, "avg_block_size")

    def predict_request_scheduling_delay(self, replica_scheduler, request):
        value = super().predict_request_scheduling_delay(replica_scheduler, request)
        return self._apply_noise(
            value, replica_scheduler, request, "request_scheduling_delay"
        )

    def predict_request_makespan(self, replica_scheduler, request):
        value = super().predict_request_makespan(replica_scheduler, request)
        return self._apply_noise(value, replica_scheduler, request, "request_makespan")

    def predict_average_latency(self, replica_scheduler, request):
        value = super().predict_average_latency(replica_scheduler, request)
        return self._apply_noise(value, replica_scheduler, request, "average_latency")

    def predict_average_batch_size(self, replica_scheduler, request):
        value = super().predict_average_batch_size(replica_scheduler, request)
        return self._apply_noise(value, replica_scheduler, request, "average_batch_size")

    def predict_average_execution_latency(self, replica_scheduler, request):
        value = super().predict_average_execution_latency(replica_scheduler, request)
        return self._apply_noise(
            value, replica_scheduler, request, "average_execution_latency"
        )

    def predict_waiting_and_ending_time(self, replica_scheduler, request):
        value = super().predict_waiting_and_ending_time(replica_scheduler, request)
        return self._apply_noise(
            value, replica_scheduler, request, "waiting_and_ending_time"
        )
