import atexit
import os
from typing import List, Tuple

from vidur.config import (
    BlockOfflineGlobalSchedulerConfig,
    BlockStarOfflineGlobalSchedulerConfig,
)
from vidur.entities import Request
from vidur.request_timeline_predictor.base_request_timeline_predictor import (
    get_target_metric_value,
)
from vidur.request_timeline_predictor.deterministic_noise import (
    DeterministicNoiseProvider,
)
from vidur.request_timeline_predictor.request_timeline_predictor_registry import (
    RequestTimelinePredictorRegistry,
)
from vidur.scheduler.global_scheduler.base_global_scheduler import BaseGlobalScheduler
from vidur.scheduler.global_scheduler.process_workers import (
    ProcessParallelEvaluator,
    WorkerInitData,
)
from vidur.scheduler.utils.replica_state import snapshot_replica_scheduler_state
from vidur.types.optimal_global_scheduler_target_metric import TargetMetric


class BlockOfflineGlobalScheduler(BaseGlobalScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        global_config = self._config.cluster_config.global_scheduler_config
        if not isinstance(
            global_config,
            (BlockOfflineGlobalSchedulerConfig, BlockStarOfflineGlobalSchedulerConfig),
        ):
            raise ValueError("Invalid global scheduler config for Block offline scheduler")

        self._target_metric = TargetMetric.from_str(global_config.target_metric)
        predictor_config = global_config.request_timeline_predictor_config
        noise_fraction = getattr(predictor_config, "noise_fraction", 0.0)
        deterministic_noise = getattr(global_config, "deterministic_noise", False)
        self._parallel = getattr(global_config, "parallel", False)
        # Thread backend removed due to nondeterminism; always use process when parallel is enabled.
        self._parallel_backend = "process" if self._parallel else "none"
        self._parallel_workers = 0

        if self._parallel and noise_fraction > 0 and not deterministic_noise:
            raise ValueError(
                "Parallel Block offline scheduler requires deterministic noise when noise_fraction > 0. "
                "Set --deterministic-noise on or disable noise with --block-noise 0."
            )

        self._request_timeline_predictor = RequestTimelinePredictorRegistry.get(
            predictor_config.get_type()
        )

        configure = getattr(self._request_timeline_predictor, "configure", None)
        if callable(configure):
            configure(
                noise_fraction=noise_fraction,
                noise_distribution=getattr(predictor_config, "noise_distribution", "uniform"),
                random_seed=getattr(predictor_config, "random_seed", -1),
            )

        if deterministic_noise:
            noise_provider = DeterministicNoiseProvider(
                noise_fraction=noise_fraction,
                seed=getattr(self._config, "seed", 0),
            )
        else:
            noise_provider = None

        set_noise_provider = getattr(
            self._request_timeline_predictor, "set_noise_provider", None
        )
        if callable(set_noise_provider):
            set_noise_provider(noise_provider)

        self._request_timeline_predictor.attach_execution_time_predictor(
            self._execution_time_predictor
        )
        if hasattr(self._request_timeline_predictor, "use_estimated_time"):
            self._request_timeline_predictor.use_estimated_time = True

        # Optional fast mode via config (enabled by default for offline configs).
        if getattr(global_config, "fast_predict", False):
            disable_copy = getattr(
                self._request_timeline_predictor,
                "disable_copy_of_base_replica_scheduler",
                None,
            )
            if callable(disable_copy):
                disable_copy()

        enable_parallel_mode = getattr(
            self._request_timeline_predictor, "enable_parallel_mode", None
        )

        self._process_evaluator: ProcessParallelEvaluator | None = None
        if self._parallel:
            # Process-based parallel evaluator
            if callable(enable_parallel_mode):
                enable_parallel_mode(False)
            init_data: dict[int, WorkerInitData] = {}
            metrics_config = self._config.metrics_config
            simulation_seed = getattr(self._config, "seed", 0)
            predictor_random_seed = getattr(predictor_config, "random_seed", -1)
            noise_distribution = getattr(
                predictor_config, "noise_distribution", "uniform"
            )
            threshold_batch_size = getattr(
                self._request_timeline_predictor,
                "threshold_batch_size_for_time_estimation",
                36,
            )
            use_estimated_time = getattr(
                self._request_timeline_predictor, "use_estimated_time", True
            )
            for replica_scheduler in self._replica_schedulers.values():
                replica_id = replica_scheduler.replica_id
                init_data[replica_id] = WorkerInitData(
                    replica_id=replica_id,
                    replica_config=self._config.cluster_config.replica_config,
                    replica_scheduler_config=self._config.cluster_config.replica_scheduler_config,
                    request_generator_config=self._config.request_generator_config,
                    execution_time_predictor_config=self._config.execution_time_predictor_config,
                    metrics_config=metrics_config,
                    predictor_type=predictor_config.get_type(),
                    target_metric=global_config.target_metric,
                    fast_predict=getattr(global_config, "fast_predict", False),
                    deterministic_noise=deterministic_noise,
                    noise_fraction=noise_fraction,
                    noise_distribution=noise_distribution,
                    predictor_random_seed=predictor_random_seed,
                    simulation_seed=simulation_seed,
                    threshold_batch_size=threshold_batch_size,
                    use_estimated_time=use_estimated_time,
                )
            self._process_evaluator = ProcessParallelEvaluator(init_data)
        elif callable(enable_parallel_mode):
            enable_parallel_mode(False)

        self._replica_scheduler_list = list(self._replica_schedulers.values())

    def schedule(self) -> List[Tuple[int, Request]]:
        self.sort_requests()

        request_mapping: List[Tuple[int, Request]] = []

        while self._request_queue:
            request = self._request_queue.pop(0)
            if not self._parallel:
                metric_items = [
                    (
                        replica_scheduler.replica_id,
                        get_target_metric_value(
                            self._target_metric,
                            replica_scheduler,
                            request,
                            self._request_timeline_predictor,
                        ),
                    )
                    for replica_scheduler in self._replica_scheduler_list
                ]
            else:
                evaluator = self._process_evaluator
                if evaluator is None:
                    raise RuntimeError("Process evaluator not initialized")
                scheduler_items = [
                    (replica_scheduler.replica_id, replica_scheduler)
                    for replica_scheduler in self._replica_scheduler_list
                ]
                metric_items = evaluator.evaluate_request(
                    request,
                    scheduler_items,
                    snapshot_replica_scheduler_state,
                )

            metric_map = dict(metric_items)

            if self._target_metric.name.startswith("MAX"):
                selected_replica_id = max(metric_map.items(), key=lambda kv: kv[1])[0]
            else:
                selected_replica_id = min(metric_map.items(), key=lambda kv: kv[1])[0]

            request_mapping.append((selected_replica_id, request))

        return request_mapping
