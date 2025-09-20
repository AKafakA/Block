from __future__ import annotations

import atexit
import copy
import multiprocessing
import time
from dataclasses import dataclass
from multiprocessing.connection import Connection, wait
from typing import Any, Dict, Iterable, List, Optional, Tuple

from vidur.config import (
    BaseRequestGeneratorConfig,
    BaseReplicaSchedulerConfig,
    BaseExecutionTimePredictorConfig,
    MetricsConfig,
    ReplicaConfig,
)
from vidur.entities import Replica, Request
from vidur.execution_time_predictor import ExecutionTimePredictorRegistry
from vidur.request_timeline_predictor.base_request_timeline_predictor import (
    get_target_metric_value,
)
from vidur.request_timeline_predictor.deterministic_noise import (
    DeterministicNoiseProvider,
)
from vidur.request_timeline_predictor.request_timeline_predictor_registry import (
    RequestTimelinePredictorRegistry,
)
from vidur.scheduler.replica_scheduler.base_replica_scheduler import (
    BaseReplicaScheduler,
)
from vidur.scheduler.replica_scheduler.replica_scheduler_registry import (
    ReplicaSchedulerRegistry,
)
from vidur.scheduler.utils.replica_state import restore_replica_scheduler_state
from vidur.types.optimal_global_scheduler_target_metric import TargetMetric


EVALUATION_TIMEOUT_SECS = 120.0


@dataclass
class WorkerInitData:
    replica_id: int
    replica_config: ReplicaConfig
    replica_scheduler_config: BaseReplicaSchedulerConfig
    request_generator_config: BaseRequestGeneratorConfig
    execution_time_predictor_config: BaseExecutionTimePredictorConfig
    metrics_config: MetricsConfig
    predictor_type: str
    target_metric: str
    fast_predict: bool
    deterministic_noise: bool
    noise_fraction: float
    noise_distribution: str
    predictor_random_seed: int
    simulation_seed: int
    threshold_batch_size: int
    use_estimated_time: bool


class ProcessReplicaWorker:
    def __init__(self, init_data: WorkerInitData, timeout: float = EVALUATION_TIMEOUT_SECS) -> None:
        self._timeout = timeout
        ctx = multiprocessing.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe()
        self._conn: Connection = parent_conn
        self._process = ctx.Process(target=_worker_main, args=(child_conn, init_data))
        self._process.daemon = True
        self._process.start()
        self._closed = False

    @property
    def connection(self) -> Connection:
        return self._conn

    def evaluate(self, request: Request, snapshot: Dict[str, Any]) -> None:
        payload = {
            "kind": "evaluate",
            "request": copy.deepcopy(request),
            "snapshot": snapshot,
        }
        self._conn.send(payload)

    def recv(self) -> Tuple[str, Optional[float], Optional[str]]:
        if not self._conn.poll(self._timeout):
            return ("timeout", None, None)
        message = self._conn.recv()
        return (message.get("status"), message.get("metric"), message.get("error"))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._conn.send({"kind": "shutdown"})
        except Exception:
            pass
        finally:
            try:
                self._conn.close()
            except Exception:
                pass
            if self._process.is_alive():
                self._process.join(timeout=1.0)


class ProcessParallelEvaluator:
    def __init__(
        self,
        init_data_per_replica: Dict[int, WorkerInitData],
        timeout: float = EVALUATION_TIMEOUT_SECS,
    ) -> None:
        self._timeout = timeout
        self._workers: Dict[int, ProcessReplicaWorker] = {
            replica_id: ProcessReplicaWorker(init_data, timeout=timeout)
            for replica_id, init_data in init_data_per_replica.items()
        }
        self._conn_to_replica: Dict[Connection, int] = {
            worker.connection: replica_id
            for replica_id, worker in self._workers.items()
        }
        atexit.register(self.shutdown)

    def shutdown(self) -> None:
        for worker in self._workers.values():
            worker.close()
        self._workers.clear()
        self._conn_to_replica.clear()

    def evaluate_request(
        self,
        request: Request,
        replica_schedulers: Iterable[Tuple[int, BaseReplicaScheduler]],
        snapshot_fn,
    ) -> List[Tuple[int, float]]:
        pending: Dict[int, ProcessReplicaWorker] = {}
        results: Dict[int, float] = {}
        for replica_id, replica_scheduler in replica_schedulers:
            worker = self._workers.get(replica_id)
            if worker is None:
                continue
            snapshot = snapshot_fn(replica_scheduler)
            worker.evaluate(request, snapshot)
            pending[replica_id] = worker

        deadline = time.monotonic() + self._timeout
        while pending:
            remaining = max(0.0, deadline - time.monotonic())
            ready_conns = wait(
                [worker.connection for worker in pending.values()],
                timeout=remaining if remaining else 0.0,
            )
            if not ready_conns:
                # Timed out waiting for workers
                for replica_id in list(pending.keys()):
                    results[replica_id] = float("inf")
                    pending.pop(replica_id, None)
                break

            for conn in ready_conns:
                replica_id = self._conn_to_replica.get(conn)
                if replica_id is None:
                    continue
                worker = pending.pop(replica_id, None)
                if worker is None:
                    continue
                status, metric, error = worker.recv()
                if status == "ok" and metric is not None:
                    results[replica_id] = metric
                elif status == "timeout":
                    results[replica_id] = float("inf")
                else:
                    results[replica_id] = float("inf")
        ordered_metrics: List[Tuple[int, float]] = []
        for replica_id, _ in replica_schedulers:
            if replica_id in results:
                ordered_metrics.append((replica_id, results[replica_id]))
        return ordered_metrics


class _ReplicaEvaluatorWorker:
    def __init__(self, conn: Connection, init_data: WorkerInitData) -> None:
        self._conn = conn
        self._init_data = init_data
        self._setup()

    def _setup(self) -> None:
        data = self._init_data
        replica = Replica(data.replica_config, data.request_generator_config)
        replica._id = data.replica_id

        execution_time_predictor = ExecutionTimePredictorRegistry.get(
            data.execution_time_predictor_config.get_type(),
            predictor_config=data.execution_time_predictor_config,
            replica_config=data.replica_config,
            replica_scheduler_config=data.replica_scheduler_config,
            metrics_config=data.metrics_config,
        )

        replica_scheduler = ReplicaSchedulerRegistry.get(
            data.replica_scheduler_config.get_type(),
            replica_config=data.replica_config,
            replica_scheduler_config=data.replica_scheduler_config,
            request_generator_config=data.request_generator_config,
            replica=replica,
            num_stages=replica.num_pipeline_stages,
            execution_time_predictor=execution_time_predictor,
        )

        predictor = RequestTimelinePredictorRegistry.get(data.predictor_type)

        configure = getattr(predictor, "configure", None)
        if callable(configure):
            configure(
                noise_fraction=data.noise_fraction,
                noise_distribution=data.noise_distribution,
                random_seed=data.predictor_random_seed,
            )

        if data.deterministic_noise:
            noise_provider = DeterministicNoiseProvider(
                noise_fraction=data.noise_fraction,
                seed=data.simulation_seed,
            )
            set_noise_provider = getattr(predictor, "set_noise_provider", None)
            if callable(set_noise_provider):
                set_noise_provider(noise_provider)

        predictor.attach_execution_time_predictor(execution_time_predictor)

        if data.fast_predict:
            disable_copy = getattr(
                predictor, "disable_copy_of_base_replica_scheduler", None
            )
            if callable(disable_copy):
                disable_copy()

        enable_parallel_mode = getattr(predictor, "enable_parallel_mode", None)
        if callable(enable_parallel_mode):
            enable_parallel_mode(False)

        if hasattr(predictor, "use_estimated_time"):
            predictor.use_estimated_time = data.use_estimated_time
        if hasattr(predictor, "threshold_batch_size_for_time_estimation"):
            predictor.threshold_batch_size_for_time_estimation = data.threshold_batch_size

        self._target_metric = TargetMetric.from_str(data.target_metric)
        self._replica_scheduler = replica_scheduler
        self._request_timeline_predictor = predictor

    def run(self) -> None:
        while True:
            message = self._conn.recv()
            kind = message.get("kind")
            if kind == "shutdown":
                break
            if kind != "evaluate":
                continue
            request: Request = message["request"]
            snapshot: Dict[str, Any] = message["snapshot"]
            try:
                restore_replica_scheduler_state(self._replica_scheduler, snapshot)
                metric = get_target_metric_value(
                    self._target_metric,
                    self._replica_scheduler,
                    request,
                    self._request_timeline_predictor,
                )
                self._conn.send({"status": "ok", "metric": metric})
            except Exception as exc:
                self._conn.send({"status": "error", "error": repr(exc)})


def _worker_main(conn: Connection, init_data: WorkerInitData) -> None:
    worker = _ReplicaEvaluatorWorker(conn, init_data)
    worker.run()
    conn.close()
