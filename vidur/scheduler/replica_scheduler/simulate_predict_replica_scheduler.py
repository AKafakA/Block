import copy
import heapq
import logging
from collections import OrderedDict
from typing import Dict, Tuple

from vidur.entities import Batch, Request
from vidur.execution_time_predictor import BaseExecutionTimePredictor
from vidur.scheduler.replica_scheduler.base_replica_scheduler import BaseReplicaScheduler
from vidur.scheduler.utils.replica_state import (
    restore_replica_scheduler_state,
    snapshot_replica_scheduler_state,
)


class SimulatePredictReplicaScheduler:
    """
    Simulate the replica scheduler and predict the scheduling delay, request makespan, average batch size and
    average decoding latency
    Rely on actual replica scheduler to simulate the batch scheduling
    and use the execution time predictor to predict the execution time of each batch
    """

    def __init__(self, replica_scheduler: BaseReplicaScheduler,
                 request: Request,
                 execution_time_predictor: BaseExecutionTimePredictor,
                 use_estimated_execution_time=True,
                 copy_replica_scheduler=True,
                 start_time=0,
                 threshold_batch_size_for_time_estimation=36,
                 running_until_target_finished=True,
                 batch_execution_time_caching_map=None,
                 max_batch_execution_time_cache_per_request: int = 32) -> None:
        self._replica_id = replica_scheduler.replica_id
        self._raw_replica_scheduler = replica_scheduler
        self._copy_needed = copy_replica_scheduler
        if copy_replica_scheduler:
            self._replica_scheduler = copy.deepcopy(replica_scheduler)
        else:
            self._replica_scheduler = replica_scheduler

        self._target_request = copy.deepcopy(request)
        self._target_request._num_decode_tokens = request.num_predicted_decode_tokens
        self._execution_time_predictor = execution_time_predictor
        self._all_request_batch_info = []
        self._scheduled_batch_heap = []
        self._scheduled_batch_id = 0
        self._estimate_execution_time = use_estimated_execution_time
        self._default_execution_time = 0.02
        self._threshold_batch_size_for_time_estimation = threshold_batch_size_for_time_estimation
        self._start_time = start_time
        self._request_ids = set()
        self._running_until_target_finished = running_until_target_finished
        self._batch_execution_time_caching_map = batch_execution_time_caching_map
        self._execution_time_cache: Dict[Tuple[int, Tuple[int, ...], Tuple[int, ...]], float] = {}
        self._max_batch_execution_time_cache_per_request = max(0, max_batch_execution_time_cache_per_request)
        self._logger = logging.getLogger(__name__)

    def simulate(self):
        assert self._target_request is not None

        self._all_request_batch_info.clear()
        self._scheduled_batch_heap.clear()
        self._request_ids.clear()
        self._scheduled_batch_id = 0

        snapshot = None
        if not self._copy_needed:
            snapshot = snapshot_replica_scheduler_state(self._replica_scheduler)

        try:
            self._run_simulation()
        finally:
            if snapshot is not None:
                restore_replica_scheduler_state(self._replica_scheduler, snapshot)
            # Clear execution-time cache between simulations to avoid unbounded growth
            self._execution_time_cache.clear()

    def _run_simulation(self) -> None:
        replica_scheduler = self._replica_scheduler

        replica_scheduler.add_request(self._target_request)
        existing_batches = replica_scheduler.running_batches
        replica_scheduler.running_batches = []
        push_batch = self.__push_batch
        for batch in existing_batches:
            push_batch(copy.copy(batch), self._start_time)
        new_batches = replica_scheduler.on_schedule()
        # so the initialized batch == the number of stages then only be pushed after pop so that the batch number
        # is limited by the number of stages
        for new_batch in new_batches:
            push_batch(new_batch, self._start_time)
        add_request_id = self._request_ids.add
        all_info_append = self._all_request_batch_info.append
        target_request = self._target_request
        while self._scheduled_batch_heap:
            (
                batch_id,
                batch_execution_time,
                schedule_time,
                completed_at,
                batch,
                num_allocated_blocks,
            ) = self.__pop_batch()
            for request_id in batch.request_ids:
                add_request_id(request_id)
            all_info_append(
                {
                    "batch_id": batch_id,
                    "batch_execution_time": batch_execution_time,
                    "schedule_time": schedule_time,
                    "batch_size": batch.size,
                    "num_allocated_blocks": num_allocated_blocks,
                    "request_ids": batch.request_ids,
                    "completed_time": completed_at,
                    "target_request_prefilled": target_request.is_prefill_complete,
                }
            )
            if self._running_until_target_finished and self._target_request.completed:
                break

    def __push_batch(self, batch: Batch, schedule_time):
        batch_execution_time = []
        replica_scheduler = self._replica_scheduler
        stage_schedulers = replica_scheduler.replica_stage_schedulers
        get_time = self.__get_execution_time
        for stage_id in stage_schedulers:
            execution_time = get_time(batch, stage_id)
            # if the stage is busy, wait for the current batch to complete.
            # TODO: not sure if this will introduce a duplicated time so keep it as comments but be rechecked later
            # if replica_stage_scheduler.is_busy:
            #     replica_stage_scheduler = self._replica_scheduler.get_replica_stage_scheduler(stage_id)
            #     execution_time += replica_stage_scheduler.current_execution_time
            batch_execution_time.append(execution_time)
        batch_id = self._scheduled_batch_id
        self._scheduled_batch_id += 1
        completed_at = sum(batch_execution_time) + schedule_time
        batch_info = (completed_at, schedule_time, batch_id, batch, batch_execution_time)
        batch.on_schedule(schedule_time)
        heapq.heappush(self._scheduled_batch_heap, batch_info)

    def __pop_batch(self):
        (
            completed_at,
            schedule_time,
            batch_id,
            batch,
            batch_execution_time,
        ) = heapq.heappop(self._scheduled_batch_heap)
        batch.on_batch_end(completed_at)
        self._replica_scheduler.on_batch_end(batch)
        new_batches = self._replica_scheduler.on_schedule()
        num_allocated_blocks = self._replica_scheduler.num_allocated_blocks
        push_batch = self.__push_batch
        for new_batch in new_batches:
            push_batch(new_batch, completed_at)
        return (
            batch_id,
            batch_execution_time,
            schedule_time,
            completed_at,
            batch,
            num_allocated_blocks,
        )

    def _optimized_clone(self, rs: BaseReplicaScheduler):
        # Create a new uninitialized instance of the same class
        cls = rs.__class__
        cloned = cls.__new__(cls)

        # Shallow copy immutable configs and identifiers
        cloned._config = rs._config
        cloned._replica_config = rs._replica_config
        cloned._request_generator_config = rs._request_generator_config
        cloned._replica_id = rs._replica_id
        cloned._num_stages = rs._num_stages

        # Scalar state
        cloned._num_allocated_blocks = rs._num_allocated_blocks
        cloned._allocation_map = rs._allocation_map.copy()
        cloned._num_blocks = getattr(rs, "_num_blocks", None)
        cloned._max_batch_size = getattr(rs, "_max_batch_size", None)
        cloned._max_blocks_per_sequence = rs._max_blocks_per_sequence

        if hasattr(rs, "_max_micro_batch_size"):
            cloned._max_micro_batch_size = rs._max_micro_batch_size
        if hasattr(rs, "_watermark_blocks"):
            cloned._watermark_blocks = rs._watermark_blocks

        # Deep-copy queues and running batches (requests are mutable)
        cloned._preempted_requests = copy.deepcopy(rs._preempted_requests)
        cloned._request_queue = copy.deepcopy(rs._request_queue)
        cloned.running_batches = copy.deepcopy(rs.running_batches)
        cloned._num_running_batches = rs._num_running_batches

        # Deep-copy stage schedulers (their __deepcopy__ keeps predictor shallow)
        cloned._replica_stage_schedulers = copy.deepcopy(rs._replica_stage_schedulers)

        return cloned


    def __get_execution_time(self, batch: Batch, stage_id: int):
        threshold = self._threshold_batch_size_for_time_estimation
        if batch.size <= threshold or threshold < 0:
            return self._default_execution_time

        request_key = tuple(batch.request_ids)
        tokens_key = tuple(batch.num_tokens)
        cache_key = (stage_id, request_key, tokens_key)

        cache = self._execution_time_cache
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        execution_time = None
        batch_cache_map = self._batch_execution_time_caching_map
        if batch_cache_map is not None:
            batch_size = batch.size
            first_request_id = batch.request_ids[0]
            last_request_id = batch.request_ids[-1]
            batch_cache = batch_cache_map.get(batch_size)
            if batch_cache is not None:
                first_cache = batch_cache.get(first_request_id)
                if isinstance(first_cache, OrderedDict):
                    execution_time = first_cache.get(last_request_id)
                    if execution_time is not None:
                        first_cache.move_to_end(last_request_id)
                elif isinstance(first_cache, dict):
                    execution_time = first_cache.get(last_request_id)

        if execution_time is None:
            execution_time = self.get_real_execution_time(batch, stage_id)
            if batch_cache_map is not None:
                batch_size = batch.size
                first_request_id = batch.request_ids[0]
                last_request_id = batch.request_ids[-1]
                batch_cache = batch_cache_map.setdefault(batch_size, {})
                first_cache = batch_cache.get(first_request_id)
                if not isinstance(first_cache, OrderedDict):
                    first_cache = (
                        OrderedDict(first_cache.items())
                        if isinstance(first_cache, dict)
                        else OrderedDict()
                    )
                    batch_cache[first_request_id] = first_cache
                first_cache[last_request_id] = execution_time
                first_cache.move_to_end(last_request_id)
                if self._max_batch_execution_time_cache_per_request and len(first_cache) > self._max_batch_execution_time_cache_per_request:
                    first_cache.popitem(last=False)

        cache[cache_key] = execution_time
        return execution_time

    def get_real_execution_time(self, batch: Batch, stage_id: int):
        try:
            return self._execution_time_predictor.get_execution_time(batch, stage_id).total_time
        except Exception as exc:
            batch_size = batch.size
            # find the batch size in the batch execution time caching map closest to the current batch size
            # and use its execution time as a fallback
            if self._batch_execution_time_caching_map is not None and self._batch_execution_time_caching_map:
                if batch_size in self._batch_execution_time_caching_map:
                    closest_batch_size = batch_size
                else:
                    closest_batch_size = min(
                        self._batch_execution_time_caching_map.keys(),
                        key=lambda x: abs(x - batch_size),
                    )
                batch_cache = self._batch_execution_time_caching_map.get(closest_batch_size, {})
                fallback_time = None
                for first_cache in batch_cache.values():
                    if isinstance(first_cache, OrderedDict) and first_cache:
                        last_key = next(reversed(first_cache))
                        fallback_time = first_cache[last_key]
                        break
                    if isinstance(first_cache, dict) and first_cache:
                        fallback_time = next(iter(first_cache.values()))
                        break
                if fallback_time is not None:
                    self._logger.warning(
                        "Execution-time predictor miss (replica=%s, stage=%s, batch_size=%s, key=%s); "
                        "using cached execution time for batch size %s instead",
                        self._replica_id,
                        stage_id,
                        batch_size,
                        exc,
                        closest_batch_size,
                    )
                    return fallback_time
            else:
                self._logger.warning(
                    "Execution-time predictor miss (replica=%s, stage=%s, batch_size=%s, key=%s); "
                    "using default execution time %.3f instead",
                    self._replica_id,
                    stage_id,
                    batch_size,
                    exc,
                    self._default_execution_time,
                )
                return self._default_execution_time

    def get_target_request_batches(self, request_id):
        return [selected_batch for selected_batch in self._all_request_batch_info
                if request_id in selected_batch["request_ids"]]

    @property
    def target_request_scheduled_at(self):
        return self.get_target_request_batches(self._target_request.id)[0]["schedule_time"]

    @property
    def target_request_completed_at(self):
        last_batch = self.get_target_request_batches(self._target_request.id)[-1]
        return last_batch["completed_time"]

    @property
    def target_request_prefilled_at(self):
        target_batches = self.get_target_request_batches(self._target_request.id)
        min_prefilled_time = target_batches[-1]["completed_time"]
        for batch in target_batches:
            if batch["target_request_prefilled"]:
                min_prefilled_time = min(min_prefilled_time, batch["completed_time"])
        return min_prefilled_time

    @property
    def target_request_end_to_end(self):
        return self.target_request_completed_at - self._start_time

    @property
    def average_execution_time(self):
        return (sum([sum(info["batch_execution_time"]) for info in self._all_request_batch_info]) /
                len(self._all_request_batch_info))

    @property
    def average_latency(self):
        if not self._request_ids:
            # No batches scheduled in the what-if simulation; treat as
            # infinitely bad latency so this replica is not selected.
            return float("inf")
        execution_time = []
        for request_id in self._request_ids:
            batches = self.get_target_request_batches(request_id)
            if not batches:
                # Should not happen, but guard for safety
                continue
            execution_time.append(batches[-1]["completed_time"] - batches[0]["schedule_time"])
        if not execution_time:
            return float("inf")
        return sum(execution_time) / len(execution_time)

    @property
    def average_stage_time(self):
        stage_times = []
        for info in self._all_request_batch_info:
            stage_times.extend(info["batch_execution_time"])
        return sum(stage_times) / len(stage_times)

    @property
    def average_batch_size(self):
        return (sum([info["batch_size"] for info in self._all_request_batch_info]) /
                len(self._all_request_batch_info))

    @property
    def max_batch_size(self):
        return max([info["batch_size"] for info in self._all_request_batch_info])

    @property
    def avg_block_size(self):
        return (sum([info["num_allocated_blocks"] for info in self._all_request_batch_info]) /
                len(self._all_request_batch_info))
