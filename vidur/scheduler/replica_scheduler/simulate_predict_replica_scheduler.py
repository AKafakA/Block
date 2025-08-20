import copy
from vidur.entities import Request, Batch
from vidur.execution_time_predictor import BaseExecutionTimePredictor
from vidur.scheduler.replica_scheduler.base_replica_scheduler import BaseReplicaScheduler
import heapq


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
                 batch_execution_time_caching_map=None) -> None:
        self._replica_id = replica_scheduler.replica_id
        self._raw_replica_scheduler = replica_scheduler
        if copy_replica_scheduler:
            self._replica_scheduler = copy.deepcopy(replica_scheduler)
            self._target_request = copy.deepcopy(request)
            self._target_request._num_decode_tokens = request.num_predicted_decode_tokens
        else:
            self._replica_scheduler = replica_scheduler
            self._target_request = request
            self._target_request._num_decode_tokens = request.num_predicted_decode_tokens
        self._copy_needed = copy_replica_scheduler
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

    def simulate(self):
        assert self._target_request is not None
        self._replica_scheduler.add_request(self._target_request)
        existing_batches = self._replica_scheduler.running_batches
        self._replica_scheduler.running_batches = []
        for batch in existing_batches:
            self.__push_batch(copy.copy(batch), self._start_time)
        new_batches = self._replica_scheduler.on_schedule()
        # so the initialized batch == the number of stages then only be pushed after pop so that the batch number
        # is limited by the number of stages
        for new_batch in new_batches:
            self.__push_batch(new_batch, self._start_time)
        while self._scheduled_batch_heap:
            (batch_id, batch_execution_time, schedule_time, batch, num_allocated_blocks) = self.__pop_batch()
            for request_id in batch.request_ids:
                self._request_ids.add(request_id)
            self._all_request_batch_info.append({
                "batch_id": batch_id,
                "batch_execution_time": batch_execution_time,
                "schedule_time": schedule_time,
                "batch_size": batch.size,
                "num_allocated_blocks": num_allocated_blocks,
                "request_ids": batch.request_ids,
                "completed_time": schedule_time + sum(batch_execution_time),
                "target_request_prefilled": self._target_request.is_prefill_complete
            })
            if self._running_until_target_finished and self._target_request.completed:
                break

    def __push_batch(self, batch: Batch, schedule_time):
        batch_execution_time = []
        for stage_id in self._replica_scheduler.replica_stage_schedulers.keys():
            execution_time = self.__get_execution_time(batch, stage_id)
            # if the stage is busy, wait for the current batch to complete.
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
        (completed_at, schedule_time, batch_id, batch, batch_execution_time) = heapq.heappop(self._scheduled_batch_heap)
        batch.on_batch_end(completed_at)
        self._replica_scheduler.on_batch_end(batch)
        new_batches = self._replica_scheduler.on_schedule()
        num_allocated_blocks = self._replica_scheduler.num_allocated_blocks
        for new_batch in new_batches:
            self.__push_batch(new_batch, completed_at)
        return batch_id, batch_execution_time, schedule_time, batch, num_allocated_blocks

    def __get_execution_time(self, batch: Batch, stage_id: int):
        if batch.size > self._threshold_batch_size_for_time_estimation >= 0:
            if self._batch_execution_time_caching_map is not None:
                batch_size = batch.size
                first_request_id = batch.request_ids[0]
                last_request_id = batch.request_ids[-1]
                catch_time = self._batch_execution_time_caching_map.get(
                    batch_size, {}).get(first_request_id, {}).get(last_request_id, None)
                if catch_time is not None:
                    return catch_time
                else:
                    batch_execution_time = self.get_real_execution_time(batch, stage_id)
                    # self._batch_execution_time_catching_map[batch_size][first_request_id][last_request_id] = \
                    #     batch_execution_time
                    if self._batch_execution_time_caching_map.get(batch_size) is None:
                        self._batch_execution_time_caching_map[batch_size] = {}
                    if self._batch_execution_time_caching_map[batch_size].get(first_request_id) is None:
                        self._batch_execution_time_caching_map[batch_size][first_request_id] = {}
                    self._batch_execution_time_caching_map[batch_size][first_request_id][last_request_id] = \
                        batch_execution_time
                    return batch_execution_time
            else:
                return self.get_real_execution_time(batch, stage_id)
        else:
            return self._default_execution_time

    def get_real_execution_time(self, batch: Batch, stage_id: int):
        try:
            return self._execution_time_predictor.get_execution_time(batch, stage_id).total_time
        except Exception as e:
            batch_size = batch.size
            # find the batch size in the batch execution time caching map closest to the current batch size
            # and use its execution time as a fallback
            if self._batch_execution_time_caching_map is not None:
                if batch_size in self._batch_execution_time_caching_map:
                    closest_batch_size = batch_size
                else:
                    closest_batch_size = min(self._batch_execution_time_caching_map.keys(),
                                             key=lambda x: abs(x - batch_size))
                # randomly select one results from caching map
                first_request_id = list(self._batch_execution_time_caching_map[closest_batch_size].keys())[-1]
                last_request_id = list(self._batch_execution_time_caching_map[closest_batch_size][first_request_id].keys())[-1]
                print(f"Error in getting real execution time: {e} with batch size {batch_size} and stage id {stage_id}",
                      f"using cached execution time for batch size {closest_batch_size} instead")
                return self._batch_execution_time_caching_map[closest_batch_size][first_request_id][last_request_id]
            else:
                print(f"Error in getting real execution time: {e} with batch size {batch_size} and stage id {stage_id}",
                      f"using default execution time {self._default_execution_time} instead")
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
        execution_time = []
        for request_id in self._request_ids:
            batches = self.get_target_request_batches(request_id)
            execution_time.append(batches[-1]["completed_time"] - batches[0]["schedule_time"])
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
