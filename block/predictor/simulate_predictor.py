import random
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from math import ceil
import itertools
import logging
import aiohttp
import asyncio

import orjson

from vidur.config import DummyRequestGeneratorConfig, MetricsConfig, \
    SimulationRequestTimelinePredictorConfig
from vidur.entities import Replica, Request
from vidur.execution_time_predictor import ExecutionTimePredictorRegistry
from block.predictor.predictor import Predictor
from vidur.scheduler.replica_scheduler.replica_scheduler_registry import ReplicaSchedulerRegistry
from vidur.types import ReplicaSchedulerType
from vidur.types.optimal_global_scheduler_target_metric import TargetMetric

logging.basicConfig(level=logging.INFO,
                    filemode='a+',
                    filename='predictor_benchmark.log')

NUM_FIELD_PER_REQUEST = 7


def get_predicted_metrics(replica_scheduler, response_data, predicted_target_request,
                          target_metric: TargetMetric, request_timeline_predictor):
    from vidur.request_timeline_predictor.base_request_timeline_predictor import get_target_metric_value
    metric = get_target_metric_value(target_metric, replica_scheduler, predicted_target_request,
                                     request_timeline_predictor)

    return metric


def convert_list_to_request_infos(raw: list) -> list:
    """
    Convert the raw list of request information into a dictionary with keys
    "request_id", "arrival_time", "seq_prompts_length", "seq_total_output_length",
    "seq_computed_length", "is_prefill", and "seq_expected_decoded_length".
    """
    request_infos = []
    for i in range(0, len(raw), NUM_FIELD_PER_REQUEST):
        request_infos.append({
            "request_id": raw[i],
            "arrival_time": raw[i + 1],
            "seq_total_output_length": raw[i + 2],
            "seq_prompts_length": raw[i + 3],
            "seq_computed_length": raw[i + 4],
            "is_prefill": raw[i + 5] == 1,
            # when use the estimated length, when the length underestimate the length, use the real generated length
            # when the length overestimate the length, use the estimated length + 10 indicating the case,
            # response will end soon
            "seq_expected_decoded_length": raw[i + 6] if raw[i + 6] + raw[i + 3] > raw[i + 2] else
            (raw[i + 2] - raw[i + 3]) + 10
        })
        if not (raw[i + 5] == 1 or raw[i + 5] == 0):
            raise ValueError(f"Invalid value for is_prefill: {raw[i + 5]}")
    return request_infos


class SimulatePredictor(Predictor):
    """A raw implementation of the predictor class
      that extends the Simulator class which is used to predict the completion time of the request.
      It will always create a mirror of replica scheduler and make the prediction based on the target metric.
    """

    def __init__(self, config, port, use_multiprocessing=False):
        super().__init__(config, port)
        self._logger = logging.getLogger(__name__)
        self._config = config
        self._generate_config = DummyRequestGeneratorConfig()
        self._metrics_config = MetricsConfig(
            write_metrics=False,
            create_output_dir=False
        )
        self._simulation_config = SimulationRequestTimelinePredictorConfig()
        self._replica = Replica(config.replica_config, self._generate_config)
        self._execution_time_predictor = ExecutionTimePredictorRegistry.get(
            config.execution_time_predictor_config.get_type(),
            predictor_config=config.execution_time_predictor_config,
            replica_config=config.replica_config,
            replica_scheduler_config=config.replica_scheduler_config,
            metrics_config=self._metrics_config,
        )
        self._enable_chunked_prefill = config.replica_scheduler_config.get_type() == ReplicaSchedulerType.SARATHI
        self._request_queue = []
        if config.target_metric.upper() in TargetMetric.__members__.keys():
            self._target_metric = TargetMetric.from_str(config.target_metric)
            self._need_to_predict = True
        else:
            self._need_to_predict = False
        from vidur.request_timeline_predictor.simulate_request_timeline_predictor import \
            SimulateRequestTimelinePredictor
        self._request_timeline_predictor = SimulateRequestTimelinePredictor()
        self._request_timeline_predictor.attach_execution_time_predictor(self._execution_time_predictor)
        self._request_timeline_predictor.disable_copy_of_base_replica_scheduler()
        self._request_timeline_predictor.use_estimated_time = config.enable_batch_time_estimation
        self._request_timeline_predictor.threshold_batch_size_for_time_estimation = \
            config.threshold_batch_size_for_time_estimation
        self._port = port
        self._start_time = time.time()
        if self._need_to_predict:
            self._backend_url = f"http://localhost:{self._port}/schedule_trace"
        else:
            self._backend_url = f"http://localhost:{self._port}/simple_schedule_trace"
        self._query_timeout = aiohttp.ClientTimeout(total=config.prediction_timeout * 0.9)
        if use_multiprocessing:
            self._executor = ThreadPoolExecutor(max_workers=6)
        else:
            self._executor = None

    async def predict(self, target_request: Request):
        response_data = await self.get_response_data(target_request.id)
        metrics = {}
        if len(response_data) == 0:
            current_gpu_blocks = -1
            current_total_requests = -1
            current_num_preempted = -1
            current_running_request = -1
        else:
            current_gpu_blocks = response_data["free_gpu_blocks"]
            current_num_preempted = response_data["num_preempted"]
            if "total_requests_count" in response_data and "current_running_requests_count" in response_data:
                current_total_requests = response_data["total_requests_count"]
                current_running_request = response_data["current_running_requests_count"]
            else:
                current_total_requests = (
                        (len(response_data["waiting"]) + len(response_data["running"]) + len(response_data["swap"]))
                        // NUM_FIELD_PER_REQUEST)
                current_running_request = len(response_data["running"]) // NUM_FIELD_PER_REQUEST
        if self._need_to_predict:
            if len(response_data) > 0:
                replica_scheduler = self.get_replica_scheduler_with_backend_response(response_data)
                loop = asyncio.get_event_loop()
                target_metric = await loop.run_in_executor(
                    self._executor,
                    get_predicted_metrics,
                    replica_scheduler,
                    response_data,
                    target_request,
                    self._target_metric,
                    self._request_timeline_predictor
                )
            else:
                target_metric = -1
        elif self._config.target_metric == "min_current_gpu_blocks":
            target_metric = current_gpu_blocks
        elif self._config.target_metric == "min_current_requests":
            target_metric = current_total_requests
        elif self._config.target_metric == "min_infass_load":
            target_metric = (current_gpu_blocks / max(current_running_request, 1))*(-1)
        elif self._config.target_metric == "min_lunmnix_load":
            adjusted_free_gpu_blocks = response_data.get("adjusted_free_gpu_blocks", current_gpu_blocks)
            target_metric = (adjusted_free_gpu_blocks / max(current_running_request, 1))*(-1)
        else:
            target_metric = random.randint(0, 100)
        metrics["target_metric"] = target_metric
        metrics["gpu_blocks"] = current_gpu_blocks
        metrics["num_requests"] = current_total_requests
        metrics["num_preempted"] = current_num_preempted
        return metrics

    def __generate_requests_from_backend(self, request_info: dict, source: str) -> Request:
        request_id = int(request_info["request_id"])
        arrival_time = request_info["arrival_time"]
        context_length = request_info["seq_prompts_length"]
        total_length = request_info["seq_total_output_length"]
        prefilled_length = request_info["seq_computed_length"]
        is_prefill = request_info["is_prefill"]
        total_decode_length = request_info["seq_expected_decoded_length"]
        if self._enable_chunked_prefill:
            if is_prefill:
                # total length = sequence.prompts_length + sequence.decoded_length
                # with chunked prefilled, running consisted ongoing prefilling request, decoding request w/o preemption
                # if it is prefilling and not preempted, the total length = sequence.prompts_length
                # else if preempted, the total length = prompts_length + decoded_length before preemption
                # and the decoded tokens can also be recomputed concurrently as prompted request
                context_length = total_length
                processed_length = prefilled_length
                is_prefill_complete = False
            else:
                # if it is decoded, context length = sequence.prompts_length and processed length = decoded_length
                is_prefill_complete = True
                processed_length = total_length
            request = Request(arrival_time, context_length, total_decode_length, processed_length)
            if source == 'running':
                request._is_prefill_complete = is_prefill_complete
        else:
            # if not chunked prefill, the ongoing request is always prefilled and preempted will be appended to waiting
            # queue
            request = Request(arrival_time, context_length, total_decode_length, total_length)
            if source == 'running':
                request._is_prefill_complete = True
        request.source = source
        request.set_id(request_id)
        return request

    async def get_response_data(self, request_id):
        start_time = time.time()
        async with aiohttp.ClientSession(timeout=self._query_timeout) as session:
            try:
                async with session.get(self._backend_url) as response:
                    response_data = orjson.loads(await response.read())
                    return response_data
            except asyncio.TimeoutError as e:
                connect_time = (time.time() - start_time) * 1000
                print(f"timed out: {e} in time, use default results {connect_time}")
                return {}

    def get_replica_scheduler_with_backend_response(self, response):
        replica_scheduler = ReplicaSchedulerRegistry.get(
            self._config.replica_scheduler_config.get_type(),
            replica_config=self._config.replica_config,
            replica_scheduler_config=self._config.replica_scheduler_config,
            request_generator_config=self._generate_config,
            replica=self._replica,
            num_stages=self._replica.num_pipeline_stages,
            execution_time_predictor=self._execution_time_predictor,
        )
        waiting_request_length = convert_list_to_request_infos(response["waiting"])
        running_request_length = convert_list_to_request_infos(response["running"])
        swap_request_length = convert_list_to_request_infos(response["swap"])

        for requests_info in running_request_length:
            request = self.__generate_requests_from_backend(requests_info, 'running')
            if request.num_processed_tokens == request.total_tokens:
                continue
            if self._enable_chunked_prefill:
                allocated_tokens = max(request.num_processed_tokens, request.num_prefill_tokens)
            else:
                allocated_tokens = request.num_processed_tokens
            num_required_blocks = ceil(
                allocated_tokens / self._config.replica_scheduler_config.block_size
            )
            replica_scheduler.allocate(request.id, num_required_blocks)
            request.loading_tokens = request.num_processed_tokens
            replica_scheduler.add_preempted_request(request)
        preempted_request = []
        waiting_request = []
        for requests_info in itertools.chain(waiting_request_length, swap_request_length):
            request = self.__generate_requests_from_backend(requests_info, 'waiting')
            if request.num_processed_tokens == request.total_tokens:
                continue
            if request.num_processed_tokens > 0:
                request.restart()
                preempted_request.append(request)
            else:
                waiting_request.append(request)
        for request in itertools.chain(preempted_request, waiting_request):
            replica_scheduler.add_request(request)
        return replica_scheduler
