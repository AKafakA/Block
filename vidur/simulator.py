import atexit
import heapq
import json
import time
from typing import List

from math import ceil
from vidur.config import SimulationConfig
from vidur.entities import Cluster
from vidur.events import BaseEvent, RequestArrivalEvent
from vidur.events.global_schedule_event import GlobalScheduleEvent
from vidur.logger import init_logger
from vidur.metrics import MetricsStore
from vidur.request_generator import RequestGeneratorRegistry
from vidur.scheduler import BaseGlobalScheduler, GlobalSchedulerRegistry
from pyinstrument import Profiler

logger = init_logger(__name__)

DEFAULT_PROGRESS_LOG_INTERVAL = 500
DEFAULT_PROGRESS_LOG_TIME_SECS = 300  # 5 minutes


class Simulator:
    def __init__(self, config: SimulationConfig) -> None:
        self._config: SimulationConfig = config

        self._time = 0
        self._terminate = False
        self._time_limit = self._config.time_limit
        if not self._time_limit:
            self._time_limit = float("inf")

        self._event_queue = []

        self._event_trace = []
        self._event_chrome_trace = []

        self._cluster = Cluster(
            self._config.cluster_config,
            self._config.metrics_config,
            self._config.request_generator_config,
        )
        self._metric_store = MetricsStore(self._config)
        self._request_generator = RequestGeneratorRegistry.get(
            self._config.request_generator_config.get_type(),
            self._config.request_generator_config,
        )
        self._scheduler = GlobalSchedulerRegistry.get(
            self._config.cluster_config.global_scheduler_config.get_type(),
            self._config,
            self._cluster.replicas,
        )

        self._total_requests = 0
        self._progress_log_interval = DEFAULT_PROGRESS_LOG_INTERVAL
        self._next_progress_log = DEFAULT_PROGRESS_LOG_INTERVAL
        self._last_progress_log = 0
        self._progress_log_time_secs = DEFAULT_PROGRESS_LOG_TIME_SECS
        self._last_progress_time = 0.0

        self._init_event_queue()
        atexit.register(self._write_output)

    @property
    def scheduler(self) -> BaseGlobalScheduler:
        return self._scheduler

    @property
    def metric_store(self) -> MetricsStore:
        return self._metric_store

    def run(self) -> None:
        logger.info(
            f"Starting simulation with cluster: {self._cluster} and {len(self._event_queue)} requests"
        )

        profiler = Profiler()
        profiler.start()
        # code you want to profile
        start_time = time.time()
        self._last_progress_time = start_time

        while self._event_queue and not self._terminate:
            _, event = heapq.heappop(self._event_queue)
            self._set_time(event._time)
            new_events = event.handle_event(self._scheduler, self._metric_store)
            for event in new_events:
                if isinstance(event, GlobalScheduleEvent):
                    scheduled = self._scheduler.num_scheduled_requests
                    if scheduled and scheduled >= self._next_progress_log:
                        if self._total_requests:
                            pct = (scheduled / self._total_requests) * 100
                            logger.info(
                                "Processed %d/%d requests (%.1f%%) after %.1fs",
                                scheduled,
                                self._total_requests,
                                pct,
                                time.time() - start_time,
                            )
                        else:
                            logger.info(
                                "Processed %d requests after %.1fs",
                                scheduled,
                                time.time() - start_time,
                            )
                        self._last_progress_log = scheduled
                        while scheduled >= self._next_progress_log:
                            self._next_progress_log += self._progress_log_interval
            self._add_events(new_events)

            # Time-based progress logging every N seconds
            now = time.time()
            if now - self._last_progress_time >= self._progress_log_time_secs:
                scheduled = self._scheduler.num_scheduled_requests
                if self._total_requests:
                    pct = (scheduled / self._total_requests) * 100 if self._total_requests else 0.0
                    logger.info(
                        "Processed %d/%d requests (%.1f%%) after %.1fs",
                        scheduled,
                        self._total_requests,
                        pct,
                        now - start_time,
                    )
                else:
                    logger.info(
                        "Processed %d requests after %.1fs",
                        scheduled,
                        now - start_time,
                    )
                self._last_progress_time = now

            if self._config.metrics_config.write_json_trace:
                self._event_trace.append(event.to_dict())

            if self._config.metrics_config.enable_chrome_trace:
                chrome_trace = event.to_chrome_trace()
                if chrome_trace:
                    self._event_chrome_trace.append(chrome_trace)

        assert self._scheduler.is_empty() or self._terminate
        end_time = time.time()
        final_count = self._scheduler.num_scheduled_requests
        if final_count and final_count != self._last_progress_log:
            if self._total_requests:
                pct = (final_count / self._total_requests) * 100
                logger.info(
                    "Processed %d/%d requests (%.1f%%) after %.1fs",
                    final_count,
                    self._total_requests,
                    pct,
                    end_time - start_time,
                )
            else:
                logger.info(
                    "Processed %d requests after %.1fs",
                    final_count,
                    end_time - start_time,
                )
        logger.info(f"Simulation took: {end_time - start_time}s")

        logger.info(f"Simulation ended at: {self._time}s")
        profiler.stop()
        profiler.print(file=open("profiler.log", "w"))

    def _write_output(self) -> None:
        logger.info("Writing output")

        self._metric_store.plot()
        logger.info("Metrics written")

        if self._config.metrics_config.write_json_trace:
            self._write_event_trace()
            logger.info("Json event trace written")

        if self._config.metrics_config.enable_chrome_trace:
            self._write_chrome_trace()
            logger.info("Chrome event trace written")

    def _add_event(self, event: BaseEvent) -> None:
        heapq.heappush(self._event_queue, (event._priority_number, event))

    def _add_events(self, events: List[BaseEvent]) -> None:
        for event in events:
            self._add_event(event)

    def _init_event_queue(self) -> None:
        requests = self._request_generator.generate()

        self._total_requests = len(requests)
        if self._total_requests:
            # Use dynamic progress step: every 10% for small runs; 1000 for large runs
            if self._total_requests >= 1000:
                self._progress_log_interval = 1000
            else:
                self._progress_log_interval = max(10, ceil(self._total_requests / 10))
            self._next_progress_log = min(self._progress_log_interval, self._total_requests)
        else:
            self._next_progress_log = self._progress_log_interval

        for request in requests:
            self._add_event(RequestArrivalEvent(request.arrived_at, request))

    def _set_time(self, time: float) -> None:
        self._time = time
        if self._time > self._time_limit:
            logger.info(
                f"Time limit reached: {self._time_limit}s terminating the simulation."
            )
            self._terminate = True

    def _write_event_trace(self) -> None:
        trace_file = f"{self._config.metrics_config.output_dir}/event_trace.json"
        with open(trace_file, "w") as f:
            json.dump(self._event_trace, f)

    def _write_chrome_trace(self) -> None:
        trace_file = f"{self._config.metrics_config.output_dir}/chrome_trace.json"

        chrome_trace = {"traceEvents": self._event_chrome_trace}

        with open(trace_file, "w") as f:
            json.dump(chrome_trace, f)
