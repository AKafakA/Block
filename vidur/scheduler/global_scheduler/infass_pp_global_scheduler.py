import math
import random
from typing import List, Tuple

from vidur.config import InfassPlusPlusGlobalSchedulerConfig
from vidur.entities import Request
from vidur.scheduler.global_scheduler.base_global_scheduler import BaseGlobalScheduler


class InfassPlusPlusGlobalScheduler(BaseGlobalScheduler):
    """INFaaS++ heuristic: minimize usedMemory / batchSize.

    - usedMemory is the number of allocated blocks on the replica
    - batchSize is the total number of in-flight requests across running batches
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(
            self._config.cluster_config.global_scheduler_config,
            InfassPlusPlusGlobalSchedulerConfig,
        ):
            raise ValueError("Invalid config for INFaaS++ scheduler")

    def schedule(self) -> List[Tuple[int, Request]]:
        self.sort_requests()
        request_mapping: List[Tuple[int, Request]] = []

        while self._request_queue:
            request = self._request_queue.pop(0)
            loads = {
                scheduler.replica_id: self._compute_infass_load(scheduler)
                for scheduler in self._replica_schedulers.values()
            }
            min_load = min(loads.values())
            best_replicas = [rid for rid, load in loads.items() if load == min_load]
            replica_id = random.choice(best_replicas)
            request_mapping.append((replica_id, request))

        return request_mapping

    def _compute_infass_load(self, replica_scheduler) -> float:
        running_requests = sum(
            len(batch.request_ids) for batch in replica_scheduler.running_batches
        )
        batch_size = max(running_requests, 1)
        used_blocks = getattr(replica_scheduler, "_num_allocated_blocks", 0)
        return used_blocks / batch_size
