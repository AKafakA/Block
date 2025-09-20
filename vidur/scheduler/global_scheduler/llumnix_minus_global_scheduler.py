import math
import random
from typing import List, Tuple

from vidur.config import LlumnixMinusGlobalSchedulerConfig
from vidur.entities import Request
from vidur.scheduler.global_scheduler.base_global_scheduler import BaseGlobalScheduler


class LlumnixMinusGlobalScheduler(BaseGlobalScheduler):
    """Llumnix- heuristic: minimize (usedMemory + prefillMemory) / batchSize.

    - usedMemory: number of allocated blocks on the replica
    - prefillMemory: sum over pending requests of ceil(prefill_tokens / block_size)
    - batchSize: total number of in-flight requests (across running batches)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(
            self._config.cluster_config.global_scheduler_config,
            LlumnixMinusGlobalSchedulerConfig,
        ):
            raise ValueError("Invalid config for LLumnix- scheduler")

    def schedule(self) -> List[Tuple[int, Request]]:
        self.sort_requests()
        request_mapping: List[Tuple[int, Request]] = []

        while self._request_queue:
            request = self._request_queue.pop(0)
            loads = {
                scheduler.replica_id: self._compute_llumnix_load(scheduler)
                for scheduler in self._replica_schedulers.values()
            }
            min_load = min(loads.values())
            best_replicas = [rid for rid, load in loads.items() if load == min_load]
            replica_id = random.choice(best_replicas)
            request_mapping.append((replica_id, request))

        return request_mapping

    def _compute_llumnix_load(self, replica_scheduler) -> float:
        running_requests = sum(
            len(batch.request_ids) for batch in replica_scheduler.running_batches
        )
        batch_size = max(running_requests, 1)

        used_blocks = getattr(replica_scheduler, "_num_allocated_blocks", 0)
        block_size = getattr(replica_scheduler._config, "block_size", 1) or 1

        # Sum required blocks to prefill all pending requests in the queue
        prefill_blocks = 0
        for req in list(replica_scheduler._request_queue):
            prefill_blocks += math.ceil(req.num_prefill_tokens / block_size)

        return (used_blocks + prefill_blocks) / batch_size
