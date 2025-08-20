from typing import List, Tuple

from vidur.entities import Request
from vidur.scheduler.global_scheduler.base_global_scheduler import BaseGlobalScheduler


class MinMemoryScheduler(BaseGlobalScheduler):
    """
    global scheduler select replica with minimum memory usage
    """

    def schedule(self) -> List[Tuple[int, Request]]:
        self.sort_requests()

        request_mapping = []
        memory_map = {
            replica_scheduler.replica_id: replica_scheduler.memory_usage_percent
            for replica_scheduler in self._replica_schedulers.values()
        }

        # using a very simple implementation here, to keep wiring simple
        while self._request_queue:
            request = self._request_queue.pop(0)
            replica_id = min(memory_map.items(), key=lambda x: x[1])[0]
            request_mapping.append((replica_id, request))

        return request_mapping
