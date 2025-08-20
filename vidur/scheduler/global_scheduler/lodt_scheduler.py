from typing import List, Tuple

from vidur.entities import Request
from vidur.scheduler.global_scheduler.base_global_scheduler import BaseGlobalScheduler


def caculate_load_scores(num_requests, num_free_blocks):
    return (num_requests / num_free_blocks) * (-1)


class LODTScheduler(BaseGlobalScheduler):
    """
    Length-aware global scheduler to schedule requests based on the number of unprocessed tokens
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def schedule(self) -> List[Tuple[int, Request]]:
        self.sort_requests()

        request_mapping = []
        # keep a map of replica_id -> replica_scheduler
        # this is used to find the replica with the least outstanding requests
        pending_requests_map = {
            replica_scheduler.replica_id: caculate_load_scores(replica_scheduler.num_pending_requests,
                                                               replica_scheduler.num_free_blocks)
            for replica_scheduler in self._replica_schedulers.values()
        }

        while self._request_queue:
            request = self._request_queue.pop(0)
            replica_id = min(pending_requests_map.items(), key=lambda x: x[1])[0]
            pending_requests_map[replica_id] += request.total_tokens
            request_mapping.append((replica_id, request))

        return request_mapping
