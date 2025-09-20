from __future__ import annotations

import copy
from typing import Any, Dict

from vidur.scheduler.replica_scheduler.base_replica_scheduler import (
    BaseReplicaScheduler,
)


def _collect_requests(replica_scheduler: BaseReplicaScheduler) -> Dict[int, Any]:
    seen: Dict[int, Any] = {}
    for request in list(replica_scheduler._request_queue) + list(
        replica_scheduler._preempted_requests
    ):
        seen[request.id] = request
    for batch in replica_scheduler.running_batches:
        for request in batch.requests:
            seen[request.id] = request
    return seen


def _snapshot_requests_state(replica_scheduler: BaseReplicaScheduler) -> Dict[int, Dict[str, Any]]:
    requests = _collect_requests(replica_scheduler)
    snapshot: Dict[int, Dict[str, Any]] = {}
    for request_id, request in requests.items():
        snapshot[request_id] = {
            "_num_prefill_tokens": request._num_prefill_tokens,
            "_num_decode_tokens": request._num_decode_tokens,
            "_num_processed_tokens": request._num_processed_tokens,
            "_scheduled_at": request._scheduled_at,
            "_execution_time": request._execution_time,
            "_model_execution_time": request._model_execution_time,
            "_scheduling_delay": request._scheduling_delay,
            "_preempted_time": request._preempted_time,
            "_completed_at": request._completed_at,
            "_prefill_completed_at": request._prefill_completed_at,
            "_latest_stage_scheduled_at": request._latest_stage_scheduled_at,
            "_latest_stage_completed_at": request._latest_stage_completed_at,
            "_latest_iteration_scheduled_at": request._latest_iteration_scheduled_at,
            "_latest_iteration_completed_at": request._latest_iteration_completed_at,
            "_latest_iteration_scheduling_delay": request._latest_iteration_scheduling_delay,
            "_scheduled": request._scheduled,
            "_preempted": request._preempted,
            "_completed": request._completed,
            "_is_prefill_complete": request._is_prefill_complete,
            "_num_restarts": request._num_restarts,
        }
    return snapshot


def _restore_requests_state(replica_scheduler: BaseReplicaScheduler, snapshot: Dict[int, Dict[str, Any]]) -> None:
    requests = _collect_requests(replica_scheduler)
    for request_id, state in snapshot.items():
        request = requests.get(request_id)
        if request is None:
            continue
        for attribute, value in state.items():
            setattr(request, attribute, value)


def _snapshot_stage_scheduler_state(stage_scheduler: Any) -> Dict[str, Any]:
    return {
        "current_execution_time": stage_scheduler.current_execution_time,
        "_is_busy": stage_scheduler._is_busy,
        "_batch_queue": list(stage_scheduler._batch_queue),
    }


def _restore_stage_scheduler_state(stage_scheduler: Any, snapshot: Dict[str, Any]) -> None:
    stage_scheduler.current_execution_time = snapshot["current_execution_time"]
    stage_scheduler._is_busy = snapshot["_is_busy"]
    stage_scheduler._batch_queue.clear()
    stage_scheduler._batch_queue.extend(snapshot["_batch_queue"])


def snapshot_replica_scheduler_state(replica_scheduler: BaseReplicaScheduler) -> Dict[str, Any]:
    return {
        "running_batches": list(replica_scheduler.running_batches),
        "_num_running_batches": replica_scheduler._num_running_batches,
        "_request_queue": list(replica_scheduler._request_queue),
        "_preempted_requests": list(replica_scheduler._preempted_requests),
        "_allocation_map": replica_scheduler._allocation_map.copy(),
        "_num_allocated_blocks": replica_scheduler._num_allocated_blocks,
        "_requests_state": _snapshot_requests_state(replica_scheduler),
        "_replica_stage_schedulers": {
            stage_id: _snapshot_stage_scheduler_state(stage_scheduler)
            for stage_id, stage_scheduler in replica_scheduler._replica_stage_schedulers.items()
        },
    }


def restore_replica_scheduler_state(
    replica_scheduler: BaseReplicaScheduler, state: Dict[str, Any]
) -> None:
    replica_scheduler.running_batches.clear()
    replica_scheduler.running_batches.extend(state["running_batches"])
    replica_scheduler._num_running_batches = state["_num_running_batches"]

    replica_scheduler._request_queue.clear()
    replica_scheduler._request_queue.extend(state["_request_queue"])

    replica_scheduler._preempted_requests.clear()
    replica_scheduler._preempted_requests.extend(state["_preempted_requests"])

    replica_scheduler._allocation_map = state["_allocation_map"].copy()
    replica_scheduler._num_allocated_blocks = state["_num_allocated_blocks"]

    for stage_id, stage_snapshot in state["_replica_stage_schedulers"].items():
        _restore_stage_scheduler_state(
            replica_scheduler._replica_stage_schedulers[stage_id], stage_snapshot
        )

    _restore_requests_state(replica_scheduler, state["_requests_state"])
