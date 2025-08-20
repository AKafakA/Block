from vidur.entities import Request
from vidur.execution_time_predictor import BaseExecutionTimePredictor
from vidur.scheduler.replica_scheduler.base_replica_scheduler import BaseReplicaScheduler
from vidur.types.optimal_global_scheduler_target_metric import TargetMetric


class BaseRequestTimelinePredictor:
    def __init__(self):
        self._execution_time_predictor = None

    def attach_execution_time_predictor(self, execution_time_predictor: BaseExecutionTimePredictor):
        self._execution_time_predictor = execution_time_predictor

    def predict_avg_block_size(self, replica_scheduler: BaseReplicaScheduler, request: Request):
        raise NotImplementedError("predict method is not implemented")

    def predict_scheduling_delay(self, replica_scheduler: BaseReplicaScheduler, request: Request):
        raise NotImplementedError("predict method is not implemented")

    def predict_request_makespan(self, replica_scheduler: BaseReplicaScheduler, request: Request):
        raise NotImplementedError("predict method is not implemented")

    def predict_average_decoding_latency(self, replica_scheduler, request):
        raise NotImplementedError("predict method is not implemented")

    def predict_average_batch_size(self, replica_scheduler, request):
        raise NotImplementedError("predict method is not implemented")

    def predict_min_batch_size(self, replica_scheduler, request):
        raise NotImplementedError("predict method is not implemented")

    def predict_average_latency(self, replica_scheduler, request):
        raise NotImplementedError("predict method is not implemented")

    def predict_request_scheduling_delay(self, replica_scheduler, request):
        raise NotImplementedError("predict method is not implemented")

    def predict_waiting_and_ending_time(self, replica_scheduler: BaseReplicaScheduler, request: Request):
        """
        Predict the waiting and ending time for a given request.
        This method is a placeholder and should be implemented in subclasses.
        """
        raise NotImplementedError("predict_waiting_and_ending_time method is not implemented")


def get_target_metric_value(target_metric: TargetMetric,
                            replica_scheduler: BaseReplicaScheduler,
                            request: Request,
                            request_timeline_predictor: BaseRequestTimelinePredictor):
    if target_metric == TargetMetric.MIN_LATENCY:
        return request_timeline_predictor.predict_average_latency(replica_scheduler, request)
    elif target_metric == TargetMetric.MIN_NEW_REQUEST_LATENCY:
        return request_timeline_predictor.predict_waiting_and_ending_time(replica_scheduler, request)
    elif target_metric == TargetMetric.MIN_SCHEDULING_DELAY:
        return request_timeline_predictor.predict_request_scheduling_delay(replica_scheduler, request)
    elif target_metric == TargetMetric.MIN_DECODING_DELAY:
        return request_timeline_predictor.predict_average_decoding_latency(replica_scheduler, request)
    elif target_metric == TargetMetric.MAX_AVG_BATCH_SIZE:
        return request_timeline_predictor.predict_average_batch_size(replica_scheduler, request)
    elif target_metric == TargetMetric.MAX_MIN_BATCH_SIZE:
        return request_timeline_predictor.predict_average_batch_size(replica_scheduler, request)
    elif target_metric == TargetMetric.MIN_GPU_BLOCKS:
        return request_timeline_predictor.predict_avg_block_size(replica_scheduler, request)
    else:
        raise ValueError(f"Unknown target metric {target_metric} need to be predicted")
