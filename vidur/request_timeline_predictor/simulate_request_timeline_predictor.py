from vidur.request_timeline_predictor.base_request_timeline_predictor import BaseRequestTimelinePredictor
from vidur.scheduler.replica_scheduler.simulate_predict_replica_scheduler import SimulatePredictReplicaScheduler


class SimulateRequestTimelinePredictor(BaseRequestTimelinePredictor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_estimated_time = True
        self._copy_base_replica_scheduler = True
        self.threshold_batch_size_for_time_estimation = 36
        self._batch_execution_time_caching_maps = {}
        self._parallel_mode = False

    def disable_copy_of_base_replica_scheduler(self):
        self._copy_base_replica_scheduler = False

    def enable_parallel_mode(self, enabled: bool) -> None:
        self._parallel_mode = enabled

    def _make_simulator(self, replica_scheduler, request, use_estimated_execution_time):
        batch_cache = None if self._parallel_mode else self._batch_execution_time_caching_maps
        return SimulatePredictReplicaScheduler(
            replica_scheduler=replica_scheduler,
            request=request,
            execution_time_predictor=self._execution_time_predictor,
            use_estimated_execution_time=use_estimated_execution_time,
            copy_replica_scheduler=self._copy_base_replica_scheduler,
            threshold_batch_size_for_time_estimation=self.threshold_batch_size_for_time_estimation,
            batch_execution_time_caching_map=batch_cache,
        )

    def predict_avg_block_size(self, replica_scheduler, request):
        simulate_predict_replica_scheduler = self._make_simulator(
            replica_scheduler,
            request,
            self.use_estimated_time,
        )
        simulate_predict_replica_scheduler.simulate()
        return simulate_predict_replica_scheduler.avg_block_size

    def predict_request_scheduling_delay(self, replica_scheduler, request):
        simulate_predict_replica_scheduler = self._make_simulator(
            replica_scheduler,
            request,
            self.use_estimated_time,
        )
        simulate_predict_replica_scheduler.simulate()
        return simulate_predict_replica_scheduler.target_request_scheduled_at

    def predict_request_makespan(self, replica_scheduler, request):
        simulate_predict_replica_scheduler = self._make_simulator(
            replica_scheduler,
            request,
            self.use_estimated_time,
        )
        simulate_predict_replica_scheduler.simulate()
        return simulate_predict_replica_scheduler.target_request_end_to_end

    def predict_average_latency(self, replica_scheduler, request):
        simulate_predict_replica_scheduler = self._make_simulator(
            replica_scheduler,
            request,
            self.use_estimated_time,
        )
        simulate_predict_replica_scheduler.simulate()
        return simulate_predict_replica_scheduler.average_latency

    def predict_average_batch_size(self, replica_scheduler, request):
        simulate_predict_replica_scheduler = self._make_simulator(
            replica_scheduler,
            request,
            False,
        )
        simulate_predict_replica_scheduler.simulate()
        return simulate_predict_replica_scheduler.average_batch_size

    def predict_average_execution_latency(self, replica_scheduler, request):
        simulate_predict_replica_scheduler = self._make_simulator(
            replica_scheduler,
            request,
            self.use_estimated_time,
        )
        simulate_predict_replica_scheduler.simulate()
        return simulate_predict_replica_scheduler.average_execution_time

    def predict_waiting_and_ending_time(self, replica_scheduler, request):
        """
        Predict the waiting and ending time for a given request.
        This method is a placeholder and should be implemented in subclasses.
        """
        simulate_predict_replica_scheduler = self._make_simulator(
            replica_scheduler,
            request,
            self.use_estimated_time,
        )
        simulate_predict_replica_scheduler.simulate()
        return simulate_predict_replica_scheduler.target_request_end_to_end
