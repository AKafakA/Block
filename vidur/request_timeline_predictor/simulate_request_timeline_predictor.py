from vidur.request_timeline_predictor.base_request_timeline_predictor import BaseRequestTimelinePredictor
from vidur.scheduler.replica_scheduler.simulate_predict_replica_scheduler import SimulatePredictReplicaScheduler
from vidur.config.config import SimulationRequestTimelinePredictorConfig


class SimulateRequestTimelinePredictor(BaseRequestTimelinePredictor):
    def __init__(self, timeline_predictor_config: SimulationRequestTimelinePredictorConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_estimated_time = True
        self._copy_base_replica_scheduler = True
        self.threshold_batch_size_for_time_estimation = 36
        self._batch_execution_time_caching_maps = {}
        self._base_simulator = None
        self._requests = None
        self._timeline_predictor_config = timeline_predictor_config
        self._enable_fast_cloning = timeline_predictor_config.enable_fast_cloning

    def set_requests(self, requests):
        self._requests = requests
        self._base_simulator = None

    def disable_copy_of_base_replica_scheduler(self):
        self._copy_base_replica_scheduler = False

    def _get_base_simulator(self, replica_scheduler):
        if self._base_simulator is None:
            self._base_simulator = SimulatePredictReplicaScheduler(
                replica_scheduler=replica_scheduler,
                request=self._requests[0],
                execution_time_predictor=self._execution_time_predictor,
                use_estimated_execution_time=self.use_estimated_time,
                copy_replica_scheduler=self._copy_base_replica_scheduler,
                threshold_batch_size_for_time_estimation=self.threshold_batch_size_for_time_estimation,
                batch_execution_time_caching_map=self._batch_execution_time_caching_maps,
                running_until_target_finished=False
            )
            self._base_simulator.simulate()
        return self._base_simulator

    def predict_avg_block_size(self, replica_scheduler, request):
        if self._enable_fast_cloning:
            base_simulator = self._get_base_simulator(replica_scheduler)
            cloned_simulator = base_simulator.clone(request)
            cloned_simulator.simulate(from_cloned=True)
            return cloned_simulator.avg_block_size
        else:
            simulate_predict_replica_scheduler = SimulatePredictReplicaScheduler(
                replica_scheduler=replica_scheduler,
                request=request,
                execution_time_predictor=self._execution_time_predictor,
                use_estimated_execution_time=self.use_estimated_time,
                copy_replica_scheduler=self._copy_base_replica_scheduler,
                threshold_batch_size_for_time_estimation=self.threshold_batch_size_for_time_estimation,
                batch_execution_time_caching_map=self._batch_execution_time_caching_maps,
            )
            simulate_predict_replica_scheduler.simulate()
            return simulate_predict_replica_scheduler.avg_block_size

    def predict_request_scheduling_delay(self, replica_scheduler, request):
        if self._enable_fast_cloning:
            base_simulator = self._get_base_simulator(replica_scheduler)
            cloned_simulator = base_simulator.clone(request)
            cloned_simulator.simulate(from_cloned=True)
            return cloned_simulator.target_request_scheduled_at
        else:
            simulate_predict_replica_scheduler = SimulatePredictReplicaScheduler(
                replica_scheduler=replica_scheduler,
                request=request,
                execution_time_predictor=self._execution_time_predictor,
                use_estimated_execution_time=self.use_estimated_time,
                copy_replica_scheduler=self._copy_base_replica_scheduler,
                threshold_batch_size_for_time_estimation=self.threshold_batch_size_for_time_estimation,
                batch_execution_time_caching_map=self._batch_execution_time_caching_maps,
            )
            simulate_predict_replica_scheduler.simulate()
            return simulate_predict_replica_scheduler.target_request_scheduled_at

    def predict_request_makespan(self, replica_scheduler, request):
        if self._enable_fast_cloning:
            base_simulator = self._get_base_simulator(replica_scheduler)
            cloned_simulator = base_simulator.clone(request)
            cloned_simulator.simulate(from_cloned=True)
            return cloned_simulator.target_request_end_to_end
        else:
            simulate_predict_replica_scheduler = SimulatePredictReplicaScheduler(
                replica_scheduler=replica_scheduler,
                request=request,
                execution_time_predictor=self._execution_time_predictor,
                use_estimated_execution_time=self.use_estimated_time,
                copy_replica_scheduler=self._copy_base_replica_scheduler,
                threshold_batch_size_for_time_estimation=self.threshold_batch_size_for_time_estimation,
                batch_execution_time_caching_map=self._batch_execution_time_caching_maps,
            )
            simulate_predict_replica_scheduler.simulate()
            return simulate_predict_replica_scheduler.target_request_end_to_end

    def predict_average_latency(self, replica_scheduler, request):
        if self._enable_fast_cloning:
            base_simulator = self._get_base_simulator(replica_scheduler)
            cloned_simulator = base_simulator.clone(request)
            cloned_simulator.simulate(from_cloned=True)
            return cloned_simulator.average_latency
        else:
            simulate_predict_replica_scheduler = SimulatePredictReplicaScheduler(
                replica_scheduler=replica_scheduler,
                request=request,
                execution_time_predictor=self._execution_time_predictor,
                use_estimated_execution_time=self.use_estimated_time,
                copy_replica_scheduler=self._copy_base_replica_scheduler,
                threshold_batch_size_for_time_estimation=self.threshold_batch_size_for_time_estimation,
                batch_execution_time_caching_map=self._batch_execution_time_caching_maps,
            )
            simulate_predict_replica_scheduler.simulate()
            return simulate_predict_replica_scheduler.average_latency

    def predict_average_batch_size(self, replica_scheduler, request):
        if self._enable_fast_cloning:
            base_simulator = self._get_base_simulator(replica_scheduler)
            cloned_simulator = base_simulator.clone(request)
            cloned_simulator.simulate(from_cloned=True)
            return cloned_simulator.average_batch_size
        else:
            simulate_predict_replica_scheduler = SimulatePredictReplicaScheduler(
                replica_scheduler=replica_scheduler,
                request=request,
                execution_time_predictor=self._execution_time_predictor,
                use_estimated_execution_time=False,
                copy_replica_scheduler=self._copy_base_replica_scheduler,
                threshold_batch_size_for_time_estimation=self.threshold_batch_size_for_time_estimation,
                batch_execution_time_caching_map=self._batch_execution_time_caching_maps,
            )
            simulate_predict_replica_scheduler.simulate()
            return simulate_predict_replica_scheduler.average_batch_size

    def predict_average_execution_latency(self, replica_scheduler, request):
        if self._enable_fast_cloning:
            base_simulator = self._get_base_simulator(replica_scheduler)
            cloned_simulator = base_simulator.clone(request)
            cloned_simulator.simulate(from_cloned=True)
            return cloned_simulator.average_execution_time
        else:
            simulate_predict_replica_scheduler = SimulatePredictReplicaScheduler(
                replica_scheduler=replica_scheduler,
                request=request,
                execution_time_predictor=self._execution_time_predictor,
                use_estimated_execution_time=self.use_estimated_time,
                copy_replica_scheduler=self._copy_base_replica_scheduler,
                threshold_batch_size_for_time_estimation=self.threshold_batch_size_for_time_estimation,
                batch_execution_time_caching_map=self._batch_execution_time_caching_maps,
            )
            simulate_predict_replica_scheduler.simulate()
            return simulate_predict_replica_scheduler.average_execution_time

    def predict_waiting_and_ending_time(self, replica_scheduler, request):
        if self._enable_fast_cloning:
            base_simulator = self._get_base_simulator(replica_scheduler)
            cloned_simulator = base_simulator.clone(request)
            cloned_simulator.simulate(from_cloned=True)
            return cloned_simulator.target_request_end_to_end
        else:
            simulate_predict_replica_scheduler = SimulatePredictReplicaScheduler(
                replica_scheduler=replica_scheduler,
                request=request,
                execution_time_predictor=self._execution_time_predictor,
                use_estimated_execution_time=self.use_estimated_time,
                copy_replica_scheduler=self._copy_base_replica_scheduler,
                threshold_batch_size_for_time_estimation=self.threshold_batch_size_for_time_estimation,
                batch_execution_time_caching_map=self._batch_execution_time_caching_maps,
            )
            simulate_predict_replica_scheduler.simulate()
            return simulate_predict_replica_scheduler.target_request_end_to_end
