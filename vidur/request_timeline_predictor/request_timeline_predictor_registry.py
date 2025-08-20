from vidur.request_timeline_predictor.simulate_request_timeline_predictor import SimulateRequestTimelinePredictor
from vidur.types.request_timeline_predictor_type import RequestTimelinePredictorType
from vidur.utils.base_registry import BaseRegistry


class RequestTimelinePredictorRegistry(BaseRegistry):
    @classmethod
    def get_key_from_str(cls, key_str: str) -> RequestTimelinePredictorType:
        return RequestTimelinePredictorType.from_str(key_str)


RequestTimelinePredictorRegistry.register(RequestTimelinePredictorType.SIMULATE, SimulateRequestTimelinePredictor)
