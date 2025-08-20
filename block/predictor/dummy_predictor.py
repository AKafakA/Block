from block.predictor.predictor import Predictor


class DummyPredictor(Predictor):
    def __init__(self, config, port):
        super().__init__(config, port)

    async def predict(self, target_request):
        return {
            "target_metric": 1.0,
            "gpu_blocks": 1.0,
            "num_requests": 1.0,
            "num_preempted": 1.0
        }
