import aiohttp
import time


class Instance:
    def __init__(self, instance_id,
                 ip_address,
                 predictor_ports,
                 backend_port,
                 query_predictor_timeout=10,
                 query_backend_timeout=10 * 60 * 2):
        self._instance_id = instance_id
        self._predictor_ports = predictor_ports
        self._backend_port = backend_port
        self._predictor_urls = [f"http://{ip_address}:{port}/predict" for port in predictor_ports]
        self._backend_url = f"http://{ip_address}:{backend_port}/generate_benchmark"
        self.ip_address = ip_address
        self.total_request = 0
        self.start_time = time.time()
        self.request_timeline = []
        self._predicted_latency = {}
        self.predicted_error = []
        self.predicted_error_ratio = []
        self.serving_time = []
        self._predictor_timeout = aiohttp.ClientTimeout(total=query_predictor_timeout)
        self._backend_timeout = aiohttp.ClientTimeout(total=query_backend_timeout)

    def __str__(self):
        return (f"Instance {self._instance_id} with predictor port {self._predictor_port} "
                f"and backend port {self._backend_port}")

    async def query_predictor(self, request_id: int,
                              num_context_tokens: int,
                              predicted_num_context_tokens: int,
                              arrived_at: float):
        predict_parameters = {
            "id": request_id,
            "arrival_time": arrived_at,
            "num_context_tokens": num_context_tokens,
            "num_decode_tokens": predicted_num_context_tokens,
        }
        predict_url = self._predictor_urls[request_id % len(self._predictor_urls)]
        async with aiohttp.ClientSession(timeout=self._predictor_timeout) as session:
            async with session.post(predict_url, json=predict_parameters, ssl=False) as response:
                response_dict = await response.json()
                response_dict['instance_id'] = self._instance_id
                self._predicted_latency[request_id] = response_dict['target_metric']
                return response_dict

    async def query_backend(self, prompt: str, max_response_len: int, request_id: int,
                            predicted_num_decode_tokens: int):
        self.request_timeline.append(time.time() - self.start_time)
        self.total_request += 1
        max_tokens = max_response_len
        request_dict = {
            "prompt": prompt,
            "n": 1,
            "best_of": 1,
            "temperature": 0.0,
            "top_k": 1,
            "max_tokens": max_tokens,
            "ignore_eos": True,
            "stream": False,
            "request_id": str(request_id),
            "num_predicted_tokens": predicted_num_decode_tokens,
        }
        start = time.time()
        async with aiohttp.ClientSession(timeout=self._backend_timeout) as session:
            async with session.post(self._backend_url, json=request_dict, ssl=False) as response:
                response_dict = await response.json()
                serving_time = time.time() - start
                response_dict['serving_time'] = serving_time
                response_dict['instance_id'] = self._instance_id
                if self._predicted_latency.get(request_id):
                    self.serving_time.append((serving_time, self._predicted_latency[request_id]))
                    self.predicted_error.append(serving_time - self._predicted_latency[request_id])
                    self.predicted_error_ratio.append(abs(serving_time - self._predicted_latency[request_id])
                                                      / serving_time)
                return response_dict

    def get_current_qpm(self):
        current_time = time.time()
        return sum([1 for time_of_request in self.request_timeline
                    if current_time - time_of_request <= 60])

    @property
    def predicted_latency(self):
        return self._predicted_latency.values()
