from abc import ABC, abstractmethod

import aiohttp
import time


class Instance(ABC):
    def __init__(self, instance_id,
                 hostname,
                 ip_address,
                 predictor_ports,
                 model_name,
                 query_predictor_timeout,
                 query_backend_timeout,
                 backend_port=8000):
        self._instance_id = instance_id
        self._hostname = hostname
        self._predictor_ports = predictor_ports
        self._backend_port = backend_port
        self._predictor_urls = [f"http://{ip_address}:{port}/predict" for port in predictor_ports]
        self._ip_address = ip_address
        self._model_name = model_name
        self.total_request = 0
        self.start_time = time.time()
        self.request_timeline = []
        self._predicted_latency = {}
        self.predicted_error = []
        self.predicted_error_ratio = []
        self.serving_time = []
        self._predictor_timeout = aiohttp.ClientTimeout(total=query_predictor_timeout)
        self._backend_timeout = aiohttp.ClientTimeout(total=query_backend_timeout)
        self._session = None

    async def get_session(self):
        if self._session is None or self._session.closed:
            # Optimized connector for high throughput
            connector = aiohttp.TCPConnector(
                limit=0,  # Unlimited parallel connections
                ttl_dns_cache=300,  # Cache DNS for 5 minutes
                use_dns_cache=True,
                keepalive_timeout=60  # Keep sockets open for 60s
            )
            self._session = aiohttp.ClientSession(
                timeout=self._backend_timeout,
                connector=connector
            )
        return self._session

    async def query_predictor(self, request_id: int,
                              num_context_tokens: int,
                              predicted_num_context_tokens: dict):
        predict_parameters = {
            "id": request_id,
            "num_prompt_len": num_context_tokens,
            "num_predicted_output_len": predicted_num_context_tokens[self._model_name],
        }
        predict_url = self._predictor_urls[request_id % len(self._predictor_urls)]
        async with aiohttp.ClientSession(timeout=self._predictor_timeout) as session:
            async with session.post(predict_url, json=predict_parameters, ssl=False) as response:
                response_dict = await response.json()
                response_dict['instance_id'] = self._instance_id
                self._predicted_latency[request_id] = response_dict['latency_prediction']
                return response_dict

    @abstractmethod
    async def query_backend(self, payload: dict, headers: dict = None):
        pass

    async def query_instance(self,
                            payload: dict,
                            predicted_num_decode_tokens: int):
        self.request_timeline.append(time.time() - self.start_time)
        self.total_request += 1
        start = time.time()
        request_id = payload.get("request_id")
        response_dict = await self.query_backend(
            payload,
            headers={}
        )
        serving_time = time.time() - start
        response_dict['serving_time'] = serving_time
        response_dict['instance_id'] = self._instance_id
        response_dict['host'] = self._hostname

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


    @property
    def model_name(self):
        return self._model_name






