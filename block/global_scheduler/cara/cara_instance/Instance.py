from abc import ABC, abstractmethod

import aiohttp
import asyncio
import time
import random
import logging

logger = logging.getLogger(__name__)


class Instance(ABC):
    def __init__(self, instance_id,
                 hostname,
                 ip_address,
                 predictor_ports,
                 model_name,
                 query_predictor_timeout,
                 query_backend_timeout,
                 backend_port=8000,
                 enable_predictor_feedback=False,
                 feedback_sample_rate=1.0):
        self._instance_id = instance_id
        self._hostname = hostname
        self._predictor_ports = predictor_ports
        self._backend_port = backend_port
        self._predictor_urls = [f"http://{ip_address}:{port}/predict" for port in predictor_ports]
        self._predictor_log_urls = [f"http://{ip_address}:{port}/log_actual" for port in predictor_ports]
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
        # CARA predictor feedback settings
        self._enable_predictor_feedback = enable_predictor_feedback
        self._feedback_sample_rate = feedback_sample_rate
        # Round-robin counter for predictor selection (per-instance)
        self._predictor_index = 0
        # Track which predictor was used for each request (for feedback)
        self._request_to_predictor_index = {}

    def get_predictor_flush_urls(self):
        return [url.replace("/log_actual", "/flush") for url in self._predictor_log_urls]

    async def query_predictor(self, request_id: int,
                              num_context_tokens: int,
                              predicted_num_context_tokens: dict):
        predict_parameters = {
            "request_id": request_id,
            "num_prompt_tokens": num_context_tokens,
            "num_predicted_output_tokens": predicted_num_context_tokens[self._model_name],
        }
        # Round-robin predictor selection within this instance
        predictor_idx = self._predictor_index % len(self._predictor_urls)
        predict_url = self._predictor_urls[predictor_idx]
        self._predictor_index += 1

        # Store which predictor was used for this request (for feedback routing)
        self._request_to_predictor_index[request_id] = predictor_idx

        # Use singleton session with predictor timeout override
        async with aiohttp.ClientSession(timeout=self._predictor_timeout) as session:
            async with session.post(predict_url, json=predict_parameters, ssl=False, timeout=self._predictor_timeout) as response:
                response_dict = await response.json()
                response_dict['instance_id'] = self._instance_id
                # Store known fields defensively; CARA dummy predictor returns 'target_metric'
                if 'latency_prediction' in response_dict:
                    self._predicted_latency[request_id] = response_dict['latency_prediction']
                elif 'target_metric' in response_dict:
                    self._predicted_latency[request_id] = response_dict['target_metric']
                return response_dict

    @abstractmethod
    async def query_backend(self, payload: dict, headers: dict = None):
        pass

    async def _send_predictor_feedback(self, request_id: str, response_dict: dict):
        """Send actual metrics back to predictor for training data collection.

        Args:
            request_id: Request identifier
            response_dict: Response from backend containing actual metrics
        """
        # Sample based on feedback_sample_rate
        if random.random() > self._feedback_sample_rate:
            return

        try:
            # Extract metrics from response
            e2e_latency = response_dict.get('server_latency', 0)
            ttft = response_dict.get('ttft', 0)
            tpot = response_dict.get('tpot', 0)

            # Calculate TPOT from ITL if not provided
            if tpot == 0 and 'itl' in response_dict and response_dict['itl']:
                itl = response_dict['itl']
                if isinstance(itl, list) and len(itl) > 0:
                    tpot = sum(itl) / len(itl)

            # Send feedback to the SAME predictor that made the prediction
            predictor_idx = self._request_to_predictor_index.get(request_id)
            if predictor_idx is None:
                logger.warning(f"No predictor index found for request {request_id}, skipping feedback")
                return

            log_url = self._predictor_log_urls[predictor_idx]
            feedback_data = {
                "request_id": str(request_id),
                "e2e_latency": e2e_latency,
                "ttft": ttft,
                "tpot": tpot
            }

            async with aiohttp.ClientSession(timeout=self._predictor_timeout) as session:
                async with session.post(log_url, json=feedback_data, ssl=False) as response:
                    if response.status == 200:
                        logger.debug(f"Sent feedback for request {request_id} to predictor {predictor_idx}")
                    else:
                        logger.warning(f"Failed to send feedback: status {response.status}")

            # Clean up the mapping to prevent memory leak
            del self._request_to_predictor_index[request_id]

        except Exception as e:
            logger.warning(f"Error sending predictor feedback for {request_id}: {e}")

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

        # Send feedback to predictor for training data collection (CARA only)
        if self._enable_predictor_feedback:
            # Fire and forget - don't await to avoid blocking response
            asyncio.create_task(self._send_predictor_feedback(request_id, response_dict))

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




