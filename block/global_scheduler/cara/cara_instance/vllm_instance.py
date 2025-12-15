import json
import os
import time

from block.global_scheduler.cara.cara_instance.Instance import Instance
import aiohttp


class StreamedResponseHandler:
    """Copied from vLLM endpoint_request_func"""

    def __init__(self):
        self.buffer = ""

    def add_chunk(self, chunk_bytes: bytes) -> list[str]:
        """Add a chunk of bytes to the buffer and return any complete
        messages."""
        chunk_str = chunk_bytes.decode("utf-8")
        self.buffer += chunk_str

        messages = []

        # Split by double newlines (SSE message separator)
        while "\n\n" in self.buffer:
            message, self.buffer = self.buffer.split("\n\n", 1)
            message = message.strip()
            if message:
                messages.append(message)

        # if self.buffer is not empty, check if it is a complete message
        # by removing data: prefix and check if it is a valid JSON
        if self.buffer.startswith("data: "):
            message_content = self.buffer.removeprefix("data: ").strip()
            if message_content == "[DONE]":
                messages.append(self.buffer.strip())
                self.buffer = ""
            elif message_content:
                try:
                    json.loads(message_content)
                    messages.append(self.buffer.strip())
                    self.buffer = ""
                except json.JSONDecodeError:
                    # Incomplete JSON, wait for more chunks.
                    pass

        return messages

class VllmInstance(Instance):

    def __init__(self, instance_id,
                 hostname,
                 ip_address,
                 predictor_ports,
                 model_name,
                 query_predictor_timeout=10,
                 query_backend_timeout=30 * 60,  # 30 minutes timeout for vLLM
                 backend_port=8000):
        super().__init__(instance_id,
                         hostname,
                         ip_address,
                         predictor_ports,
                         model_name,
                         query_predictor_timeout,
                         query_backend_timeout,
                         backend_port)
        self.api_url = f"http://{ip_address}:{backend_port}/v1/completions"

    async def query_backend(self, payload: dict, headers: dict = None):

        generated_text = ""
        st = time.perf_counter()  # Server-side E2E start time
        most_recent_timestamp = st
        ttft = 0
        itl = []
        output_tokens = 0
        success = False
        error = ""
        server_e2e_latency = 0.0  # Total time from request start to response complete

        vllm_payload = {
            "model": self._model_name,
            "prompt": payload["prompt"],
            "temperature": 0.0,
            "repetition_penalty": payload.get("repetition_penalty", 1.0),
            "max_tokens": payload["max_tokens"],
            "logprobs": None,
            "stream": True,
            "stream_options": {
                "include_usage": True,
            },
            # Add stop tokens if provided to prevent infinite repetition
            "stop": payload.get("stop", []),
        }

        if not headers:
            headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"}

        async with aiohttp.ClientSession(timeout=self._backend_timeout) as session:
            async with session.post(self.api_url, json=vllm_payload, ssl=False, headers=headers) as response:
                if response.status == 200:
                    first_chunk_received = False
                    handler = StreamedResponseHandler()
                    async for chunk_bytes in response.content.iter_any():
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue
                        messages = handler.add_chunk(chunk_bytes)
                        for message in messages:
                            # NOTE: SSE comments (often used as pings) start with
                            # a colon. These are not JSON data payload and should
                            # be skipped.
                            if message.startswith(":"):
                                continue
                            chunk = message.removeprefix("data: ")

                            if chunk != "[DONE]":
                                data = json.loads(chunk)
                                # NOTE: Some completion API might have a last
                                # usage summary response without a token so we
                                # want to check a token was generated
                                if choices := data.get("choices"):
                                    # Note that text could be empty here
                                    # e.g. for special tokens
                                    text = choices[0].get("text")
                                    timestamp = time.perf_counter()
                                    # First token
                                    if not first_chunk_received:
                                        first_chunk_received = True
                                        ttft = time.perf_counter() - st
                                    # Decoding phase
                                    else:
                                        itl.append(timestamp - most_recent_timestamp)

                                    most_recent_timestamp = timestamp
                                    generated_text += text or ""
                                elif usage := data.get("usage"):
                                    output_tokens = usage.get("completion_tokens")
                    if first_chunk_received:
                        success = True
                    else:
                        success = False
                        error = (
                            "Never received a valid chunk to calculate TTFT."
                            "This response will be marked as failed!"
                        )
                    generated_text = generated_text
                else:
                    error = response.reason or ""
                    success = False

        # Calculate server-side E2E latency (total time from request start to completion)
        server_e2e_latency = time.perf_counter() - st

        return {
            "generated_text": generated_text,
            "ttft": ttft,
            "itl": itl,
            "output_tokens": output_tokens,
            "success": success,
            "error": error,
            "model": self._model_name,
            "server_latency": server_e2e_latency,  # Server-side E2E latency
            "instance_id": self._instance_id,
            "host": self._hostname,
        }







