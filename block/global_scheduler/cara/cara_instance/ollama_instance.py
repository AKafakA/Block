import json
import time
import aiohttp
from block.global_scheduler.cara.cara_instance.Instance import Instance


class OllamaStreamHandler:
    """
    Handles streaming responses from Ollama's /api/generate endpoint.
    Ollama returns a stream of JSON objects, usually separated by newlines.
    """
    def __init__(self):
        self.buffer = ""

    def add_chunk(self, chunk_bytes: bytes) -> list[dict]:
        """Add a chunk of bytes to the buffer and return complete JSON objects."""
        chunk_str = chunk_bytes.decode("utf-8")
        self.buffer += chunk_str
        messages = []

        # Ollama sends newline-delimited JSON objects
        while "\n" in self.buffer:
            message, self.buffer = self.buffer.split("\n", 1)
            message = message.strip()
            if message:
                try:
                    json_msg = json.loads(message)
                    messages.append(json_msg)
                except json.JSONDecodeError:
                    # Incomplete JSON (unlikely with line split, but safe to keep)
                    pass
        return messages


class OllamaInstance(Instance):

    def __init__(self, instance_id,
                 ip_address,
                 predictor_ports,
                 model_name,
                 query_predictor_timeout=10,
                 query_backend_timeout=10 * 60 * 2,
                 backend_port=11434):  # Default Ollama port
        super().__init__(instance_id,
                         ip_address,
                         predictor_ports,
                         model_name,
                         query_predictor_timeout,
                         query_backend_timeout,
                         backend_port)
        # Ollama native generation endpoint
        self.api_url = f"http://{ip_address}:{backend_port}/api/generate"

    async def query_backend(self, payload: dict, headers: dict = None):

        generated_text = ""
        st = time.perf_counter()
        most_recent_timestamp = st
        ttft = 0
        itl = []
        output_tokens = 0
        success = False
        error = ""

        # Adapt payload for Ollama API
        ollama_payload = {
            "model": self._model_name,
            "prompt": payload["prompt"],
            "stream": True,
            "options": {
                "temperature": 0.0,
                "repeat_penalty": 1.0,
                # Ollama uses 'num_predict' instead of 'max_tokens'
                "num_predict": min(payload["max_tokens"], 8192),
            }
        }

        # Ollama generally doesn't require an API key, but we keep headers logic flexible
        if not headers:
            headers = {}

        async with aiohttp.ClientSession(timeout=self._backend_timeout) as session:
            try:
                async with session.post(self.api_url, json=ollama_payload, headers=headers) as response:
                    if response.status == 200:
                        first_chunk_received = False
                        handler = OllamaStreamHandler()

                        async for chunk_bytes in response.content.iter_any():
                            chunk_bytes = chunk_bytes.strip()
                            if not chunk_bytes:
                                continue

                            # Parse JSON objects from the stream
                            json_responses = handler.add_chunk(chunk_bytes)

                            for data in json_responses:
                                # Ollama response format: {"response": "token", "done": false, ...}
                                # Final response: {"done": true, "eval_count": 100, ...}

                                if not data.get("done"):
                                    text = data.get("response", "")
                                    timestamp = time.perf_counter()

                                    # TTFT (Time To First Token) - only count if we got actual text
                                    if not first_chunk_received and text:
                                        first_chunk_received = True
                                        ttft = time.perf_counter() - st
                                    elif first_chunk_received:
                                        # ITL (Inter-Token Latency)
                                        itl.append(timestamp - most_recent_timestamp)

                                    most_recent_timestamp = timestamp
                                    generated_text += text
                                else:
                                    # Request is done, capture usage stats
                                    # Ollama provides 'eval_count' as the output token count
                                    output_tokens = data.get("eval_count", 0)
                                    break

                        if first_chunk_received:
                            success = True
                        else:
                            success = False
                            error = (
                                "Never received a valid chunk to calculate TTFT. "
                                "This response will be marked as failed!"
                            )
                    else:
                        error = f"HTTP {response.status}: {response.reason}"
                        success = False
            except Exception as e:
                success = False
                error = str(e)

        return {
            "generated_text": generated_text,
            "ttft": ttft,
            "itl": itl,
            "output_tokens": output_tokens,
            "success": success,
            "error": error,
        }