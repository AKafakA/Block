import asyncio
import json
import time
from block.global_scheduler.cara.cara_instance.Instance import Instance
from block.global_scheduler.cara.utils import MAX_EMPTY_READS_BEFORE_TIMEOUT


class OllamaInstance(Instance):

    def __init__(self, instance_id,
                 hostname,
                 ip_address,
                 predictor_ports,
                 model_name,
                 query_predictor_timeout=10,
                 query_backend_timeout=2 * 60 * 60,  # 60 minutes timeout for Ollama
                 backend_port=11434):  # Default Ollama port
        super().__init__(instance_id,
                         hostname,
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
        st = time.perf_counter()  # Server-side E2E start time
        most_recent_timestamp = st
        ttft = 0
        itl = []
        output_tokens = 0
        success = False
        error = ""
        server_e2e_latency = 0.0  # Total time from request start to response complete

        # Adapt payload for Ollama API
        ollama_payload = {
            "model": self._model_name,
            "prompt": payload["prompt"],
            "stream": True,
            "raw": True,
            # Add stop tokens at top level (not in options) to prevent infinite repetition
            "options": {
                "temperature": 0.0,
                "repeat_penalty": payload.get("repetition_penalty", 1.0),
                # Ollama uses 'num_predict' instead of 'max_tokens'
                "num_predict": min(payload["max_tokens"], 8192),
                "stop": payload.get("stop", []),
            }
        }

        # Ollama generally doesn't require an API key, but we keep headers logic flexible
        if not headers:
            headers = {}

        session = await self.get_session()
        try:
            async with session.post(self.api_url, json=ollama_payload, headers=headers) as response:
                if response.status == 200:
                    first_token_with_text_received = False  # Track first token with actual text for TTFT
                    received_done_signal = False  # Track if we got the final done: true chunk
                    chunks_received = 0
                    last_chunk_data = None
                    empty_read_count = 0  # Safety counter to prevent infinite loops

                    # Read line by line for newline-delimited JSON
                    try:
                        while not received_done_signal:
                            line = await response.content.readline()

                            # Handle empty reads - check if truly EOF or just temporary
                            if line == b'':
                                # Check if connection is actually closed
                                if response.content.at_eof():
                                    # True EOF - stream ended
                                    break

                                # Not EOF yet - might be slow generation, wait briefly
                                empty_read_count += 1

                                # Safety check: prevent infinite waiting
                                if empty_read_count >= MAX_EMPTY_READS_BEFORE_TIMEOUT:
                                    error = (
                                        f"Stream stalled: {MAX_EMPTY_READS_BEFORE_TIMEOUT} consecutive empty reads. "
                                        f"Generated text length: {len(generated_text)}, "
                                        f"Chunks received: {chunks_received}"
                                    )
                                    break

                                # Small delay to avoid busy waiting
                                await asyncio.sleep(0.001)  # 1ms instead of 10msBu
                                continue

                            # Reset empty read counter when we get data
                            empty_read_count = 0

                            line = line.strip()
                            if not line:
                                continue

                            chunks_received += 1

                            # Parse the JSON line
                            try:
                                data = json.loads(line)
                                last_chunk_data = data  # Keep track of last valid chunk
                            except json.JSONDecodeError as je:
                                # Log the decode error but continue
                                continue

                            # Process the JSON object
                            # Ollama response format: {"response": "token", "done": false, ...}
                            # Final response: {"done": true, "eval_count": 100, ...}

                            if data.get("done"):
                                # Request is done, capture usage stats
                                # Ollama provides 'eval_count' as the output token count
                                output_tokens = data.get("eval_count", 0)
                                received_done_signal = True
                                success = True  # Successfully received the completion signal
                                break
                            else:
                                text = data.get("response", "")
                                timestamp = time.perf_counter()

                                # TTFT (Time To First Token) - only count if we got actual text
                                if not first_token_with_text_received and text:
                                    first_token_with_text_received = True
                                    ttft = time.perf_counter() - st
                                elif first_token_with_text_received and text:
                                    # ITL (Inter-Token Latency) - only track for chunks with text
                                    itl.append(timestamp - most_recent_timestamp)

                                if text:  # Only update timestamp for chunks with actual text
                                    most_recent_timestamp = timestamp
                                    generated_text += text
                    except asyncio.TimeoutError:
                        # Stream reading timed out - mark as failure
                        success = False
                        error = (
                            f"Timeout while reading stream. "
                            f"Generated text length: {len(generated_text)}, "
                            f"Chunks received: {chunks_received}, "
                            f"First token received: {first_token_with_text_received}"
                        )

                    # If we didn't receive the done signal, mark as failure
                    if not received_done_signal and not error:
                        success = False
                        # Estimate output tokens for debugging
                        estimated_tokens = len(generated_text) // 4
                        expected_tokens = ollama_payload["options"]["num_predict"]
                        completion_ratio = estimated_tokens / expected_tokens if expected_tokens > 0 else 0

                        error = (
                            f"Stream ended without 'done' signal. "
                            f"Generated text length: {len(generated_text)}, "
                            f"Estimated tokens: {estimated_tokens}/{expected_tokens} ({completion_ratio:.1%}), "
                            f"Chunks received: {chunks_received}, "
                            f"First token received: {first_token_with_text_received}"
                        )
                else:
                    error = f"HTTP {response.status}: {response.reason}"
                    success = False
        except asyncio.TimeoutError:
            success = False
            error = f"Request timeout after {self._backend_timeout.total}s"
        except Exception as e:
            success = False
            error = f"{type(e).__name__}: {str(e)}"

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