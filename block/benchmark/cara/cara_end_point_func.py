import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Any

import aiohttp
from tqdm import tqdm
from vllm.benchmarks.lib.endpoint_request_func import RequestFunc, RequestFuncInput



def _update_headers_common(
    headers: dict[str, Any],
    request_func_input: RequestFuncInput,
) -> None:
    """Update headers with common fields. Copied from vLLM's endpoint_request_func.py."""
    if request_func_input.extra_headers:
        headers |= request_func_input.extra_headers
    if request_func_input.request_id:
        headers["x-request-id"] = request_func_input.request_id


@dataclass
class RequestFuncOutput:
    """
       The output of the request function including metrics.
       Should be aligned with vLLM's RequestFuncOutput, but extend with model name and scheduling overhead.
    """
    generated_text: str = ""
    success: bool = False
    latency: float = 0.0  # Client-side E2E latency (user-perceived)
    output_tokens: int = 0
    ttft: float = 0.0  # Time to first token (server-side measurement)
    itl: list[float] = field(default_factory=list)  # Inter-token latencies (server-side)
    tpot: float = 0.0  # avg next-token latencies
    prompt_len: int = 0
    error: str = ""
    start_time: float = 0.0
    model: str = ""
    request_id: str = ""
    scheduling_overhead: float = 0.0  # CARA-specific: client E2E - server E2E = network + CARA routing
    instance_id: str = ""  # Instance that handled the request
    host: str = ""  # Host IP address of the instance


async def async_request_cara_openai_completions(
    request_func_input: RequestFuncInput,
    session: aiohttp.ClientSession,
    pbar: tqdm | None = None,
) -> RequestFuncOutput:
    """
    The async request function for creating OpenAI Completions API to call Cara backend.
    Args:
        request_func_input: The input for the request function.
        pbar: The progress bar to display the progress.
        session: The aiohttp session to use.
    Returns:
        The output of the request function.
    """
    api_url = request_func_input.api_url
    payload = {
        "request_id": request_func_input.request_id,
        # the model passing will be ignored and get resolved by Cara server side
        "model": "cara",
        "prompt": request_func_input.prompt,
        "prompt_len": request_func_input.prompt_len,
        "temperature": 0.0,
        "repetition_penalty": 1.0,
        "max_tokens": request_func_input.output_len,
        # CARA server returns complete JSON response, not streaming
        "stream": False,
    }
    headers = {
        "Content-Type": "application/json",
    }
    _update_headers_common(headers, request_func_input)
    output = RequestFuncOutput()
    output.prompt_len = request_func_input.prompt_len
    output.request_id = request_func_input.request_id

    st = time.perf_counter()
    output.start_time = st
    try:
        async with session.post(url=api_url, json=payload, headers=headers) as response:
            if response.status == 200:
                response_map = await response.json()
                # CARA server returns success field to indicate if request was processed
                if response_map.get("success", False):
                    output.prompt_len = request_func_input.prompt_len
                    output.success = True

                    # Measure client-side E2E latency (user-perceived latency)
                    # This includes: network time + CARA scheduling overhead + backend processing
                    output.latency = time.perf_counter() - st

                    # Get server-side metrics from backend instance
                    output.output_tokens = response_map.get("output_tokens", 0)
                    output.generated_text = response_map.get("generated_text", "")
                    output.ttft = response_map.get("ttft", 0.0)  # Backend's time to first token
                    output.itl = response_map.get("itl", [])     # Backend's inter-token latencies
                    output.model = response_map.get("model", "")
                    output.instance_id = response_map.get("instance_id", "")
                    output.host = response_map.get("host", "")

                    # Get server-side E2E latency (reported by backend instance)
                    server_latency = response_map.get("server_latency", 0.0)

                    # Calculate scheduling overhead: client E2E - server E2E
                    # This captures: client->CARA network + CARA routing/scheduling + backend->client network
                    # Using server-reported E2E is more accurate than ttft + sum(itl) because it includes
                    # all backend overhead (response serialization, etc.)
                    output.scheduling_overhead = output.latency - server_latency

                else:
                    # Request failed on CARA server side
                    output.success = False
                    output.error = response_map.get("error", "Unknown error from CARA server")
                    output.instance_id = response_map.get("instance_id", "")
                    output.host = response_map.get("host", "")
                    output.model = response_map.get("model", "")
            else:
                # HTTP error - try to parse response for debugging info
                output.success = False
                try:
                    error_data = await response.json()
                    output.error = error_data.get("error", f"HTTP {response.status}: {response.reason or 'Unknown error'}")
                    output.instance_id = error_data.get("instance_id", "")
                    output.host = error_data.get("host", "")
                    output.model = error_data.get("model", "")
                except:
                    output.error = f"HTTP {response.status}: {response.reason or 'Unknown error'}"
    except Exception:
        output.success = False
        exc_info = sys.exc_info()
        output.error = "".join(traceback.format_exception(*exc_info))

    if pbar:
        pbar.update(1)
    return output


# Create the CARA async request functions dictionary along with VLLM's request functions
CARA_ASYNC_REQUEST_FUNCS : dict[str, RequestFunc] = {
    "cara": async_request_cara_openai_completions,
}