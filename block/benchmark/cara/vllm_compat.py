"""
Compatibility layer for vLLM benchmarks module.

This module provides fallback implementations for vLLM benchmark utilities
that are not available in older vLLM versions (e.g., cara_p100_v_6.0 for P100 nodes).

For newer vLLM versions (A100 nodes), it imports from the official vllm.benchmarks module.
For older versions, it provides minimal implementations to maintain compatibility.
"""

import argparse
import asyncio
import contextlib
import gc
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Optional, Protocol

import aiohttp
import uuid
from transformers import AutoTokenizer


# =============================================================================
# Try importing from new vLLM benchmarks module (A100 nodes)
# =============================================================================
try:
    from vllm.benchmarks.datasets import (
        SampleRequest,
        BenchmarkDataset,
        CustomDataset,
        is_valid_sequence,
        add_dataset_parser,
        get_samples as vllm_get_samples,
    )
    from vllm.benchmarks.lib.endpoint_request_func import (
        ASYNC_REQUEST_FUNCS,
        OPENAI_COMPATIBLE_BACKENDS,
        RequestFuncInput,
        RequestFunc,
    )
    from vllm.benchmarks.lib.ready_checker import wait_for_endpoint
    from vllm.benchmarks.lib.utils import (
        convert_to_pytorch_benchmark_format,
        write_to_json,
    )

    VLLM_BENCHMARKS_AVAILABLE = True

except ImportError:
    # =============================================================================
    # Fallback implementations for old vLLM versions (P100 nodes)
    # =============================================================================
    VLLM_BENCHMARKS_AVAILABLE = False

    # -------------------------------------------------------------------------
    # Dataset classes and utilities
    # -------------------------------------------------------------------------

    @dataclass
    class SampleRequest:
        """Represents a sample request for benchmarking."""
        prompt: str
        prompt_len: int
        expected_output_len: int
        request_id: str = ""
        multi_modal_data: Optional[dict] = None
        lora_path: Optional[str] = None


    class BenchmarkDataset:
        """Base class for benchmark datasets."""
        def __init__(
            self,
            dataset_path: Optional[str] = None,
            random_seed: int = 0,
            **kwargs
        ):
            self.dataset_path = dataset_path
            self.random_seed = random_seed
            self.data = []

        def load_data(self) -> None:
            """Load dataset. Override in subclasses."""
            raise NotImplementedError

        def sample(self, *args, **kwargs) -> list:
            """Sample requests from dataset. Override in subclasses."""
            raise NotImplementedError

        def maybe_oversample_requests(
            self,
            samples: list,
            num_requests: int,
            request_id_prefix: str = "",
            no_oversample: bool = False,
        ) -> None:
            """Oversample requests if needed."""
            if no_oversample or len(samples) >= num_requests:
                return

            # Simple oversampling by repeating samples
            while len(samples) < num_requests:
                idx = len(samples) % len(samples) if samples else 0
                original = samples[idx]
                samples.append(
                    SampleRequest(
                        prompt=original.prompt,
                        prompt_len=original.prompt_len,
                        expected_output_len=original.expected_output_len,
                        request_id=request_id_prefix + str(len(samples)),
                    )
                )


    class CustomDataset(BenchmarkDataset):
        """Custom dataset loaded from JSONL file."""
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.load_data()

        def load_data(self) -> None:
            """Load data from JSONL file."""
            if self.dataset_path is None:
                raise ValueError("dataset_path must be provided")

            import json
            self.data = []
            with open(self.dataset_path, 'r') as f:
                for line in f:
                    if line.strip():
                        self.data.append(json.loads(line))

        def sample(
            self,
            tokenizer,
            num_requests: int,
            output_len: Optional[int] = None,
            request_id_prefix: str = "",
            no_oversample: bool = False,
            max_total_len: int = 2048,
            **kwargs,
        ) -> list:
            """Sample requests from custom dataset."""
            samples = []
            for idx, entry in enumerate(self.data):
                if len(samples) >= num_requests:
                    break

                prompt = entry.get("prompt", "")
                prompt_ids = tokenizer(prompt).input_ids
                prompt_len = len(prompt_ids)

                expected_len = entry.get("output_len", output_len or 128)

                if prompt_len + expected_len > max_total_len:
                    continue

                samples.append(
                    SampleRequest(
                        prompt=prompt,
                        prompt_len=prompt_len,
                        expected_output_len=expected_len,
                        request_id=request_id_prefix + str(idx),
                    )
                )

            self.maybe_oversample_requests(
                samples, num_requests, request_id_prefix, no_oversample
            )
            return samples


    def is_valid_sequence(
        prompt_len: int,
        output_len: int,
        max_total_len: int = 2048,
        min_output_len: int = 4,
        skip_min_output_len_check: bool = False,
    ) -> bool:
        """Check if sequence length is valid."""
        if prompt_len + output_len > max_total_len:
            return False
        if not skip_min_output_len_check and output_len < min_output_len:
            return False
        return True


    def add_dataset_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add dataset-related arguments to parser."""
        parser.add_argument(
            "--dataset-name",
            type=str,
            default="sharegpt",
            help="Name of the dataset to use (sharegpt, sonnet, custom, lmsys, etc.)",
        )
        parser.add_argument(
            "--dataset-path",
            type=str,
            default=None,
            help="Path to the dataset file",
        )
        parser.add_argument(
            "--num-prompts",
            type=int,
            default=1000,
            help="Number of prompts to process",
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=0,
            help="Random seed for reproducibility",
        )
        parser.add_argument(
            "--custom-output-len",
            type=int,
            default=None,
            help="Custom output length for generation (overrides dataset default)",
        )
        parser.add_argument(
            "--skip-chat-template",
            action="store_true",
            help="Skip applying chat template to prompts",
        )
        parser.add_argument(
            "--request-id-prefix",
            type=str,
            default=f"bench-{uuid.uuid4().hex[:8]}-",
            help="Prefix for request IDs",
        )
        parser.add_argument(
            "--no-oversample",
            action="store_true",
            help="Do not oversample requests if dataset is smaller than num-prompts",
        )
        parser.add_argument(
            "--max-total-len",
            type=int,
            default=2048,
            help="Maximum total sequence length (prompt + output)",
        )
        return parser


    def vllm_get_samples(args, tokenizer):
        """Fallback implementation - should not be called in CARA."""
        raise NotImplementedError(
            "vllm.benchmarks.datasets.get_samples not available. "
            "Use block.benchmark.cara.dataset.get_samples instead."
        )


    # -------------------------------------------------------------------------
    # Endpoint request function utilities
    # -------------------------------------------------------------------------

    @dataclass
    class RequestFuncInput:
        """Input for request function."""
        prompt: str
        api_url: str
        prompt_len: int
        output_len: int
        model: str
        best_of: int = 1
        use_beam_search: bool = False
        request_id: str = ""


    class RequestFunc(Protocol):
        """Protocol for request functions."""
        async def __call__(
            self,
            request_func_input: RequestFuncInput,
            pbar: Optional[Any] = None,
        ) -> Any:
            ...


    # Minimal set of async request functions for old vLLM
    ASYNC_REQUEST_FUNCS = {}
    OPENAI_COMPATIBLE_BACKENDS = ["vllm", "openai"]


    # -------------------------------------------------------------------------
    # Ready checker utilities
    # -------------------------------------------------------------------------

    async def wait_for_endpoint(
        endpoint_url: str,
        timeout: int = 300,
        check_interval: int = 5,
    ) -> bool:
        """Wait for endpoint to become ready."""
        print(f"Waiting for endpoint {endpoint_url} to become ready...")
        start_time = time.time()

        async with aiohttp.ClientSession() as session:
            while time.time() - start_time < timeout:
                try:
                    async with session.get(
                        f"{endpoint_url}/health",
                        timeout=aiohttp.ClientTimeout(total=check_interval)
                    ) as response:
                        if response.status == 200:
                            print(f"Endpoint {endpoint_url} is ready!")
                            return True
                except (aiohttp.ClientError, Exception):
                    pass

                await asyncio.sleep(check_interval)

        print(f"Endpoint {endpoint_url} did not become ready within {timeout}s")
        return False


    # -------------------------------------------------------------------------
    # Utility functions
    # -------------------------------------------------------------------------

    def convert_to_pytorch_benchmark_format(
        benchmark_result: dict,
        model_id: str,
    ) -> dict:
        """Convert benchmark results to PyTorch format."""
        # Simple passthrough for compatibility
        return {
            "model_id": model_id,
            **benchmark_result,
        }


    def write_to_json(data: dict, filename: str) -> None:
        """Write data to JSON file."""
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)


# =============================================================================
# Additional vLLM utilities (for old vLLM versions without these modules)
# =============================================================================

# Try importing from new vLLM utils
try:
    from vllm.transformers_utils.tokenizer import get_tokenizer
except ImportError:
    # Fallback for old vLLM
    def get_tokenizer(
        tokenizer_name: str,
        *args,
        trust_remote_code: bool = False,
        tokenizer_mode: str = "auto",
        **kwargs
    ):
        """Fallback tokenizer loader using HuggingFace transformers."""
        return AutoTokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=trust_remote_code,
            **kwargs
        )

try:
    from vllm.utils.gc_utils import freeze_gc_heap
except ImportError:
    # Fallback for old vLLM
    @contextlib.contextmanager
    def freeze_gc_heap():
        """Fallback GC heap freezer (no-op context manager)."""
        # Disable GC during the context
        gc_enabled = gc.isenabled()
        try:
            if gc_enabled:
                gc.collect()
                gc.disable()
            yield
        finally:
            if gc_enabled:
                gc.enable()

try:
    from vllm.utils.network_utils import join_host_port
except ImportError:
    # Fallback for old vLLM
    def join_host_port(host: str, port: int) -> str:
        """Join host and port into URL format."""
        # Handle IPv6 addresses
        if ':' in host and not host.startswith('['):
            host = f'[{host}]'
        return f"{host}:{port}"


# =============================================================================
# Export unified interface
# =============================================================================

__all__ = [
    # Dataset utilities
    "SampleRequest",
    "BenchmarkDataset",
    "CustomDataset",
    "is_valid_sequence",
    "add_dataset_parser",
    "vllm_get_samples",
    # Request function utilities
    "RequestFuncInput",
    "RequestFunc",
    "ASYNC_REQUEST_FUNCS",
    "OPENAI_COMPATIBLE_BACKENDS",
    # Endpoint utilities
    "wait_for_endpoint",
    # Result utilities
    "convert_to_pytorch_benchmark_format",
    "write_to_json",
    # vLLM utilities
    "get_tokenizer",
    "freeze_gc_heap",
    "join_host_port",
    # Compatibility flag
    "VLLM_BENCHMARKS_AVAILABLE",
]
