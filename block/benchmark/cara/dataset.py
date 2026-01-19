# -----------------------------------------------------------------------------
# Lmsys Dataset Implementation
# -----------------------------------------------------------------------------
import os
import random
from dataclasses import dataclass
from typing import Optional

import pandas as pd
from transformers import PreTrainedTokenizerBase


# =============================================================================
# Local canonical classes - used consistently across all vLLM versions
# =============================================================================

@dataclass
class SampleRequest:
    """Canonical sample request format for CARA benchmarks.

    This local definition ensures consistent behavior across old/new vLLM versions.
    """
    prompt: str
    prompt_len: int
    expected_output_len: int
    request_id: str = ""
    multi_modal_data: Optional[dict] = None


def is_valid_sequence(
    prompt_len: int,
    output_len: int,
    min_len: int = 4,
    max_prompt_len: int = 1024,
    max_total_len: int = 2048,
    skip_min_output_len_check: bool = False,
) -> bool:
    """Check if sequence length is valid (matches vLLM's is_valid_sequence).

    Args:
        prompt_len: Length of prompt in tokens
        output_len: Length of output in tokens
        min_len: Minimum length for both prompt and output (default 4)
        max_prompt_len: Maximum prompt length (default 1024)
        max_total_len: Maximum total length prompt + output (default 2048)
        skip_min_output_len_check: Skip minimum output length check
    """
    prompt_too_short = prompt_len < min_len
    output_too_short = (not skip_min_output_len_check) and (output_len < min_len)
    prompt_too_long = prompt_len > max_prompt_len
    combined_too_long = (prompt_len + output_len) > max_total_len

    return not (prompt_too_short or output_too_short or prompt_too_long or combined_too_long)


class BenchmarkDataset:
    """Base class for benchmark datasets (local implementation)."""
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

        original_len = len(samples)
        if original_len == 0:
            return

        # Simple oversampling by repeating samples
        idx = 0
        while len(samples) < num_requests:
            original = samples[idx % original_len]
            samples.append(
                SampleRequest(
                    prompt=original.prompt,
                    prompt_len=original.prompt_len,
                    expected_output_len=original.expected_output_len,
                    request_id=request_id_prefix + str(len(samples)),
                    multi_modal_data=original.multi_modal_data,
                )
            )
            idx += 1

# -----------------------------------------------------------------------------
# ShareGPT Dataset Implementation
# -----------------------------------------------------------------------------

class LmsysDataset(BenchmarkDataset):
    """
    Implements the LMSYS dataset.  Loads data from a parquet file and generates
    sample requests based on conversation turns.
    """

    def __init__(self, **kwargs) -> None:
        # Accept disable_shuffle from kwargs (default False)
        self.disable_shuffle = bool(kwargs.pop("disable_shuffle", False))
        super().__init__(**kwargs)
        self.load_data()

    def load_data(self) -> None:
        if self.dataset_path is None:
            raise ValueError("dataset_path must be provided for loading data.")
        dataset_path = self.dataset_path
        dataset_list = []
        for file in os.listdir(dataset_path):
            path = os.path.join(dataset_path, file)
            if file.endswith('.parquet'):
                dataset_list.extend(pd.read_parquet(path).to_dict(orient='records'))

        self.data = [
            data for data in dataset_list
            if (
                    ("conversation" in data and len(data["conversation"]) >= 2)
            )
        ]
        random.seed(self.random_seed)
        if not getattr(self, "disable_shuffle", False):
            random.shuffle(self.data)

    def sample(
        self,
        tokenizer: PreTrainedTokenizerBase,
        num_requests: int,
        lora_path: str | None = None,
        max_loras: int | None = None,
        output_len: int | None = None,
        enable_multimodal_chat: bool = False,
        request_id_prefix: str = "",
        no_oversample: bool = False,
        min_prompt_len: int = 4,
        max_prompt_len: int = 1024,
        max_total_len: int = 2048,
        **kwargs,
    ) -> list:
        samples: list = []
        ind = 0
        for entry in self.data:
            if len(samples) >= num_requests:
                break
            prompt, completion = (
                entry["conversation"][0]["content"],
                entry["conversation"][1]["content"],
            )

            prompt_ids = tokenizer(prompt).input_ids
            completion_ids = tokenizer(completion).input_ids
            prompt_len = len(prompt_ids)
            new_output_len = len(completion_ids) if output_len is None else output_len
            if not is_valid_sequence(
                prompt_len,
                new_output_len,
                min_len=min_prompt_len,
                max_prompt_len=max_prompt_len,
                max_total_len=max_total_len,
                skip_min_output_len_check=output_len is not None,
            ):
                continue
            samples.append(
                SampleRequest(
                    prompt=prompt,
                    prompt_len=prompt_len,
                    expected_output_len=new_output_len,
                    request_id=request_id_prefix + str(ind),
                )
            )
            ind += 1
        self.maybe_oversample_requests(
            samples, num_requests, request_id_prefix, no_oversample
        )
        return samples


# -----------------------------------------------------------------------------
# Add the new Lmsys dataset to the dataset loader without modifying vLLM code.
# -----------------------------------------------------------------------------
class CaraCustomDataset(BenchmarkDataset):
    """Local custom JSONL dataset implementation used across all nodes.

    Ensures consistent defaults (e.g., output length) regardless of whether
    vLLM's benchmark package is available on the host.
    """

    def __init__(self, **kwargs) -> None:
        self.disable_shuffle = bool(kwargs.pop("disable_shuffle", False))
        super().__init__(**kwargs)
        self.load_data()

    def load_data(self) -> None:
        if self.dataset_path is None:
            raise ValueError("dataset_path must be provided for loading data.")
        self.data = []
        import json
        with open(self.dataset_path, 'r') as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))
        # Shuffle by default for randomized request order
        random.seed(self.random_seed)
        if not getattr(self, "disable_shuffle", False):
            random.shuffle(self.data)

    def sample(
        self,
        tokenizer: PreTrainedTokenizerBase,
        num_requests: int,
        output_len: int | None = None,
        request_id_prefix: str = "",
        no_oversample: bool = False,
        min_prompt_len: int = 4,
        max_prompt_len: int = 1024,
        max_total_len: int = 2048,
        **kwargs,
    ) -> list:
        samples: list = []
        for idx, entry in enumerate(self.data):
            if len(samples) >= num_requests:
                break
            prompt = entry.get("prompt", "")
            prompt_len = len(tokenizer(prompt).input_ids)
            expected_len = entry.get("output_len", output_len if output_len is not None else 1024)

            if not is_valid_sequence(
                prompt_len,
                expected_len,
                min_len=min_prompt_len,
                max_prompt_len=max_prompt_len,
                max_total_len=max_total_len,
                skip_min_output_len_check=(output_len is not None),
            ):
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

def get_samples(args, tokenizer) -> list[SampleRequest]:
    """
    Load samples from dataset based on args.dataset_name.

    Supports:
    - "custom" with lmsys path: LMSYS dataset (parquet files)
    - "custom" (other): JSONL custom dataset
    - Other types: Try vLLM's get_samples if available (A100 nodes only)

    Always returns local SampleRequest objects for consistency across vLLM versions.
    """
    if args.dataset_name == "custom":
        # Check if it's LMSYS dataset (directory with parquet files)
        if args.dataset_path and args.dataset_path.endswith("lmsys"):
            dataset = LmsysDataset(
                dataset_path=args.dataset_path,
                random_seed=args.seed,
                disable_shuffle=getattr(args, 'disable_shuffle', False),
            )
            return dataset.sample(
                num_requests=args.num_prompts,
                tokenizer=tokenizer,
                output_len=args.custom_output_len,
                skip_chat_template=args.skip_chat_template,
                request_id_prefix=args.request_id_prefix,
                no_oversample=args.no_oversample,
                min_prompt_len=args.min_prompt_len,
                max_prompt_len=args.max_prompt_len,
                max_total_len=args.max_total_len,
            )
        # Standard custom dataset (JSONL file) — force local implementation
        dataset = CaraCustomDataset(
            dataset_path=args.dataset_path,
            random_seed=args.seed,
            disable_shuffle=getattr(args, 'disable_shuffle', False),
        )
        return dataset.sample(
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            output_len=args.custom_output_len,
            request_id_prefix=args.request_id_prefix,
            no_oversample=args.no_oversample,
            min_prompt_len=args.min_prompt_len,
            max_prompt_len=args.max_prompt_len,
            max_total_len=args.max_total_len,
        )
    else:
        # For non-custom datasets (sharegpt, sonnet, etc), try vLLM's implementation
        # This will only work on A100 nodes with new vLLM
        from block.benchmark.cara.vllm_compat import vllm_get_samples, VLLM_BENCHMARKS_AVAILABLE

        if not VLLM_BENCHMARKS_AVAILABLE:
            raise ValueError(
                f"Dataset '{args.dataset_name}' requires vLLM benchmarks module, "
                f"which is not available on this node (old vLLM version). "
                f"Please use --dataset-name custom --dataset-path <path_to_jsonl> instead."
            )

        vllm_samples = vllm_get_samples(args, tokenizer)

        # Convert vLLM's SampleRequest to local SampleRequest for consistency
        return [
            SampleRequest(
                prompt=s.prompt,
                prompt_len=s.prompt_len,
                expected_output_len=s.expected_output_len,
                request_id=getattr(s, 'request_id', ''),
                multi_modal_data=getattr(s, 'multi_modal_data', None),
            )
            for s in vllm_samples
        ]
