# -----------------------------------------------------------------------------
# Lmsys Dataset Implementation
# -----------------------------------------------------------------------------
import os
import random

import pandas as pd
from vllm.benchmarks.datasets import BenchmarkDataset, is_valid_sequence, SampleRequest, CustomDataset
from transformers import PreTrainedTokenizerBase

# -----------------------------------------------------------------------------
# ShareGPT Dataset Implementation
# -----------------------------------------------------------------------------

class LmsysDataset(BenchmarkDataset):
    """
    Implements the LMSYS dataset.  Loads data from a parquet file and generates
    sample requests based on conversation turns.
    """

    def __init__(self, **kwargs) -> None:
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
def get_samples(args, tokenizer) -> list[SampleRequest]:
    if args.dataset_name == "custom" and args.dataset_path.endswith("lmsys"):
        dataset = LmsysDataset(
            dataset_path=args.dataset_path,
            random_seed=args.seed
        )
        input_requests = dataset.sample(
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            output_len=args.custom_output_len,
            skip_chat_template=args.skip_chat_template,
            request_id_prefix=args.request_id_prefix,
            no_oversample=args.no_oversample,
            max_total_len=args.max_total_len,
        )
        return input_requests
    else:
        from vllm.benchmarks.datasets import get_samples
        return get_samples(args, tokenizer)

