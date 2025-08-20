import functools

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import aiohttp
import argparse
import asyncio
import json
import os
import random
import time
import pandas as pd
import numpy as np
import sys
from enum import Enum
from transformers import AutoTokenizer
from typing import List
import resource

resource.setrlimit(resource.RLIMIT_NOFILE, (65536, 65536))

num_finished_requests = 0
server_num_requests = {}
exp_start_time = 0

csv_data_prefill_column_locations = {
    "arxiv": 0,
    "code": 1,
    "burstgpt": 2
}


def fill_missing_metrics(metric, fill_strategy="average_of_neighbors"):
    for i in range(len(metric)):
        if metric[i] is None:
            if fill_strategy == "average_of_neighbors":
                left = i - 1
                right = i + 1
                while left >= 0 and metric[left] is None:
                    left -= 1
                while right < len(metric) and metric[right] is None:
                    right += 1
                if left >= 0 and right < len(metric):
                    metric[i] = (metric[left] + metric[right]) / 2
                elif left >= 0:
                    metric[i] = metric[left]
                elif right < len(metric):
                    metric[i] = metric[right]


def get_wait_time(qps: float, distribution: str, burstiness: float = 1.0) -> float:
    mean_time_between_requests = 1.0 / qps
    if distribution == "uniform":
        return mean_time_between_requests
    elif distribution == "gamma":
        assert burstiness > 0, (
            f"A positive burstiness factor is expected, but given {burstiness}.")
        theta = 1.0 / (qps * burstiness)
        return np.random.gamma(shape=burstiness, scale=theta)
    else:
        return np.random.exponential(mean_time_between_requests)


def request_gen(generator, qps: float, distribution="uniform"):
    while True:
        try:
            item = next(generator)
            yield item
            if distribution != "burst":
                time.sleep(get_wait_time(1.0 / qps, distribution))
        except StopIteration:
            return


async def async_request_gen(generator, qps: float, distribution="uniform", burstiness: float = 0.0):
    while True:
        try:
            item = next(generator)
            yield item
            if distribution != "burst":
                await asyncio.sleep(get_wait_time(qps, distribution, burstiness))
        except StopIteration:
            return


class GenerationBackend(str, Enum):
    vLLM = "vLLM"
    block = "block"
    llumnix = "llumnix"


async def query_model_block(prompt, verbose, ip_ports, timeout_in_seconds):
    prompt, prompt_len, max_response_len, estimated_response_len, request_id = prompt
    global server_num_requests
    global_scheduler_ip_port = ip_ports[0]
    timeout = aiohttp.ClientTimeout(total=timeout_in_seconds)
    global num_finished_requests

    request_dict = {
        "request_id": request_id,
        "prompt": prompt,
        "max_response_len": max_response_len,
        "predicted_response_len": estimated_response_len,
        "prompt_len": prompt_len,
    }

    async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as session:
        if verbose:
            print('Querying model')
        try:
            async with session.post(f'http://{global_scheduler_ip_port}/generate_benchmark', json=request_dict,
                                    ssl=False) as resp:
                if verbose:
                    print('Done')

                output = await resp.json()
                num_finished_requests += 1
                if 'per_token_latency' in output:
                    output['response_len'] = len(output['per_token_latency'])
                elif 'generated_text' in output:
                    output['response_len'] = len(output['generated_text'].split())
                else:
                    output['response_len'] = 0
                return prompt, output
        except asyncio.TimeoutError:
            print(f"Timeout when connecting to {global_scheduler_ip_port}")
            sys.exit(1)
        except aiohttp.ClientError as e:
            print(f"Connect to {global_scheduler_ip_port} failed with: {str(e)}")
            sys.exit(1)


async def query_model_vllm(prompt, verbose, ip_ports, timeout_in_seconds, with_request_id=True):
    prompt, prompt_len, max_response_len, _, request_id = prompt

    # Evenly dispatch request to the given api servers.
    global server_num_requests
    server_id = min(server_num_requests, key=server_num_requests.get)
    server_num_requests[server_id] += 1
    timeout = aiohttp.ClientTimeout(total=timeout_in_seconds)
    global num_finished_requests

    async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as session:
        best_of = 1
        output_len = max_response_len
        request_dict = {
            "prompt": prompt,
            "n": 1,
            "best_of": best_of,
            "temperature": 0.0,
            "max_tokens": output_len,
            "top_k": 1,
            "ignore_eos": True,
            "stream": False,
        }
        if with_request_id:
            request_dict["request_id"] = request_id

        if verbose:
            print('Querying model')
        try:
            async with session.post(f'http://{ip_ports[server_id]}/generate_benchmark', json=request_dict,
                                    ssl=False) as resp:
                if verbose:
                    print('Done')

                output = await resp.json()
                # necessary for latency calc
                if 'per_token_latency' in output:
                    output['response_len'] = len(output['per_token_latency'])
                elif 'generated_text' in output:
                    output['response_len'] = len(output['generated_text'].split())
                else:
                    output['response_len'] = 0
                if verbose and 'generated_text' in output:
                    print(json.dumps(output['generated_text']))
                num_finished_requests += 1
                print("num_finised_requests: {}".format(num_finished_requests))
                return prompt, output
        except aiohttp.ClientError as e:
            print(f"Connect to {ip_ports[server_id]} failed with: {str(e)}")
            sys.exit(1)


def load_prompts(prompt_file):
    with open(prompt_file) as f:
        prompts = [json.loads(l) for l in f.readlines()]
    return prompts


def get_tok_id_lens(tokenizer, batch):
    tokenized = tokenizer.batch_encode_plus(batch)
    lens = [len(s) for s in tokenized['input_ids']]
    return lens


def calculate_throughput(queries,
                         dur_s,
                         backend,
                         tokenizer,
                         mean_token_latency,
                         mean_e2e_latency,
                         all_e2e_latencies,
                         all_per_token_latencies,
                         all_inference_latencies,
                         all_request_ids,
                         all_decode_token_latencies,
                         all_request_lens,
                         all_waiting_latencies,
                         global_scheduling_overhead,
                         fail_on_response_failure):
    # either should be provided
    if not all_waiting_latencies:
        all_waiting_latencies = [-1] * len(all_e2e_latencies)
    if not all_inference_latencies:
        all_inference_latencies = [-1] * len(all_e2e_latencies)
    if not global_scheduling_overhead:
        global_scheduling_overhead = [-1] * len(all_e2e_latencies)

    prompts = []
    responses = []
    naive_hf_lens = []
    ft_lens = []
    expected_response_lens = []
    ray_gen_lens = []
    cf_gen_lens = []
    for prompt, response in queries:
        if 'generated_text' in response:
            prompts.append(prompt)
            responses.append(response['generated_text'])
        if 'naive_hf_lens' in response:
            naive_hf_lens.append(response['naive_hf_lens'])
        if 'ray_gen_len' in response:
            ray_gen_lens.append(response['ray_gen_len'])
        if 'num_output_tokens_cf' in response:
            cf_gen_lens.append(response['num_output_tokens_cf'])
        if 'response_len' in response:
            expected_response_lens.append(response['response_len'])

    # print(f'check_len actual {list(sorted(len(response) for response in response_ids))}')
    # print(f'check_len expect {list(sorted(expected_response_lens))}')
    # print(f'self-reported {list(sorted(cf_gen_lens))}')
    # for prompt, response, expected_response_len in zip(prompt_ids, response_ids, expected_response_lens):
    #     print(f'check lens {len(prompt)=} {len(response)=} {expected_response_len=}')

    try:
        prompt_lens = get_tok_id_lens(tokenizer, prompts)
        response_lens = get_tok_id_lens(tokenizer, responses)
    except Exception:
        print(prompts)
        print(responses)
        raise

    prompt_token_count = sum(prompt_lens)
    response_token_count = sum(response_lens)

    all_prompt_lens = prompt_lens
    all_response_lens = response_lens
    all_total_tokens = [all_prompt_lens[i] + all_response_lens[i] for i in range(len(all_prompt_lens))]
    # if all waiting latencies are not provided, calculate them by e2e - inference
    if not all_waiting_latencies and all_inference_latencies and len(all_inference_latencies) == len(all_e2e_latencies):
        all_waiting_latencies = [all_e2e_latencies[i] - all_inference_latencies[i] for i in
                                 range(len(all_e2e_latencies))]
    elif not all_inference_latencies and all_waiting_latencies and len(all_waiting_latencies) == len(all_e2e_latencies):
        all_inference_latencies = [all_e2e_latencies[i] - all_waiting_latencies[i] for i in
                                   range(len(all_e2e_latencies))]

    def calculate_mean(latencies):
        if latencies:
            return np.mean(latencies)
        else:
            return -1

    mean_waiting_latency = calculate_mean(all_waiting_latencies)
    mean_inference_latency = calculate_mean(all_inference_latencies)
    mean_global_scheduling_overhead = calculate_mean(global_scheduling_overhead)

    if naive_hf_lens:
        # Manually count naive hf tok len
        total_resp_tokens = sum(
            [response_len for _, response_len in naive_hf_lens])
        total_prompt_tokens = sum(
            [prompt_len for prompt_len, _ in naive_hf_lens])
        response_token_count = total_prompt_tokens + total_resp_tokens
    if ray_gen_lens:
        response_token_count = sum(ray_gen_lens)
    if cf_gen_lens:
        response_token_count = sum(cf_gen_lens)

    # print(f'prompt_token_count {prompt_token_count} response_token_count {response_token_count}')
    throughput_tok_s = (prompt_token_count + response_token_count) / dur_s
    print(f'throughput_tok_s {throughput_tok_s:.02f}')
    qps = len(responses) / dur_s
    msg1 = f'backend {backend} dur_s {dur_s:.04f} tokens_per_s {throughput_tok_s:.02f} qps {qps:.04f}\n'
    msg2 = f'successful_responses {len(responses)} prompt_token_count {prompt_token_count} response_token_count {response_token_count}\n'
    msg3 = (f'{mean_token_latency=:.04f}(ms), {mean_e2e_latency=:.04f}(ms), {mean_inference_latency=:.04f}(ms), '
            f'{mean_waiting_latency=:.04f}(ms), {mean_global_scheduling_overhead=:.04f}(ms) \n')

    msg = msg1 + msg2 + msg3

    print(msg)

    if fail_on_response_failure:
        assert len(responses) == len(queries), \
            f"{fail_on_response_failure=}, expected number of successful respones to equal number of queries, got {len(responses)} vs {len(queries)}"
    else:
        error_count = len(queries) - len(responses)
        msg += (
            f"\n error_count {error_count} out of {len(queries)} queries with success rate {error_count / len(queries)}")
    return throughput_tok_s, qps, msg


class MeasureLatency:
    def __init__(self):
        self._request_ids = []
        self._request_lens = []
        self._request_latencies = []
        self._per_token_latencies = []
        self._decode_token_latencies = []
        self._prefill_token_latencies = []
        self._all_token_latencies = []
        self._decode_sum_latencies = []
        self._all_decode_token_latencies = []
        self._inference_latencies = []
        self._waiting_latencies = []
        self._engine_ttft = []
        self._global_scheduling_overhead = []
        self._global_scheduling_overhead_ratio = []
        self._avg_gpu_blocks = []
        self._avg_num_waiting_requests = []
        self._var_gpu_blocks = []
        self._var_num_waiting_requests = []
        self._requested_timestamps = []
        self._num_preempted = []
        self._sampled_mean_error_ratios = []
        self._sampled_predict_accuracies = []
        self._sampled_serving_latencies = []
        self._min_predicted_latency = []
        self._sampled_selected_instance_rank = []
        self._num_available_instances = []

    def measure(self, f):
        async def measured(*args, **kwargs):
            start = time.time()
            prompt, output = await f(*args, **kwargs)
            # Do not record latency if request failed.
            latency = (time.time() - start) * 1000
            if 'generated_text' in output:
                self._request_latencies.append(latency)
                try:
                    self._per_token_latencies.append(
                        latency / output['response_len'])
                except ZeroDivisionError:
                    # Not currently using this metric..
                    pass
            client_ttft = -1.0
            engine_ttft = -1.0
            time_on_backend = -1.0
            if 'request_id' in output:
                self._request_ids.append(output['request_id'])
            if 'per_token_latency' in output:
                lat_arr = np.array(output['per_token_latency'])
                mean_decode_token_latency = 0 if len(lat_arr) == 1 else np.mean(lat_arr[1:, 1])
                decode_sum_latency = 0 if len(lat_arr) == 1 else np.sum(lat_arr[1:, 1])
                self._decode_token_latencies.append(mean_decode_token_latency)
                self._request_lens.append(len(lat_arr[1:, 1]))
                self._all_token_latencies.append(lat_arr)
                self._decode_sum_latencies.append(decode_sum_latency)
                self._all_decode_token_latencies.extend(lat_arr[1:, 1])
                self._prefill_token_latencies.append(lat_arr[0][1])
                if 'time_on_backend' in output:
                    time_on_backend = output['time_on_backend']
                else:
                    start_time_on_backend = lat_arr[0][0] - lat_arr[0][1] / 1000
                    time_on_backend = (lat_arr[-1][0] - start_time_on_backend) * 1000
            if 'per_token_latency_breakdown_dict' in output:
                self._inference_latencies.append(
                    np.mean(output['per_token_latency_breakdown_dict']['step_latency_engine']))
            else:
                if 'inference_latency' in output:
                    self._inference_latencies.append(output['inference_latency'])
            if 'waiting_latency' in output:
                self._waiting_latencies.append(output['waiting_latency'])
            if 'ttft' in output:
                self._engine_ttft.append(output['ttft'])
            if 'sampled_avg_gpu_blocks' in output:
                self._avg_gpu_blocks.append(output['sampled_avg_gpu_blocks'])
                self._var_gpu_blocks.append(output['sampled_var_gpu_blocks'])
            else:
                self._avg_gpu_blocks.append(None)
                self._var_gpu_blocks.append(None)
            if 'sampled_avg_n_request' in output:
                self._avg_num_waiting_requests.append(output['sampled_avg_n_request'])
                self._var_num_waiting_requests.append(output['sampled_var_n_request'])
            else:
                self._avg_num_waiting_requests.append(None)
                self._var_num_waiting_requests.append(None)
            if 'num_preempted' in output:
                self._num_preempted.append(output['num_preempted'])
            else:
                self._num_preempted.append(None)
            if time_on_backend > 0:
                overhead = latency - time_on_backend
            else:
                overhead = None
            if overhead is not None:
                self._global_scheduling_overhead.append(overhead)
                self._global_scheduling_overhead_ratio.append(100.0 * overhead / latency)
            else:
                self._global_scheduling_overhead.append(None)
                self._global_scheduling_overhead_ratio.append(None)
            self._requested_timestamps.append(start)
            if 'sampled_mean_error_ratio' in output:
                self._sampled_mean_error_ratios.append(output['sampled_mean_error_ratio'])
            if 'sampled_predict_accuracy' in output:
                self._sampled_predict_accuracies.append(output['sampled_predict_accuracy'])
            if 'sampled_serving_latencies' in output:
                self._sampled_serving_latencies.append(output['sampled_serving_latencies'])
            if 'min_predicted_latency' in output:
                self._min_predicted_latency.append(output['min_predicted_latency'])
            if 'sampled_selected_instance_rank' in output:
                self._sampled_selected_instance_rank.append(output['sampled_selected_instance_rank'])
            if 'num_available_instances' in output:
                self._num_available_instances.append(output['num_available_instances'])
            return prompt, output

        return measured

    def fill_missing_metrics(self):
        fill_missing_metrics(self._num_preempted)
        fill_missing_metrics(self._avg_gpu_blocks)
        fill_missing_metrics(self._var_gpu_blocks)
        fill_missing_metrics(self._avg_num_waiting_requests)
        fill_missing_metrics(self._var_num_waiting_requests)
        fill_missing_metrics(self._num_preempted)
        fill_missing_metrics(self._global_scheduling_overhead)
        fill_missing_metrics(self._global_scheduling_overhead_ratio)


def get_token_ids(input_str, tokenizer):
    t = tokenizer(input_str)
    return t['input_ids']


async def benchmark(
        backend: GenerationBackend,
        tokenizer,
        prompts: List,
        verbose: bool,
        ip_ports: List[int],
        distribution: str,
        qps: float,
        burstiness: float,
        fail_on_response_failure: bool,
        timeout_in_seconds,
        generate_new_dataset: bool = False,
):
    if backend == GenerationBackend.vLLM:
        query_model = query_model_vllm
    elif backend == GenerationBackend.block:
        query_model = query_model_block
    elif backend == GenerationBackend.llumnix:
        query_model = functools.partial(query_model_vllm, with_request_id=False)
    else:
        raise ValueError(f'unknown backend {backend}')

    global server_num_requests
    num_servers = len(ip_ports)
    for server_id in range(num_servers):
        server_num_requests[server_id] = 0

    m = MeasureLatency()

    query_model = m.measure(query_model)

    if distribution == "burst":
        qps = float('inf')
    start_time = time.time()
    print(
        f'Starting with backend={backend}, num_prompts={len(prompts)}')
    print(f'traffic distribution={distribution}, qps={qps}, burstiness={burstiness} at {time.time()}')

    total_requests = len(prompts)

    async_prompts = async_request_gen(
        iter(prompts), qps=qps, distribution=distribution, burstiness=burstiness)

    tasks = []
    async for prompt in async_prompts:
        # add small extra timeout to avoid conflict with downstream
        tasks.append(
            asyncio.create_task(query_model(prompt, verbose, ip_ports, timeout_in_seconds + 0.001)))
    queries = await asyncio.gather(*tasks)
    print(f'All requested finished at {time.time() - exp_start_time} s, got {len(queries)} responses')
    dur_s = time.time() - start_time
    mean_token_latency = np.mean(m._per_token_latencies)
    mean_e2e_latency = np.mean(m._request_latencies)

    sampled_prompts = []
    sampled_responses = []
    sampled_responses_length = []

    if generate_new_dataset:
        for prompt, output in queries:
            if 'generated_text' in output:
                sampled_prompts.append(prompt)
                sampled_responses.append(output['generated_text'])
        sampled_responses_length = get_tok_id_lens(tokenizer, sampled_responses)

    m.fill_missing_metrics()

    throughput, actual_qps, msg = calculate_throughput(queries,
                                                       dur_s,
                                                       backend,
                                                       tokenizer,
                                                       mean_token_latency,
                                                       mean_e2e_latency,
                                                       m._request_latencies,
                                                       m._per_token_latencies,
                                                       m._inference_latencies,
                                                       m._request_ids,
                                                       m._decode_token_latencies,
                                                       m._request_lens,
                                                       m._waiting_latencies,
                                                       m._global_scheduling_overhead,
                                                       fail_on_response_failure)
    timestamps = [int((x - start_time)) for x in m._requested_timestamps]
    avg_instance_num = 0.0
    return throughput, \
        actual_qps, \
        m._prefill_token_latencies, \
        m._decode_token_latencies, \
        m._inference_latencies, \
        avg_instance_num, \
        m._request_latencies, \
        m._request_ids, \
        m._decode_sum_latencies, \
        m._request_lens, \
        m._all_decode_token_latencies, \
        m._waiting_latencies, \
        m._global_scheduling_overhead, \
        sampled_prompts, \
        sampled_responses, \
        sampled_responses_length, \
        m._avg_gpu_blocks, \
        m._var_gpu_blocks, \
        m._avg_num_waiting_requests, \
        m._var_num_waiting_requests, \
        m._num_preempted, \
        m._sampled_mean_error_ratios, \
        m._sampled_predict_accuracies, \
        m._sampled_serving_latencies, \
        m._min_predicted_latency, \
        m._sampled_selected_instance_rank, \
        m._num_available_instances, \
        timestamps, \
        msg


def get_dataset_list(dataset_path: str, start_idx: int = 0, num_samples: int = 10):
    print(f"Loading dataset from {dataset_path} with start_idx {start_idx} and num_samples {num_samples}")
    dataset_list = []
    for file in os.listdir(dataset_path):
        path = os.path.join(dataset_path, file)
        if path.endswith('.jsonl'):
            with open(path) as f:
                for line in f:
                    dataset_list.append(json.loads(line))
        elif path.endswith('.json'):
            with open(path) as f:
                dataset_list.extend(json.load(f))
        elif path.endswith('.parquet'):
            dataset_list.extend(pd.read_parquet(path).to_dict(orient='records'))
        elif path.endswith('.csv'):
            raw_data_list = pd.read_csv(path).values.tolist()
            dataset_list.extend(raw_data_list)
    end_idx = min(len(dataset_list), start_idx + num_samples)
    if start_idx >= len(dataset_list):
        raise ValueError(f"start_idx {start_idx} is out of range for dataset with {len(dataset_list)} samples.")
    return dataset_list[start_idx:end_idx]


def sample_requests(
        dataset_path: str,
        num_requests: int,
        tokenizer,
        max_seqlen: int,
        use_estimated_response_lens: bool,
        start_idx: int,
        task: str = 'chat'
):
    prompts = []
    prompt_lens = []
    max_response_lens = []
    estimated_response_lens = []

    # Load the dataset.
    dataset = get_dataset_list(dataset_path, start_idx, num_requests)

    # Filter dataset for conversation mode
    if task == 'chat':
        dataset = [
            data for data in dataset
            if (
                    ("conversations" in data and len(data["conversations"]) >= 2) or
                    ("conversation" in data and len(data["conversation"]) >= 2)
            )
        ]
    vocab_size = tokenizer.vocab_size
    for i in range(len(dataset)):
        data = dataset[i]
        if task == 'chat':
            if "conversations" in data:
                prompt = data["conversations"][0]["value"]
                res = data["conversations"][1]["value"]
            elif "conversation" in data:
                prompt = data["conversation"][0]["content"]
                res = data["conversation"][1]["content"]
            else:
                raise ValueError(f"Unknown dataset format: {data.keys()}")
        elif task in csv_data_prefill_column_locations.keys():
            input_len_loc = csv_data_prefill_column_locations[task]
            input_len = int(data[input_len_loc])
            output_len = int(data[input_len_loc + 1])
            prompt_ids = [(i + j) % vocab_size for j in range(input_len)]
            prompt = tokenizer.decode(prompt_ids)
            res = "a" * output_len
        else:
            raise ValueError(f"Unknown task {task}")

        prompt_token_ids = tokenizer(prompt).input_ids
        completion_token_ids = tokenizer(res).input_ids

        prompt_len = len(prompt_token_ids)
        completion_len = len(completion_token_ids)

        if (prompt_len > 0 and completion_len > 0
                and max_seqlen > prompt_len + completion_len):
            prompts.append(prompt)
            prompt_lens.append(prompt_len)
            max_response_lens.append(completion_len)
            estimated_response_lens.append(
                min(max_seqlen - prompt_len - 1, max(1, int(data.get("predicted_length", completion_len))))
                if use_estimated_response_lens else completion_len
            )

        if len(prompts) > num_requests:
            break

    sampled_ids = random.sample(range(len(prompts)), min(num_requests, len(prompts)))
    sampled_prompts = [prompts[idx] for idx in sampled_ids]
    sampled_prompt_lens = [prompt_lens[idx] for idx in sampled_ids]
    sampled_response_lens = [max_response_lens[idx] for idx in sampled_ids]
    sampled_estimated_response_lens = [estimated_response_lens[idx] for idx in sampled_ids]

    return sampled_prompts, sampled_prompt_lens, sampled_response_lens, sampled_estimated_response_lens


def generate_lens_files(
        length_output_file,
        prompt_lens,
        response_lens):
    import csv
    assert length_output_file.endswith('.csv')
    with open(length_output_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['num_prefill_tokens', 'num_decode_tokens', 'num_total_tokens', 'pd_ratio'])
        for prompt_len, response_len in zip(prompt_lens, response_lens):
            writer.writerow([prompt_len, response_len, prompt_len + response_len, (response_len * 1.0) / prompt_len])
    print(f"CSV files for length information saved to {length_output_file}")


def generate_dataset_with_real_response(
        start_id: int,
        prompts,
        responses,
        new_dataset_path: str):
    data = []
    record_id = start_id
    filtered_count = 0
    for prompt, response in zip(prompts, responses):
        if response.replace(' ', ''):
            record = {'id': record_id,
                      'conversations': [{'from': 'human', 'value': prompt}, {'from': 'model', 'value': response}]}
            data.append(record)
            record_id += 1
        else:
            filtered_count += 1
    with open(new_dataset_path, 'w') as fp:
        json.dump(data, fp)
    print(f"Dataset with real responses saved to {new_dataset_path} and tagged with {len(data)} records and "
          f" filtered out {filtered_count} empty requests.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str, required=True,
                        help="Name or path of the tokenizer.")
    parser.add_argument('--trust_remote_code',
                        action='store_true')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--backend', type=GenerationBackend,
                        choices=[e.name for e in GenerationBackend], default='block')
    parser.add_argument('--log_filename', type=str, default='benchmark.log')
    parser.add_argument('--ip_ports', nargs='+', required=True, help='List of ip:port')
    parser.add_argument('--num_sampled_requests', type=int, default=10000)
    parser.add_argument('--data_start_index', type=int, default=0,
                        help="Start index of the dataset to sample from.")
    parser.add_argument('--max_request_len', type=int, default=4096)
    parser.add_argument(
        '--distribution', choices=["uniform", "gamma", "exponential"], default="gamma")
    parser.add_argument('--qps', type=float, default=4.0)
    parser.add_argument('--burstiness', type=float, default=1.0)
    parser.add_argument('--fail_on_response_failure', type=bool, default=False,
                        help="Whether or not to fail the benchmarking script if any request fails")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--dataset_type', type=str,
                       choices=['sharegpt', 'arxiv', 'lmsys', 'burstgpt', 'code'], default='sharegpt')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--dataset_path', type=str)
    parser.add_argument('--generate_dataset_with_real_response',
                        action='store_true')
    parser.add_argument('--generate_csv_files', action='store_true')
    parser.add_argument("--output_dir", type=str, default="benchmark_output")
    parser.add_argument("--use_estimated_response_lens", action='store_true')
    parser.add_argument("--timeout_in_seconds", type=int, default=1800)

    args = parser.parse_args()

    print(f"running experiment for dataset {args.dataset_type} with backend {args.backend} and qps {args.qps} "
          f"and use_estimated_response_lens {args.use_estimated_response_lens} ")

    args.output_dir = "experiment_output/" + args.output_dir
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    backend = GenerationBackend[args.backend]
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=args.trust_remote_code)

    random.seed(0xCADE)
    np.random.seed(0xCADE)
    if args.dataset_type == "sharegpt" or args.dataset_type == "lmsys":
        prompts, prompt_lens, max_response_lens, estimated_response_lens = sample_requests(
            args.dataset_path,
            args.num_sampled_requests,
            tokenizer,
            args.max_request_len,
            args.use_estimated_response_lens,
            args.data_start_index,
            task='chat'
        )
    elif args.dataset_type == "arxiv" or args.dataset_type == "burstgpt" or args.dataset_type == "code":
        prompts, prompt_lens, max_response_lens, estimated_response_lens = sample_requests(
            args.dataset_path,
            args.num_sampled_requests,
            tokenizer,
            args.max_request_len,
            args.use_estimated_response_lens,
            args.data_start_index,
            task=args.dataset_type
        )
    else:
        raise ValueError(f"Unknown dataset type {args.dataset_type}")

    for i, (prompt_len, gen_len) in enumerate(zip(prompt_lens, max_response_lens)):
        total = prompt_len + gen_len
        if total > args.max_request_len:
            print(f'truncating long prompt+gen_len {prompt_len=} {gen_len=}')
            gen_len = args.max_request_len - prompt_len
        max_response_lens[i] = gen_len

    prompts = list(zip(prompts, prompt_lens, max_response_lens, estimated_response_lens, range(len(prompt_lens))))

    print(f"sampled {len(prompts)} after filtering by length")

    (throughput,
     actual_qps,
     prefill_token_latencies,
     decode_token_latencies,
     inference_latencies,
     avg_instance_num,
     request_latencies, request_ids,
     decode_sum_latencies, request_lens,
     all_decode_token_latencies,
     waiting_latency,
     scheduling_overhead,
     sampled_prompts,
     sampled_responses,
     sampled_responses_length,
     avg_gpu_blocks, var_gpu_blocks,
     avg_num_waiting_requests,
     var_num_waiting_requests,
     num_preempted,
     sampled_mean_error_ratios,
     sampled_predict_accuracies,
     sampled_serving_latencies,
     sampled_predict_latency,
     sampled_selected_instance_rank,
     num_available_instances,
     request_timestamps,
     messages) = asyncio.run(benchmark(
        backend,
        tokenizer,
        prompts,
        args.verbose,
        args.ip_ports,
        args.distribution,
        args.qps,
        args.burstiness,
        args.fail_on_response_failure,
        args.timeout_in_seconds,
        generate_new_dataset=args.generate_dataset_with_real_response or args.generate_csv_files
    )
    )
    print(f'Experiment finished with throughput={throughput:.2f} tokens/s, at time={time.time() - exp_start_time:.2f} s'
          f'')

    with open(args.output_dir + '/' + os.path.splitext(args.log_filename)[0] + "_logs.txt", 'w') as f:
        f.write(messages)
        ttft = np.array(prefill_token_latencies)
        p99_ttft = np.percentile(ttft, 99)
        request_latencies_arr = np.array(request_latencies)
        p99_request_latency = np.percentile(request_latencies_arr, 99)
        f.write(f"\n p99 prefill token latency: {p99_ttft:.4f} ms\n")
        f.write(f"\n p99 request latency: {p99_request_latency:.4f} ms\n")

    data = {
        "Throughput": np.float32(throughput),
        "prefill_token_latencies": np.array(prefill_token_latencies),
        "decode_sum_latencies": np.array(decode_sum_latencies),
        "request_latencies": np.array(request_latencies),
        "scheduling_overhead": np.array(scheduling_overhead),
        "actual_qps": np.float32(actual_qps),
        "avg_gpu_blocks": np.array(avg_gpu_blocks),
        "var_gpu_blocks": np.array(var_gpu_blocks),
        "num_preempted": np.array(num_preempted)
    }
    # sampled data to do the profiling
    if sampled_predict_accuracies:
        data["sampled_predict_accuracies"] = np.array(sampled_predict_accuracies)
    if sampled_mean_error_ratios:
        data["sampled_mean_error_ratios"] = np.array(sampled_mean_error_ratios)
    if sampled_serving_latencies:
        data["sampled_serving_latencies"] = np.array(sampled_serving_latencies)
    if sampled_predict_latency:
        data["sampled_predict_latency"] = np.array(sampled_predict_latency)
    if sampled_selected_instance_rank:
        data["sampled_selected_instance_rank"] = np.array(sampled_selected_instance_rank)
    if num_available_instances:
        data["num_available_instances"] = np.array(num_available_instances)
    np.savez(args.output_dir + '/' + os.path.splitext(args.log_filename)[0] + f"_all_metrics.npz", **data)

    if args.generate_csv_files or args.generate_dataset_with_real_response:
        generated_file_dir = args.dataset_path + "/" + "generate"
        if not os.path.exists(generated_file_dir):
            os.makedirs(generated_file_dir)

        if args.generate_dataset_with_real_response:
            existing_generated_dataset_files = [file for file in os.listdir(generated_file_dir)
                                                if file.endswith('with_real_response.json')]
            new_generated_dataset_path = os.path.join(generated_file_dir,
                                                      f'{args.dataset_type}_{args.num_sampled_requests}_'
                                                      f'{len(existing_generated_dataset_files) + 1}'
                                                      f'_with_real_response.json')
            generate_dataset_with_real_response(
                args.data_start_index, sampled_prompts, sampled_responses, new_generated_dataset_path)

        if args.generate_csv_files:
            generated_csv_files = [file for file in os.listdir(generated_file_dir)
                                   if file.endswith('lens.csv')]
            csv_file_name = os.path.join(generated_file_dir,
                                         f'{args.dataset_type}_{args.num_sampled_requests}_'
                                         f'{len(generated_csv_files) + 1}'
                                         f'_lens.csv')
            generate_lens_files(csv_file_name, prompt_lens, sampled_responses_length)


if __name__ == '__main__':
    exp_start_time = time.time()
    main()
    print(f'Experiment finished at {time.time() - exp_start_time} s')
