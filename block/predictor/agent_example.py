import argparse
import json
import time
from typing import List

import requests

from vidur.config import FixedRequestLengthGeneratorConfig, PoissonRequestIntervalGeneratorConfig, \
    SyntheticRequestGeneratorConfig
from vidur.entities import Request
from block.server_utils import post_predicting_request, get_predicting_response
from vidur.request_generator.synthetic_request_generator import SyntheticRequestGenerator


def generate_requests():
    length_generator_config = FixedRequestLengthGeneratorConfig(10, 10)
    request_interval_config = PoissonRequestIntervalGeneratorConfig(args.qps)

    request_generator_config = SyntheticRequestGeneratorConfig(num_requests=args.num_request,
                                                               length_generator_config=length_generator_config,
                                                               interval_generator_config=request_interval_config)

    request_generator = SyntheticRequestGenerator(request_generator_config)
    return request_generator.generate_requests()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--num_request", type=int, default=10)
    parser.add_argument("--qps", type=int, default=1)
    parser.add_argument("--use_generated_request", type=bool, default=False)

    args = parser.parse_args()
    predict_api_url = f"http://{args.host}:{args.port}/predict"
    update_api_url = f"http://{args.host}:{args.port}/update"

    # can be replaced by other length config such as uniform, zipfian, trace. Check the config.py for more details
    if args.use_generated_request:
        generated_requests = generate_requests()
    else:
        request1 = Request(0, 100, 70, 35)
        request2 = Request(3, 100, 60, 70)
        generated_requests = [request1, request2]

    for request in generated_requests:
        res = post_predicting_request(predict_api_url,
                                      request_id=request.id,
                                      num_context_tokens=request.num_prefill_tokens,
                                      num_decode_tokens=request.num_decode_tokens, arrived_at=request.arrived_at)
        output = get_predicting_response(res)
        print(output)

        time.sleep(1 / args.qps)

        # send request to the prediction API
        # send request to the update API
        # sleep for the interval time
