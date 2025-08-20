import argparse
import requests


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8200)

    args = parser.parse_args()
    api_url = f"http://{args.host}:{args.port}/generate_benchmark"

    prompt = "Who is the president of the United States?"
    prompt_length = len(prompt.split())
    expected_output = "Joe Biden"
    expected_response_len = len(expected_output.split())

    pload = {
        "prompt": prompt,
        "prompt_len": prompt_length,
        "expected_response_len": expected_response_len,
    }
    response = requests.post(api_url, json=pload)
    print(response.content)