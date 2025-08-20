import argparse
import pandas as pd


def convert_trace(traces):
    converted_traces = []
    for trace in traces:
        num_prefill_tokens = trace['num_prefill_tokens']
        num_decode_tokens = trace['num_decode_tokens']
        num_total_tokens = num_decode_tokens + num_prefill_tokens
        pd_ratio = num_prefill_tokens / num_total_tokens
        converted_trace = {
            'num_prefill_tokens': num_prefill_tokens,
            'num_decode_tokens': num_decode_tokens,
            'num_total_tokens': num_total_tokens,
            'pd_ratio': pd_ratio
        }
        converted_traces.append(converted_trace)
    return converted_traces


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, default=None)
    parser.add_argument("--output-file", type=str, default=None)

    args = parser.parse_args()
    df = pd.read_csv(args.input_file)
    traces = df.to_dict(orient='records')

    converted_traces = convert_trace(traces)
    converted_df = pd.DataFrame(converted_traces)
    converted_df.to_csv(args.output_file, index=False)


if __name__ == "__main__":
    main()
