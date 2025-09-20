import json
import logging
import os
from typing import List, Sequence

import pandas as pd

from vidur.config import TraceRequestGeneratorConfig
from vidur.entities import Request
from vidur.request_generator.base_request_generator import BaseRequestGenerator

logger = logging.getLogger(__name__)


class TraceReplayRequestGenerator(BaseRequestGenerator):
    """
    Reads a trace csv file containing request arrival time, its prompt and completion token values to generate
    inter-request times, number of tokens.
    """

    def __init__(self, config: TraceRequestGeneratorConfig):
        super().__init__(config)

        # load into a pd dataframe
        self.trace_df = pd.read_csv(config.trace_file)

        if config.max_requests and config.max_requests > 0:
            self.trace_df = self.trace_df.head(config.max_requests)

        if "arrived_at" not in self.trace_df.columns:
            self.trace_df["arrived_at"] = self.trace_df.index.astype(float)

        self._use_predicted_decode_tokens = config.use_predicted_decode_tokens
        self._predicted_decode_tokens = None

        # scale prefill and decode tokens
        self.trace_df["num_prefill_tokens"] = (
            self.trace_df["num_prefill_tokens"] * config.prefill_scale_factor
        )
        self.trace_df["num_decode_tokens"] = (
            self.trace_df["num_decode_tokens"] * config.decode_scale_factor
        )

        # make sure all the prefill and decode counts are integers
        self.trace_df["num_prefill_tokens"] = self.trace_df[
            "num_prefill_tokens"
        ].astype(int)
        self.trace_df["num_decode_tokens"] = self.trace_df["num_decode_tokens"].astype(
            int
        )

        # make sure that there is at least one prefill and decode token
        self.trace_df["num_prefill_tokens"] = self.trace_df["num_prefill_tokens"].clip(
            lower=1
        )
        self.trace_df["num_decode_tokens"] = self.trace_df["num_decode_tokens"].clip(
            lower=1
        )

        # make sure the total does not exceed the max tokens, adjust the prefill tokens if needed
        total_tokens = (
            self.trace_df["num_prefill_tokens"] + self.trace_df["num_decode_tokens"]
        )
        diff_tokens = total_tokens - config.max_tokens
        diff_tokens = diff_tokens.clip(lower=0)
        self.trace_df["num_prefill_tokens"] = (
            self.trace_df["num_prefill_tokens"] - diff_tokens
        )

        assert all(
            self.trace_df["num_prefill_tokens"] + self.trace_df["num_decode_tokens"]
            <= config.max_tokens
        )

        # rescale the time to change QPS
        self.trace_df["arrived_at"] = (
            self.trace_df["arrived_at"] * config.time_scale_factor
        )

        if self._use_predicted_decode_tokens:
            predicted_series = self._load_predicted_decode_tokens(
                config.predicted_trace_file,
                config.predicted_decode_column,
                len(self.trace_df),
            )
            self._predicted_decode_tokens = (
                predicted_series * config.decode_scale_factor
            ).astype(int).clip(lower=1)
            self.trace_df["predicted_num_decode_tokens"] = self._predicted_decode_tokens
        else:
            self._predicted_decode_tokens = self.trace_df["num_decode_tokens"].values

        logger.info(
            f"Loaded trace file {config.trace_file} with {len(self.trace_df)} requests"
        )
        # compute pd ratio and log the 25, 50, 75, 90, 95, 99 percentiles
        pd_ratio = (
            self.trace_df["num_prefill_tokens"] / self.trace_df["num_decode_tokens"]
        )
        logger.debug(
            f"Prompt/decode token ratio stats\n:{pd_ratio.describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99])}"
        )

    def generate_requests(self) -> List[Request]:
        requests = []

        for idx, (_, row) in enumerate(self.trace_df.iterrows()):
            request = Request(
                arrived_at=row["arrived_at"],
                num_prefill_tokens=row["num_prefill_tokens"],
                num_decode_tokens=row["num_decode_tokens"],
                num_predicted_decode_tokens=int(self._predicted_decode_tokens[idx]),
            )

            requests.append(request)

        return requests

    def _load_predicted_decode_tokens(
        self,
        predicted_trace_file: str,
        predicted_column: str,
        expected_length: int,
    ) -> pd.Series:
        """Load predicted decode tokens either from the CSV column or external file."""

        if predicted_column in self.trace_df.columns:
            series = self.trace_df[predicted_column]
            if len(series) != expected_length:
                raise ValueError(
                    "Predicted decode token column length does not match trace length"
                )
            return series

        if not predicted_trace_file:
            raise ValueError(
                "use_predicted_decode_tokens is True but no predicted_trace_file provided and"
                f" column '{predicted_column}' not found in trace"
            )

        if not os.path.exists(predicted_trace_file):
            raise FileNotFoundError(
                f"Predicted trace file {predicted_trace_file} not found"
            )

        extension = os.path.splitext(predicted_trace_file)[1].lower()

        if extension in (".json", ".jsonl"):
            predictions = self._load_predictions_from_json(
                predicted_trace_file, extension == ".jsonl"
            )
        else:
            predictions = self._load_predictions_from_tabular(
                predicted_trace_file, predicted_column
            )

        if len(predictions) < expected_length:
            raise ValueError(
                "Predicted decode token list shorter than trace length"
            )

        if len(predictions) > expected_length:
            predictions = predictions[:expected_length]

        return pd.Series(predictions, dtype=float)

    @staticmethod
    def _load_predictions_from_json(file_path: str, is_jsonl: bool) -> Sequence[int]:
        predictions = []
        if is_jsonl:
            with open(file_path, "r") as f:
                for line in f:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    predictions.append(int(record.get("predicted_length", 0)))
        else:
            with open(file_path, "r") as f:
                data = json.load(f)
            for record in data:
                predictions.append(int(record.get("predicted_length", 0)))
        return predictions

    @staticmethod
    def _load_predictions_from_tabular(
        file_path: str, predicted_column: str
    ) -> Sequence[int]:
        df = pd.read_csv(file_path)
        if predicted_column not in df.columns:
            raise ValueError(
                f"Predicted decode token column '{predicted_column}' not found in {file_path}"
            )
        return df[predicted_column].astype(int).tolist()
