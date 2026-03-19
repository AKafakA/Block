"""
Configuration for CARA predictors.

Unlike Block's PredictorConfig (which is for simulation-based approaches),
CARA configs are minimal and type-based for different predictor implementations.
"""
from abc import ABC
from dataclasses import dataclass, field, fields as dataclass_fields
from typing import Dict, Any
import json


@dataclass
class CARABasePredictorConfig(ABC):
    """Base configuration for all CARA predictors."""

    predictor_type: str = field(
        metadata={"help": "Type of predictor: 'dummy' or 'lstm'"}
    )

    # Common settings
    backend_port: int = field(
        default=8000,
        metadata={"help": "Port of vLLM backend instance"}
    )

    schedule_trace_timeout: int = field(
        default=5,
        metadata={"help": "Timeout for /schedule_trace query in seconds"}
    )

    # Data collection settings
    enable_data_collection: bool = field(
        default=False,
        metadata={"help": "Whether to collect training data"}
    )

    data_collection_sample_rate: float = field(
        default=1.0,
        metadata={"help": "Fraction of requests to collect (0.0-1.0)"}
    )

    data_output_dir: str = field(
        default="./training_data/cara",
        metadata={"help": "Directory to save training data"}
    )

    save_batch_size: int = field(
        default=100,
        metadata={"help": "Save collected data every N examples"}
    )

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'CARABasePredictorConfig':
        """Factory method to create appropriate config based on type."""
        predictor_type = config_dict.get("predictor_type", "dummy")

        if predictor_type == "dummy":
            allowed = {f.name for f in dataclass_fields(DummyPredictorConfig)}
            filtered = {k: v for k, v in config_dict.items() if k in allowed}
            return DummyPredictorConfig(**filtered)
        elif predictor_type == "lstm":
            allowed = {f.name for f in dataclass_fields(LSTMPredictorConfig)}
            filtered = {k: v for k, v in config_dict.items() if k in allowed}
            return LSTMPredictorConfig(**filtered)
        elif predictor_type == "learned":
            return LearnedPredictorConfig(raw_config=config_dict)
        else:
            raise ValueError(f"Unknown predictor_type: {predictor_type}")

    @classmethod
    def from_json_file(cls, json_path: str) -> 'CARABasePredictorConfig':
        """Load config from JSON file."""
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


@dataclass
class DummyPredictorConfig(CARABasePredictorConfig):
    """Configuration for dummy predictor (data collection only)."""

    predictor_type: str = "dummy"

    heuristic_mode: str = field(
        default="min_requests",
        metadata={"help": "Heuristic for dummy predictions: 'random', 'min_requests', 'max_gpu_blocks'"}
    )

    # Force data collection for dummy predictor
    enable_data_collection: bool = True


@dataclass
class LSTMPredictorConfig(CARABasePredictorConfig):
    """Configuration for LSTM-based predictor."""

    predictor_type: str = "lstm"

    # Model architecture
    hidden_size: int = field(
        default=128,
        metadata={"help": "LSTM hidden dimension"}
    )

    num_layers: int = field(
        default=2,
        metadata={"help": "Number of LSTM layers"}
    )

    dropout: float = field(
        default=0.1,
        metadata={"help": "Dropout rate"}
    )

    # Model checkpoint
    model_checkpoint_path: str = field(
        default="",
        metadata={"help": "Path to pre-trained model checkpoint"}
    )

    # Inference settings
    use_gpu: bool = field(
        default=False,
        metadata={"help": "Whether to use GPU for inference"}
    )

    # Optional: continue data collection for retraining
    enable_data_collection: bool = field(
        default=False,
        metadata={"help": "Whether to continue collecting data with trained model"}
    )


@dataclass
class LearnedPredictorConfig(CARABasePredictorConfig):
    """Configuration for learned predictor (bucket classifier + XGBoost + KNN quality)."""

    predictor_type: str = "learned"

    # Store full config dict for sub-component configs
    raw_config: Dict[str, Any] = field(default_factory=dict)

    @property
    def bucket_config(self) -> Dict[str, Any]:
        return self.raw_config.get("bucket_classifier", {})

    @property
    def xgboost_config(self) -> Dict[str, Any]:
        return self.raw_config.get("xgboost_ttft", {})

    @property
    def tpot_lookup(self) -> Dict[str, float]:
        return self.raw_config.get("tpot_lookup", {})

    @property
    def quality_config(self) -> Dict[str, Any]:
        return self.raw_config.get("quality_predictor", {})

    @property
    def instance_metadata(self) -> Dict[str, Any]:
        return self.raw_config.get("instance_metadata", {})

    @property
    def scoring_weights(self) -> Dict[str, float]:
        return self.raw_config.get("scoring_weights", {})

    @property
    def slo_defaults(self) -> Dict[str, Any]:
        return self.raw_config.get("slo_defaults", {})
