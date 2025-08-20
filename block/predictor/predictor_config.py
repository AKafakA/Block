from abc import ABC
from dataclasses import dataclass, field
from dacite import from_dict

from vidur.config import ReplicaConfig, BaseExecutionTimePredictorConfig, \
    RandomForrestExecutionTimePredictorConfig, VllmSchedulerConfig, SarathiSchedulerConfig


@dataclass
class PredictorConfig(ABC):
    # Configuration for to define a single predictor to predict the completion time of the request.
    # TODO: use the dynamic vllm memory planner instead to get the max batch size.
    #  also consider to export other meta information from block manager to increase the accuracy.
    replica_config: ReplicaConfig = field(
        default_factory=ReplicaConfig,
        metadata={"help": "Configuration to define a model instances, such as model type, and accelerator type."},
    )
    execution_time_predictor_config: BaseExecutionTimePredictorConfig = field(
        default_factory=RandomForrestExecutionTimePredictorConfig,
        metadata={"help": "Execution time predictor config."},
    )
    replica_scheduler_config: VllmSchedulerConfig = field(
        default_factory=VllmSchedulerConfig,
        metadata={"help": "Replica scheduler config."},
    )
    target_metric: str = field(
        default="min_latency",
        metadata={"help": "Target metric to optimize for."},
    )
    enable_batch_time_estimation: bool = field(
        default=True,
        metadata={"help": "Whether to use the tree model to estimate the batch execution time"},
    )
    threshold_batch_size_for_time_estimation: int = field(
        default=36,
        metadata={"help": "Threshold batch size for time estimation."},
    )
    prediction_timeout: int = field(
        default=10,
        metadata={"help": "Timeout for the prediction request."},
    )

    @classmethod
    def create_from_dict(cls, data: dict, enable_chunked_prefill: bool = False):
        chunked_size = data['replica_scheduler_config']['chunk_size']
        del data['replica_scheduler_config']['chunk_size']
        config = from_dict(data_class=cls, data=data)
        if enable_chunked_prefill:
            del data['replica_scheduler_config']['max_tokens_in_batch']
            data['replica_scheduler_config']['chunk_size'] = chunked_size
            config.replica_scheduler_config = from_dict(
                data_class=SarathiSchedulerConfig,
                data=data['replica_scheduler_config'],
            )
        else:
            config.replica_scheduler_config = from_dict(
                data_class=VllmSchedulerConfig,
                data=data['replica_scheduler_config'],
            )
        print(config)
        return config
