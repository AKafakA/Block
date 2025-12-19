"""
Centralized dataset path constants for data collection.

These are used by block/data/collect_data.py to avoid scattering
HF paths throughout the code.
"""

# allenai/reward-bench
REWARD_BENCH_BASE = "hf://datasets/allenai/reward-bench/"
REWARD_BENCH_SPLITS = {
    "raw": "data/raw-00000-of-00001.parquet",
    "filtered": "data/filtered-00000-of-00001.parquet",
}
REWARD_BENCH_DEFAULT_PREFIXES = [
    "xstest-",
    "refusals-",
    "donotanswer",
    "hep-",
]

# coseal/CodeUltraFeedback
CODE_ULTRA_FEEDBACK_PATH = (
    "hf://datasets/coseal/CodeUltraFeedback/data/train-00000-of-00001.parquet"
)

# llm-blender/mix-instruct
MIX_INSTRUCT_TRAIN_PATH = (
    "hf://datasets/llm-blender/mix-instruct/train_data_prepared.jsonl"
)

# PKU-Alignment/BeaverTails (30k train)
BEAVER_TAILS_30K_TRAIN_PATH = (
    "hf://datasets/PKU-Alignment/BeaverTails/round0/30k/train.jsonl.gz"
)

