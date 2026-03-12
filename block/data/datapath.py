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

# Additional general-purpose instruction/chat datasets
# These loaders use the Hugging Face Datasets library via dataset names
# to avoid depending on fixed file layouts. They are referenced by name
# in collect_data.py, not by file path.

# HuggingFaceH4/ultrachat_200k: multi-turn benign chat dataset
ULTRACHAT_DATASET_NAME = "HuggingFaceH4/ultrachat_200k"

# google-research-datasets/gsm8k: grade-school math word problems
GSM8K_DATASET_NAME = "gsm8k"

# Widely adopted QA/summarization/MC benchmarks
SQUAD_DATASET_NAME = "squad"
CNN_DAILYMAIL_DATASET_NAME = "cnn_dailymail"
AI2_ARC_DATASET_NAME = "ai2_arc"

# lmsys/lmsys-chat-1m: 1M real chat conversations with language tags and response labels
LMSYS_DATASET_NAME = "lmsys/lmsys-chat-1m"
LMSYS_PARQUET_FILES = [
    f"data/train-{i:05d}-of-00006-*.parquet" for i in range(6)
]
