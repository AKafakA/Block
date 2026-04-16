#!/bin/bash
# Upload CARA training data + scripts to HuggingFace
# Run from GPU VM (wd312@gxp-l40s-2) after judge validation + README update

set -e

cd ~/Code/llm/Block
source .venv/bin/activate

HF_TOKEN="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"
REPO_ID="asdwb/cara_model_estimator"
DATA_DIR="data/cara/training_data"
SCRIPTS_DIR="block/predictor/cara/offline_training"

echo "=== Uploading CARA Training Data to HuggingFace ==="
echo "Repo: ${REPO_ID}"
echo ""

python3 << 'PYEOF'
from huggingface_hub import HfApi
import os

api = HfApi(token=os.environ.get("HF_TOKEN", "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"))
REPO_ID = "asdwb/cara_model_estimator"
DATA_DIR = "data/cara/training_data"
SCRIPTS_DIR = "block/predictor/cara/offline_training"

# Ensure repo exists
try:
    api.create_repo(repo_id=REPO_ID, repo_type="dataset", exist_ok=True)
except Exception as e:
    print(f"Repo check: {e}")

# --- Data files ---
data_files = [
    ("cara_v3_all_training.json", "cara_v3_all_training.json"),
    ("cara_v3_all_training_train.jsonl", "train.jsonl"),
    ("cara_v3_all_training_test.jsonl", "test.jsonl"),
    ("judge_validation_fair_comparison.json", "judge_validation_fair_comparison.json"),
]

for fname, repo_path in data_files:
    fpath = os.path.join(DATA_DIR, fname)
    if os.path.exists(fpath):
        size_mb = os.path.getsize(fpath) / 1e6
        print(f"Uploading {fname} ({size_mb:.1f} MB) -> {repo_path}")
        api.upload_file(
            path_or_fileobj=fpath,
            path_in_repo=repo_path,
            repo_id=REPO_ID,
            repo_type="dataset",
        )
    else:
        print(f"  Skipped (not found): {fpath}")

# --- Public README (dataset card) ---
readme_path = os.path.join(DATA_DIR, "README_HF.md")
if os.path.exists(readme_path):
    print(f"Uploading README_HF.md -> README.md")
    api.upload_file(
        path_or_fileobj=readme_path,
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type="dataset",
    )

# --- Scripts from offline_training ---
scripts = [
    "prepare_benchmark_data.py",
    "llm_judge_scorer.py",
    "validate_judge_models.py",
    "split_training_data.py",
    "response_filter.py",
]

for fname in scripts:
    fpath = os.path.join(SCRIPTS_DIR, fname)
    if os.path.exists(fpath):
        size_kb = os.path.getsize(fpath) / 1e3
        print(f"Uploading {fname} ({size_kb:.1f} KB) -> scripts/{fname}")
        api.upload_file(
            path_or_fileobj=fpath,
            path_in_repo=f"scripts/{fname}",
            repo_id=REPO_ID,
            repo_type="dataset",
        )
    else:
        print(f"  Skipped (not found): {fpath}")

# --- collect_data.py from block/data ---
collect_data_path = "block/data/collect_data.py"
if os.path.exists(collect_data_path):
    size_kb = os.path.getsize(collect_data_path) / 1e3
    print(f"Uploading collect_data.py ({size_kb:.1f} KB) -> scripts/collect_data.py")
    api.upload_file(
        path_or_fileobj=collect_data_path,
        path_in_repo="scripts/collect_data.py",
        repo_id=REPO_ID,
        repo_type="dataset",
    )
else:
    print(f"  Skipped (not found): {collect_data_path}")

print()
print("Upload complete!")
print(f"View at: https://huggingface.co/datasets/{REPO_ID}")
PYEOF
