#!/usr/bin/env python3
"""
Deterministic 80/20 train/test split for CARA training data.

Uses hash-based splitting on request_id for reproducibility across reruns.
The same split is used for all predictor types (length, quality).
"""

import argparse
import hashlib
import json
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def hash_split(request_id: str, train_ratio: float = 0.8) -> bool:
    """Deterministic split using MD5 hash of request_id.

    Returns True if request should go to train set.
    """
    h = hashlib.md5(str(request_id).encode()).hexdigest()
    # Use first 8 hex chars (32 bits) for uniform distribution
    return (int(h[:8], 16) / 0xFFFFFFFF) < train_ratio


def main():
    parser = argparse.ArgumentParser(
        description="Split CARA training data into train/test sets"
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to processed training data JSON"
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.8,
        help="Fraction of data for training (default: 0.8)"
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory (default: same as input file)"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    with open(input_path) as f:
        data = json.load(f)

    requests = data["requests"]
    logger.info(f"Loaded {len(requests)} requests from {input_path}")

    # Split
    train_requests = []
    test_requests = []
    for req in requests:
        if hash_split(req["request_id"], args.train_ratio):
            train_requests.append(req)
        else:
            test_requests.append(req)

    logger.info(
        f"Split: {len(train_requests)} train ({len(train_requests)/len(requests)*100:.1f}%), "
        f"{len(test_requests)} test ({len(test_requests)/len(requests)*100:.1f}%)"
    )

    # Save
    stem = input_path.stem
    train_path = output_dir / f"{stem}_train.json"
    test_path = output_dir / f"{stem}_test.json"

    # Preserve metadata from original file
    metadata = {k: v for k, v in data.items() if k != "requests"}

    for path, reqs, label in [
        (train_path, train_requests, "train"),
        (test_path, test_requests, "test")
    ]:
        output = {**metadata, "requests": reqs, "split": label}
        with open(path, 'w') as f:
            json.dump(output, f, indent=2)
        logger.info(f"Saved {label}: {path} ({len(reqs)} requests)")


if __name__ == "__main__":
    main()
