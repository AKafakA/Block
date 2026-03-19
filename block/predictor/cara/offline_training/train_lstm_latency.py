#!/usr/bin/env python3
"""
Train LSTM latency predictor for CARA (ablation).

Uses sliding windows of schedule state to predict E2E latency.
Evaluates whether temporal information improves over point-in-time XGBoost.

Usage:
    python -m block.predictor.cara.offline_training.train_lstm_latency \
        --data-dir data/cara/latency_training/ \
        --output-dir models/cara/lstm_latency/ \
        --window-size 10 --epochs 50
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from block.predictor.cara.estimators.lstm_predictor import (
    LSTMLatencyModel,
    LSTMLatencyPredictor,
)
from block.predictor.cara.estimators.xgboost_predictor import ALL_FEATURES, build_feature_vector
from block.predictor.cara.offline_training.train_xgboost import load_latency_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def build_sliding_windows(
    records: list,
    window_size: int,
    target: str = "e2el",
) -> Tuple[np.ndarray, np.ndarray]:
    """Build sliding window sequences from time-ordered latency records.

    Returns:
        X: (N, window_size, num_features)
        y: (N,) target latency values
    """
    # Sort by timestamp
    sorted_records = sorted(records, key=lambda r: r.get("timestamp", 0))

    # Build feature vectors for all records
    all_features = []
    all_targets = []
    for rec in sorted_records:
        target_val = rec.get(target) or rec.get("actual_e2e_latency")
        if target_val is None or target_val <= 0:
            continue

        schedule_state = rec.get("schedule_state", {})
        if not schedule_state:
            schedule_state = {k: rec.get(k, 0) for k in [
                "ema_decode_tok_per_s", "ema_prefill_tok_per_s", "ema_decode_iter_ms",
                "decode_ctx_p50", "decode_ctx_p95", "decode_ctx_max",
                "num_running", "num_active_decode_seqs", "num_waiting",
                "pending_prefill_tokens", "pending_decode_tokens",
                "token_budget_per_iter", "prefill_chunk_size", "max_num_seqs",
                "kv_cache_utilization", "kv_free_blocks", "kv_evictions_per_s",
            ]}

        num_prompt = rec.get("num_prompt_tokens") or rec.get("input_len", 0)
        num_output = rec.get("num_predicted_output_tokens") or rec.get("max_tokens") or rec.get("output_len", 0)

        fv = build_feature_vector(schedule_state, int(num_prompt), int(num_output))
        all_features.append(fv)
        all_targets.append(float(target_val))

    if len(all_features) < window_size:
        return np.empty((0, window_size, len(ALL_FEATURES))), np.empty(0)

    all_features = np.stack(all_features)
    all_targets = np.array(all_targets, dtype=np.float32)

    # Create sliding windows
    X_windows = []
    y_windows = []
    for i in range(window_size, len(all_features)):
        X_windows.append(all_features[i - window_size : i])
        y_windows.append(all_targets[i - 1])  # Predict latency for last request in window

    return np.stack(X_windows), np.array(y_windows)


def train_lstm_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    hidden_dim: int = 128,
    num_layers: int = 2,
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 64,
    device: str = "cpu",
) -> Tuple[LSTMLatencyModel, Dict]:
    """Train a single LSTM latency model."""
    input_dim = X_train.shape[2]
    model = LSTMLatencyModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    ).to(device)

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(X_train_t), device=X_train_t.device)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, len(X_train_t), batch_size):
            idx = perm[i : i + batch_size]
            pred = model(X_train_t[idx])
            loss = loss_fn(pred, y_train_t[idx])

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = loss_fn(val_pred, y_val_t).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            logger.info(
                f"  Epoch {epoch+1}/{epochs}: "
                f"train_loss={epoch_loss/n_batches:.6f}, val_loss={val_loss:.6f}"
            )

    if best_state:
        model.load_state_dict(best_state)

    # Final evaluation
    model.eval()
    with torch.no_grad():
        preds = model(X_val_t).cpu().numpy()

    errors = np.abs(y_val - preds)
    metrics = {
        "mae": float(np.mean(errors)),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "mape": float(np.mean(errors / np.maximum(y_val, 1e-6)) * 100),
        "p50_error": float(np.percentile(errors, 50)),
        "p95_error": float(np.percentile(errors, 95)),
        "n_train": len(X_train),
        "n_val": len(X_val),
    }

    return model, metrics


def main():
    parser = argparse.ArgumentParser(
        description="Train LSTM latency predictor for CARA"
    )
    parser.add_argument("--data-dir", required=True, help="Latency data directory")
    parser.add_argument("--instance-types", nargs="+", default=None)
    parser.add_argument("--target", default="e2el")
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--output-dir", default="models/cara/lstm_latency")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    records_by_type = load_latency_data(args.data_dir, args.instance_types)
    if not records_by_type:
        logger.error("No latency data found!")
        return

    predictor = LSTMLatencyPredictor(
        window_size=args.window_size,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        device=args.device,
    )
    all_metrics = {}

    for inst_type, records in sorted(records_by_type.items()):
        logger.info(f"\nTraining LSTM for {inst_type} ({len(records)} records)")

        X, y = build_sliding_windows(records, args.window_size, args.target)
        if len(X) < 50:
            logger.warning(f"Too few windows for {inst_type} ({len(X)}), skipping")
            continue

        predictor.input_dim = X.shape[2]

        # Time-based split (last 20% as val, preserving temporal order)
        n_val = int(len(X) * args.val_split)
        X_train, y_train = X[:-n_val], y[:-n_val]
        X_val, y_val = X[-n_val:], y[-n_val:]

        t0 = time.time()
        model, metrics = train_lstm_model(
            X_train, y_train, X_val, y_val,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            device=args.device,
        )
        metrics["train_time_s"] = time.time() - t0

        predictor.models[inst_type] = model
        all_metrics[inst_type] = metrics

        logger.info(
            f"  {inst_type}: MAE={metrics['mae']:.4f}s, MAPE={metrics['mape']:.1f}%"
        )

    predictor.save(args.output_dir)

    output_path = Path(args.output_dir)
    with open(output_path / "training_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    print("\n" + "=" * 70)
    print("LSTM LATENCY PREDICTOR RESULTS")
    print("=" * 70)
    for inst_type, m in sorted(all_metrics.items()):
        print(f"  {inst_type}: MAE={m['mae']:.4f}s, MAPE={m['mape']:.1f}%")


if __name__ == "__main__":
    main()
