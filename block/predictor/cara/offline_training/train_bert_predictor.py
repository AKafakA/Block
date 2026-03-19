#!/usr/bin/env python3
"""
Encoder-based length/quality predictor for CARA.

Supports any HuggingFace encoder model (RoBERTa, ModernBERT, DeBERTaV3, etc.)
via AutoModelForSequenceClassification with num_labels=1 (regression mode).

Trains one regression model per target LLM model.
Uses the CARA preprocessed data format: {prompt, models: {model: {output_length}}}.

Usage:
    # ModernBERT (fast, 8K context)
    python -m block.predictor.cara.offline_training.train_bert_predictor \
        --input data/cara/training_data/cara_v3_all_training_train.json \
        --test-input data/cara/training_data/cara_v3_all_training_test.json \
        --regression-model-name answerdotai/ModernBERT-base \
        --target length --epochs 20 --device cuda

    # DeBERTaV3 (most accurate)
    python -m block.predictor.cara.offline_training.train_bert_predictor \
        --regression-model-name microsoft/deberta-v3-base ...

    # RoBERTa (Block paper baseline)
    python -m block.predictor.cara.offline_training.train_bert_predictor \
        --regression-model-name roberta-base ...
"""

import argparse
import json
import logging
import os
import time
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.simplefilter(action="ignore", category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _extract_quality(m_data: dict) -> float:
    """Extract combined quality score from model data."""
    if "quality_score" in m_data:
        return float(m_data["quality_score"])
    sim = m_data.get("similarity_score")
    judge_scores = m_data.get("llm_judge_scores", {})
    valid_judges = [v for v in judge_scores.values() if v is not None]
    judge_mean = sum(valid_judges) / len(valid_judges) if valid_judges else None
    if sim is not None and judge_mean is not None:
        return 0.5 * float(sim) + 0.5 * float(judge_mean)
    elif sim is not None:
        return float(sim)
    elif judge_mean is not None:
        return float(judge_mean)
    return 0.0


class RegressionDataset(Dataset):
    """Dataset for text regression or classification — tokenizes on-the-fly."""

    def __init__(self, texts: list[str], labels: list, tokenizer, max_length: int = 1024,
                 label_dtype: str = "float"):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_dtype = label_dtype  # "float" for regression, "long" for classification

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        if self.label_dtype == "long":
            item["labels"] = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        else:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float32)
        return item


def length_to_bucket(length: float, bucket_size: int = 64, max_length: int = 1024) -> int:
    """Convert output length to bucket index."""
    bucket = int(length) // bucket_size
    max_bucket = max_length // bucket_size - 1
    return min(bucket, max_bucket)


def _extract_similarity(m_data: dict) -> float:
    """Extract similarity score from model data."""
    sim = m_data.get("similarity_score")
    if sim is not None:
        return float(sim)
    # Fallback to quality_score if no similarity
    if "quality_score" in m_data:
        return float(m_data["quality_score"])
    return 0.0


def _extract_judge_class(m_data: dict) -> int:
    """Extract LLM judge score as integer class (0-9 for 10-class classification).

    Judge scores may be on 0-1 scale (normalized) or 1-10 scale (raw).
    We detect the scale and map to 0-9 classes.
    """
    judge_scores = m_data.get("llm_judge_scores", {})
    valid_judges = [v for v in judge_scores.values() if v is not None]
    if valid_judges:
        mean_score = sum(valid_judges) / len(valid_judges)
        if mean_score <= 1.0:
            # Normalized 0-1 scale → map to 0-9
            return max(0, min(9, int(mean_score * 10)))
        else:
            # Raw 1-10 scale → map to 0-9
            return max(0, min(9, int(mean_score) - 1))
    # Fallback: try quality_score (0-1 range) → scale to 0-9
    if "quality_score" in m_data:
        return max(0, min(9, int(float(m_data["quality_score"]) * 10)))
    return 0


def build_texts_and_labels(
    data: list, model_name: str, target: str = "length",
    bucket_size: int = 0,
) -> tuple[list[str], list[float]]:
    """Extract texts and labels from CARA training data.

    Args:
        target: "length" for regression, "length_bucket" for classification,
                "quality" for quality regression, "similarity" for similarity
                regression, "judge_class" for 10-class judge classification.
        bucket_size: If >0 and target="length_bucket", convert lengths to bucket indices.
    """
    texts = []
    labels = []
    for req in data:
        m_data = req["models"].get(model_name, {})
        if not m_data:
            continue
        texts.append(req["prompt"])
        if target == "length":
            labels.append(float(m_data.get("output_length", 0)))
        elif target == "length_bucket":
            length = float(m_data.get("output_length", 0))
            labels.append(float(length_to_bucket(length, bucket_size)))
        elif target == "similarity":
            labels.append(_extract_similarity(m_data))
        elif target == "judge_class":
            labels.append(float(_extract_judge_class(m_data)))
        else:
            labels.append(_extract_quality(m_data))
    return texts, labels


def make_classification_metrics(bucket_size: int = 64):
    """Create metrics function for bucket classification."""

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        num_buckets = logits.shape[-1]
        predictions = np.argmax(logits, axis=-1)
        labels = labels.astype(int)

        # Probabilities via softmax
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        accuracy = float(np.mean(predictions == labels))
        adjacent = float(np.mean(np.abs(predictions - labels) <= 1))

        # Top-3 accuracy
        top3 = np.argsort(probs, axis=-1)[:, -3:]
        top3_hit = float(np.mean([labels[i] in top3[i] for i in range(len(labels))]))

        # P(actual bucket)
        prob_at_actual = np.array([probs[i, labels[i]] for i in range(len(labels))])
        mean_prob_actual = float(np.mean(prob_at_actual))

        # NLL
        nll = -float(np.mean(np.log(np.clip(prob_at_actual, 1e-10, 1.0))))

        # Convert buckets back to token midpoints for MAE/MAPE
        midpoints = np.array([(i * bucket_size + bucket_size / 2) for i in range(num_buckets)])
        expected_tokens = np.sum(probs * midpoints, axis=-1)
        actual_tokens = (labels * bucket_size) + bucket_size / 2
        errors = np.abs(actual_tokens - expected_tokens)
        non_zero = actual_tokens > 0
        mae = float(np.mean(errors))
        mape = float(np.mean(errors[non_zero] / actual_tokens[non_zero]) * 100) if non_zero.any() else 0.0

        return {
            "accuracy": accuracy,
            "adjacent_accuracy": adjacent,
            "top3_accuracy": top3_hit,
            "mean_prob_actual": mean_prob_actual,
            "nll": nll,
            "mae_tokens": mae,
            "mape_tokens": mape,
        }
    return compute_metrics


def make_compute_metrics(log_transform: bool = False):
    """Create metrics function, handling log-transform inverse if needed."""
    from scipy.stats import spearmanr

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = predictions.flatten()
        if log_transform:
            predictions = np.expm1(predictions)
            labels = np.expm1(labels)
        errors = np.abs(predictions - labels)
        non_zero = labels > 0
        rho = float(spearmanr(labels, predictions).correlation) if len(labels) > 2 else 0.0
        return {
            "mae": float(np.mean(errors)),
            "median_ae": float(np.median(errors)),
            "acc_50": float(np.mean(errors <= 50)),
            "acc_100": float(np.mean(errors <= 100)),
            "mape": float(np.mean(errors[non_zero] / labels[non_zero]) * 100)
            if non_zero.any() else 0.0,
            "spearman_r": rho if not np.isnan(rho) else 0.0,
        }
    return compute_metrics


class CustomLossTrainer:
    """Mixin for custom loss functions in HuggingFace Trainer."""

    @staticmethod
    def make_trainer_class(loss_type: str = "mse"):
        """Create a Trainer subclass with the specified loss function."""
        from transformers import Trainer as _Trainer

        class _CustomTrainer(_Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                labels = inputs.pop("labels")
                outputs = model(**inputs)
                logits = outputs.logits.squeeze()

                if loss_type == "mse":
                    loss = torch.nn.functional.mse_loss(logits, labels)
                elif loss_type == "huber":
                    loss = torch.nn.functional.huber_loss(logits, labels, delta=50.0)
                elif loss_type == "smape":
                    # sMAPE: |pred - actual| / (|pred| + |actual| + 1e-8)
                    numerator = torch.abs(logits - labels)
                    denominator = torch.abs(logits) + torch.abs(labels) + 1e-8
                    loss = torch.mean(numerator / denominator)
                else:
                    loss = torch.nn.functional.mse_loss(logits, labels)

                return (loss, outputs) if return_outputs else loss

        return _CustomTrainer


def train_model_for_target(
    train_texts: list[str],
    train_labels: list[float],
    val_texts: list[str],
    val_labels: list[float],
    output_dir: str,
    regression_model_name: str = "roberta-base",
    epochs: int = 50,
    batch_size: int = 16,
    lr: float = 2e-5,
    max_length: int = 1024,
    device: str = "cuda",
    log_transform: bool = False,
    loss_type: str = "mse",
    early_stopping_patience: int = 0,
    precision: str = "fp16",
    num_labels: int = 1,
    problem_type: str = "regression",
    bucket_size: int = 0,
) -> dict:
    """Train a single encoder model for regression or classification.

    Returns evaluation metrics dict.
    """
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        EarlyStoppingCallback,
    )

    # Apply log transform to labels if requested
    if log_transform:
        logger.info("  Using log1p transform on labels")
        train_labels = [float(np.log1p(x)) for x in train_labels]
        val_labels = [float(np.log1p(x)) for x in val_labels]

    logger.info(f"  Loss function: {loss_type}")

    logger.info(f"  Loading model: {regression_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(regression_model_name, trust_remote_code=True)

    # Determine max_length from model config if possible
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(regression_model_name, trust_remote_code=True)
        model_max_len = getattr(config, "max_position_embeddings", max_length)
        max_length = min(max_length, model_max_len)
        logger.info(f"  Using max_length={max_length} (model supports {model_max_len})")
    except Exception:
        pass

    logger.info(f"  Problem type: {problem_type}, num_labels: {num_labels}")

    model = AutoModelForSequenceClassification.from_pretrained(
        regression_model_name,
        num_labels=num_labels,
        problem_type=problem_type,
        trust_remote_code=True,
    )

    # For classification, labels must be integers
    if problem_type == "single_label_classification":
        train_labels = [int(x) for x in train_labels]
        val_labels = [int(x) for x in val_labels]

    label_dtype = "long" if problem_type == "single_label_classification" else "float"
    train_dataset = RegressionDataset(train_texts, train_labels, tokenizer, max_length, label_dtype)
    val_dataset = RegressionDataset(val_texts, val_labels, tokenizer, max_length, label_dtype)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=lr,
        warmup_ratio=0.03,
        lr_scheduler_type="polynomial",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy" if problem_type == "single_label_classification" else "mae",
        greater_is_better=True if problem_type == "single_label_classification" else False,
        logging_steps=100,
        report_to="none",
        fp16=(device == "cuda" and precision == "fp16"),
        bf16=(device == "cuda" and precision == "bf16"),
        dataloader_num_workers=0,
    )

    # Dynamic padding collator — pads to longest in batch, not max_length
    from transformers import DataCollatorWithPadding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Select metrics and trainer
    if problem_type == "single_label_classification":
        metrics_fn = make_classification_metrics(bucket_size)
        TrainerClass = Trainer  # classification uses default cross-entropy
    elif loss_type != "mse":
        metrics_fn = make_compute_metrics(log_transform=log_transform)
        TrainerClass = CustomLossTrainer.make_trainer_class(loss_type)
    else:
        metrics_fn = make_compute_metrics(log_transform=log_transform)
        TrainerClass = Trainer

    callbacks = []
    if early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=early_stopping_patience))

    trainer = TrainerClass(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=metrics_fn,
        callbacks=callbacks,
    )

    trainer.train()

    # Final evaluation
    eval_results = trainer.evaluate()

    # Save best model
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    if problem_type == "single_label_classification":
        return {
            "accuracy": eval_results.get("eval_accuracy", 0),
            "adjacent_accuracy": eval_results.get("eval_adjacent_accuracy", 0),
            "mae_tokens": eval_results.get("eval_mae_tokens", 0),
            "mape_tokens": eval_results.get("eval_mape_tokens", 0),
            "n": len(val_labels),
        }
    else:
        return {
            "mae": eval_results.get("eval_mae", 0),
            "median_ae": eval_results.get("eval_median_ae", 0),
            "acc_50": eval_results.get("eval_acc_50", 0),
            "acc_100": eval_results.get("eval_acc_100", 0),
            "mape": eval_results.get("eval_mape", 0),
            "spearman_r": eval_results.get("eval_spearman_r", 0),
            "n": len(val_labels),
        }


def main():
    parser = argparse.ArgumentParser(
        description="Train encoder-based length/quality predictor for CARA"
    )
    parser.add_argument("--input", required=True, help="Training data JSON")
    parser.add_argument("--test-input", default=None, help="Test data JSON")
    parser.add_argument(
        "--regression-model-name",
        default="answerdotai/ModernBERT-base",
        help="Base encoder model (e.g., roberta-base, answerdotai/ModernBERT-base, "
             "microsoft/deberta-v3-base)",
    )
    parser.add_argument(
        "--target-models",
        nargs="+",
        default=None,
        help="Target LLM models to train for (default: all models in data)",
    )
    parser.add_argument(
        "--target",
        choices=["length", "quality", "both", "length_bucket", "similarity", "judge_class"],
        default="length",
        help="What to predict: length (regression), quality (combined regression), "
             "similarity (similarity regression), judge_class (10-class classification), "
             "length_bucket (bucket classification), both (length + quality)",
    )
    parser.add_argument("--bucket-size", type=int, default=64,
                        help="Bucket size in tokens for length_bucket target")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--log-transform", action="store_true",
                        help="Apply log1p transform to labels (helps MAPE)")
    parser.add_argument("--loss-type", choices=["mse", "huber", "smape"], default="mse",
                        help="Loss function: mse (default), huber (robust), smape (relative)")
    parser.add_argument("--early-stopping-patience", type=int, default=0,
                        help="Early stopping patience (0=disabled)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max-length", type=int, default=1024,
                        help="Max tokenizer length (ModernBERT supports 8192)")
    parser.add_argument("--output-dir", default="models/cara/encoder_length")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--precision", choices=["fp16", "bf16"], default="fp16",
                        help="Training precision (DeBERTaV3 requires bf16)")
    args = parser.parse_args()

    # Load data
    with open(args.input) as f:
        if args.input.endswith(".jsonl"):
            train_data = [json.loads(line) for line in f]
        else:
            raw = json.load(f)
            train_data = raw["requests"] if "requests" in raw else raw
    logger.info(f"Training data: {len(train_data)} requests")

    test_data = None
    if args.test_input:
        with open(args.test_input) as f:
            if args.test_input.endswith(".jsonl"):
                test_data = [json.loads(line) for line in f]
            else:
                raw = json.load(f)
                test_data = raw["requests"] if "requests" in raw else raw
        logger.info(f"Test data: {len(test_data)} requests")

    # Determine target models
    target_models = args.target_models or sorted(train_data[0]["models"].keys())
    targets = ["length", "quality"] if args.target == "both" else [args.target]

    # Classification setup
    is_classification = args.target in ("length_bucket", "judge_class")
    num_labels = 1
    problem_type = "regression"
    if args.target == "length_bucket":
        num_labels = 1024 // args.bucket_size  # e.g., 1024/64 = 16 buckets
        problem_type = "single_label_classification"
        logger.info(f"Classification mode: {num_labels} buckets, bucket_size={args.bucket_size}")
    elif args.target == "judge_class":
        num_labels = 10  # 10-class classification (judge scores 1-10 → classes 0-9)
        problem_type = "single_label_classification"
        logger.info(f"Classification mode: {num_labels} judge quality classes")

    all_results = {}
    model_short = args.regression_model_name.split("/")[-1]

    for model_name in target_models:
        model_key = model_name.replace("/", "_")
        for target in targets:
            logger.info(f"Training {target} predictor for {model_name} using {model_short}...")

            train_texts, train_labels = build_texts_and_labels(
                train_data, model_name, target, bucket_size=args.bucket_size
            )
            val_texts, val_labels = build_texts_and_labels(
                test_data or train_data[-500:], model_name, target,
                bucket_size=args.bucket_size,
            )

            out_dir = str(Path(args.output_dir) / f"{model_key}_{target}")

            t0 = time.time()
            metrics = train_model_for_target(
                train_texts, train_labels,
                val_texts, val_labels,
                output_dir=out_dir,
                regression_model_name=args.regression_model_name,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                max_length=args.max_length,
                device=args.device,
                log_transform=args.log_transform,
                loss_type=args.loss_type,
                early_stopping_patience=args.early_stopping_patience,
                precision=args.precision,
                num_labels=num_labels,
                problem_type=problem_type,
                bucket_size=args.bucket_size,
            )
            elapsed = time.time() - t0

            result_key = f"{model_name}_{target}"
            all_results[result_key] = {**metrics, "train_time_s": elapsed}
            if "accuracy" in metrics:
                logger.info(
                    f"  {result_key}: Acc={metrics['accuracy']:.3f}, "
                    f"AdjAcc={metrics.get('adjacent_accuracy', 0):.3f}, "
                    f"time={elapsed:.0f}s"
                )
            else:
                logger.info(
                    f"  {result_key}: MAE={metrics['mae']:.1f}, MAPE={metrics['mape']:.1f}%, "
                    f"Acc@50={metrics['acc_50']:.3f}, Acc@100={metrics['acc_100']:.3f}, "
                    f"time={elapsed:.0f}s"
                )

    # Print summary
    print(f"\n{'=' * 70}")
    print(f"  {model_short.upper()} PREDICTION RESULTS ({args.target.upper()})")
    print(f"{'=' * 70}")

    # Detect result type from first entry
    first_result = next(iter(all_results.values()), {})
    if "accuracy" in first_result:
        header = f"{'Target Model':<25} {'Acc':>7} {'AdjAcc':>7} {'Top3':>7} {'MAE_tok':>8} {'MAPE_tok':>9}"
        print(header)
        print("-" * 70)
        for key, m in sorted(all_results.items()):
            short_key = key.split("/")[-1] if "/" in key else key
            print(
                f"  {short_key:<23} {m.get('accuracy', 0):>7.3f} "
                f"{m.get('adjacent_accuracy', 0):>7.3f} "
                f"{m.get('top3_accuracy', 0) if 'top3_accuracy' in m else 0:>7.3f} "
                f"{m.get('mae_tokens', 0):>8.1f} {m.get('mape_tokens', 0):>8.1f}%"
            )
    else:
        header = f"{'Target Model':<25} {'MAE':>6} {'MAPE':>7} {'Acc@50':>7} {'Acc@100':>8} {'Spearman':>9}"
        print(header)
        print("-" * 70)
        for key, m in sorted(all_results.items()):
            short_key = key.split("/")[-1] if "/" in key else key
            print(
                f"  {short_key:<23} {m['mae']:>6.1f} {m['mape']:>6.1f}% "
                f"{m['acc_50']:>7.3f} {m['acc_100']:>8.3f} {m['spearman_r']:>9.3f}"
            )

    # Save results
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    results_file = output_path / "training_results.json"
    with open(results_file, "w") as f:
        json.dump(
            {"model": args.regression_model_name, "results": all_results},
            f, indent=2,
        )
    logger.info(f"Results saved to {results_file}")


if __name__ == "__main__":
    main()
