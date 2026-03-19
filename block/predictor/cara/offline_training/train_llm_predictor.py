#!/usr/bin/env python3
"""
LLM-based length/quality predictor for CARA (ablation).

Finetunes Qwen2.5-0.5B for length prediction and/or quality prediction
using LoRA for parameter-efficient training.

Input: prompt text → Output: predicted length or quality score
Per-model via prompt template: "Predict output length for model {model_id}: {prompt}"

Usage:
    python -m block.predictor.cara.offline_training.train_llm_predictor \
        --input data/cara/training_data/cara_v3_all_train.json \
        --test-input data/cara/training_data/cara_v3_all_test.json \
        --output-dir models/cara/llm_predictor/ \
        --target length --device cuda
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _extract_quality(m_data: dict) -> float:
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


class PredictionDataset(Dataset):
    """Dataset for LLM-based prediction finetuning."""

    def __init__(self, data: list, tokenizer, target: str = "length", max_length: int = 512):
        self.examples = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        for req in data:
            prompt = req["prompt"]
            for model_name, m_data in req["models"].items():
                if target == "length":
                    value = m_data.get("output_length", 0)
                else:
                    value = _extract_quality(m_data)

                # Format: "Predict {target} for model {model}: {prompt}\nAnswer: {value}"
                model_short = model_name.split("/")[-1]
                input_text = f"Predict {target} for model {model_short}: {prompt}"
                target_text = f"{value:.1f}" if target == "quality" else str(int(value))

                self.examples.append((input_text, target_text, float(value)))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        input_text, target_text, value = self.examples[idx]
        full_text = f"{input_text}\nAnswer: {target_text}"

        encoding = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # Labels: mask input portion, only predict answer
        labels = input_ids.clone()
        # Find "Answer: " position and mask everything before it
        answer_tokens = self.tokenizer.encode("\nAnswer: ", add_special_tokens=False)
        for i in range(len(input_ids) - len(answer_tokens)):
            if input_ids[i : i + len(answer_tokens)].tolist() == answer_tokens:
                labels[:i + len(answer_tokens)] = -100
                break

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "value": torch.tensor(value, dtype=torch.float32),
        }


def main():
    parser = argparse.ArgumentParser(
        description="Train LLM-based predictor for CARA (Qwen-0.5B finetuning)"
    )
    parser.add_argument("--input", required=True, help="Training data JSON")
    parser.add_argument("--test-input", default=None, help="Test data JSON")
    parser.add_argument(
        "--base-model",
        default="Qwen/Qwen2.5-0.5B",
        help="Base model to finetune",
    )
    parser.add_argument("--target", choices=["length", "quality"], default="length")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--output-dir", default="models/cara/llm_predictor")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
    from peft import LoraConfig, get_peft_model, TaskType

    # Load data
    with open(args.input) as f:
        if args.input.endswith(".jsonl"):
            train_data = [json.loads(line) for line in f]
        else:
            raw = json.load(f)
            train_data = raw["requests"] if "requests" in raw else raw
    logger.info(f"Training data: {len(train_data)} requests")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # Apply LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Create datasets
    train_dataset = PredictionDataset(train_data, tokenizer, args.target, args.max_length)
    logger.info(f"Training examples: {len(train_dataset)}")

    eval_dataset = None
    if args.test_input:
        with open(args.test_input) as f:
            if args.test_input.endswith(".jsonl"):
                test_data = [json.loads(line) for line in f]
            else:
                raw = json.load(f)
                test_data = raw["requests"] if "requests" in raw else raw
        eval_dataset = PredictionDataset(test_data, tokenizer, args.target, args.max_length)
        logger.info(f"Eval examples: {len(eval_dataset)}")

    # Training
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=0.05,
        weight_decay=0.01,
        logging_steps=50,
        eval_strategy="epoch" if eval_dataset else "no",
        save_strategy="epoch",
        save_total_limit=2,
        bf16=True,
        gradient_accumulation_steps=4,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0
    logger.info(f"Training completed in {train_time:.1f}s")

    # Save
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Save metadata
    metadata = {
        "base_model": args.base_model,
        "target": args.target,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "epochs": args.epochs,
        "train_time_s": train_time,
        "num_train_examples": len(train_dataset),
    }
    with open(Path(args.output_dir) / "llm_predictor_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
