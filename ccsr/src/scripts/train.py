"""Main training script for CCSR and ablation variants."""

import argparse
import json
import os

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    TrainingArguments,
)

from src.data.load_dataset import get_class_weights, load_sentiment_reasoning
from src.data.prompt_formatter import LABEL_NAMES
from src.models.ccsr_trainer import CCSRTrainer


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output_dir", default="outputs/ccsr")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--difficulty_scores", default=None)
    parser.add_argument("--rationale_sim", default=None)
    # Ablation flags
    parser.add_argument("--no_contrastive", action="store_true")
    parser.add_argument("--no_curriculum", action="store_true")
    parser.add_argument("--no_focal", action="store_true")
    parser.add_argument("--label_only", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.no_contrastive:
        config["contrastive"]["enabled"] = False
    if args.no_curriculum:
        config["curriculum"]["enabled"] = False
    if args.no_focal:
        config["focal_loss"]["enabled"] = False

    # ---- Dataset ----
    include_rationale = not args.label_only
    ds = load_sentiment_reasoning(
        dataset_name=config["data"]["dataset_name"],
        text_col=config["data"]["text_column"],
        rationale_col=config["data"]["rationale_column"],
        label_col=config["data"]["label_column"],
        include_rationale=include_rationale,
    )

    # ---- Model ----
    model_name = config["model"]["backbone"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=getattr(torch, config["model"]["torch_dtype"]),
        device_map="auto",
    )

    lora_config = LoraConfig(
        r=config["model"]["lora_r"],
        lora_alpha=config["model"]["lora_alpha"],
        target_modules=config["model"]["lora_target_modules"],
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ---- Tokenize ----
    max_len = config["data"]["max_seq_length"]

    def tokenize(example):
        input_text = example["input_text"]
        target_text = example["target_text"]
        full_text = f"{input_text}\n{target_text}"

        tokenized = tokenizer(
            full_text, truncation=True, max_length=max_len, padding="max_length"
        )
        input_ids = tokenizer(input_text + "\n", truncation=True, max_length=max_len)[
            "input_ids"
        ]
        labels = tokenized["input_ids"].copy()
        labels[: len(input_ids)] = [-100] * len(input_ids)
        tokenized["labels"] = labels
        tokenized["label_idx"] = example["label_idx"]
        return tokenized

    train_ds = ds["train"].map(tokenize, remove_columns=ds["train"].column_names)
    eval_ds = ds["test"].map(tokenize, remove_columns=ds["test"].column_names)
    train_ds = train_ds.map(lambda ex, idx: {"global_idx": idx}, with_indices=True)

    class_weights = get_class_weights(ds["train"])

    # Label token IDs for focal loss
    label_token_ids = {}
    for idx, (name, upper) in enumerate(LABEL_NAMES.items()):
        tid = tokenizer.encode(upper, add_special_tokens=False)[0]
        label_token_ids[idx] = tid

    # Difficulty scores
    difficulty_scores = None
    train_labels = None
    if args.difficulty_scores and config["curriculum"]["enabled"]:
        with open(args.difficulty_scores) as f:
            diff_data = json.load(f)
        difficulty_scores = diff_data["difficulty"]
        train_labels = [ex["label_idx"] for ex in ds["train"]]

    # ---- Training ----
    tc = config["training"]
    output_dir = f"{args.output_dir}/seed_{args.seed}"
    use_deepspeed = torch.cuda.device_count() > 1
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=tc["num_epochs"],
        per_device_train_batch_size=tc["per_device_batch_size"],
        gradient_accumulation_steps=tc["gradient_accumulation_steps"],
        per_device_eval_batch_size=tc["per_device_batch_size"],
        learning_rate=tc["learning_rate"],
        warmup_ratio=tc["warmup_ratio"],
        lr_scheduler_type=tc["lr_scheduler_type"],
        weight_decay=tc["weight_decay"],
        eval_strategy=tc["eval_strategy"],
        save_strategy=tc["save_strategy"],
        load_best_model_at_end=True,
        metric_for_best_model=tc["metric_for_best_model"],
        greater_is_better=True,
        seed=args.seed,
        bf16=True,
        deepspeed="configs/ds_config.json" if use_deepspeed else None,
        report_to="none",
    )

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)

    trainer = CCSRTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        contrastive_config=config["contrastive"],
        focal_config=config["focal_loss"],
        curriculum_config=config["curriculum"],
        difficulty_scores=difficulty_scores,
        train_labels=train_labels,
        rationale_sim_path=args.rationale_sim,
        label_token_ids=label_token_ids,
        class_weights=class_weights,
    )

    trainer.train()
    trainer.save_model(f"{output_dir}/best_model")
    tokenizer.save_pretrained(f"{output_dir}/best_model")
    print(f"Training complete. Best model saved to {output_dir}/best_model")


if __name__ == "__main__":
    main()
