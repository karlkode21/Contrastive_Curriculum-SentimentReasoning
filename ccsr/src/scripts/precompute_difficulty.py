"""Offline: train BERT classifier on English subset, compute per-sample difficulty scores."""

import argparse
import json
import os

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from src.data.curriculum import (
    combine_difficulty_scores,
    compute_keyword_mismatch_scores,
    compute_uncertainty_scores,
)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert_model", default="bert-base-uncased")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--output_dir", default="outputs/difficulty_scores")
    args = parser.parse_args()

    label_map = {"negative": 0, "neutral": 1, "positive": 2}
    ds = load_dataset("leduckhai/Sentiment-Reasoning")

    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
    model = AutoModelForSequenceClassification.from_pretrained(args.bert_model, num_labels=3)

    def preprocess(example):
        tokens = tokenizer(
            example["text_en"], truncation=True, padding="max_length", max_length=256
        )
        tokens["labels"] = label_map[example["label"].lower()]
        return tokens

    train_ds = ds["train"].map(preprocess, batched=False, remove_columns=ds["train"].column_names)
    train_ds.set_format("torch")

    training_args = TrainingArguments(
        output_dir=f"{args.output_dir}/bert_classifier",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        bf16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=train_ds,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    # Entropy on training predictions
    predictions = trainer.predict(train_ds)
    logits = torch.tensor(predictions.predictions)
    probs = torch.softmax(logits, dim=-1)
    entropies = (-probs * torch.log(probs + 1e-8)).sum(dim=-1).tolist()

    # Keyword mismatch
    raw_train = ds["train"]
    transcripts = raw_train["text_en"]
    labels_str = [l.lower() for l in raw_train["label"]]
    rationales = raw_train["human_justification_en"]

    uncertainty = compute_uncertainty_scores(entropies)
    mismatch = compute_keyword_mismatch_scores(transcripts, labels_str, rationales)
    difficulty = combine_difficulty_scores(uncertainty, mismatch)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(f"{args.output_dir}/difficulty_scores.json", "w") as f:
        json.dump(
            {"uncertainty": uncertainty, "mismatch": mismatch, "difficulty": difficulty}, f
        )

    print(f"Saved {len(difficulty)} difficulty scores to {args.output_dir}/difficulty_scores.json")
    print(
        f"  <0.4: {sum(1 for d in difficulty if d < 0.4)},  "
        f"0.4-0.7: {sum(1 for d in difficulty if 0.4 <= d < 0.7)},  "
        f">=0.7: {sum(1 for d in difficulty if d >= 0.7)}"
    )


if __name__ == "__main__":
    main()
