"""Run all evaluation metrics on a trained checkpoint."""

import argparse
import json
import os

import torch
from datasets import load_dataset
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.prompt_formatter import format_input, parse_prediction
from src.evaluation.attention_entropy import compute_attention_entropy
from src.evaluation.classification_metrics import compute_classification_metrics
from src.evaluation.faithfulness import compute_faithfulness
from src.evaluation.rationale_metrics import compute_rationale_metrics


def generate_predictions(
    model, tokenizer, texts: list[str], batch_size: int = 8, max_new_tokens: int = 128
) -> list[str]:
    device = next(model.parameters()).device
    model.eval()
    all_outputs = []

    for start in tqdm(range(0, len(texts), batch_size), desc="Generating"):
        end = min(start + batch_size, len(texts))
        batch = texts[start:end]
        inputs = tokenizer(
            batch, return_tensors="pt", truncation=True, padding=True, max_length=384
        ).to(device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )
        for i in range(len(batch)):
            input_len = inputs["input_ids"][i].shape[0]
            gen_ids = output_ids[i][input_len:]
            text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            all_outputs.append(text)

    return all_outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--base_model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--output_dir", default="outputs/eval_results")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--skip_faithfulness", action="store_true")
    parser.add_argument("--skip_attention", action="store_true")
    args = parser.parse_args()

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, args.model_path)

    # Load test set
    ds = load_dataset("leduckhai/Sentiment-Reasoning", split="test")
    transcripts = ds["text_en"]
    ref_labels = [l.lower() for l in ds["label"]]
    ref_rationales = ds["human_justification_en"]

    input_texts = [format_input(t) for t in transcripts]
    raw_outputs = generate_predictions(
        model, tokenizer, input_texts, batch_size=args.batch_size
    )

    pred_labels, pred_rationales = [], []
    for output in raw_outputs:
        label, rationale = parse_prediction(output)
        pred_labels.append(label or "neutral")
        pred_rationales.append(rationale or "")

    # Classification
    cls_metrics = compute_classification_metrics(pred_labels, ref_labels)
    print(f"\n=== Classification ===")
    print(f"Accuracy: {cls_metrics['accuracy']:.4f}")
    print(f"Macro F1: {cls_metrics['macro_f1']:.4f}")
    print(f"F1 NEG/NEU/POS: {cls_metrics['f1_negative']:.4f} / {cls_metrics['f1_neutral']:.4f} / {cls_metrics['f1_positive']:.4f}")

    # Rationale quality
    valid_mask = [bool(r) for r in pred_rationales]
    rat_metrics = {}
    if any(valid_mask):
        valid_preds = [r for r, v in zip(pred_rationales, valid_mask) if v]
        valid_refs = [r for r, v in zip(ref_rationales, valid_mask) if v]
        rat_metrics = compute_rationale_metrics(valid_preds, valid_refs)
        print(f"\n=== Rationale Quality ===")
        for k, v in rat_metrics.items():
            print(f"  {k}: {v:.4f}")

    # Faithfulness
    faith_metrics = {}
    if not args.skip_faithfulness and any(valid_mask):
        valid_labels = [l for l, v in zip(pred_labels, valid_mask) if v]
        faith_metrics = compute_faithfulness(valid_labels, valid_preds)
        print(f"\n=== Faithfulness: {faith_metrics['faithfulness_score']:.4f} ===")

    # Attention entropy
    attn_metrics = {}
    if not args.skip_attention:
        attn_metrics = compute_attention_entropy(model, tokenizer, input_texts[:200])
        print(f"\n=== Attention Entropy: {attn_metrics['mean_entropy']:.4f} +/- {attn_metrics['std_entropy']:.4f} ===")

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    results = {
        "classification": cls_metrics,
        "rationale": rat_metrics,
        "faithfulness": {k: v for k, v in faith_metrics.items() if k != "per_sample"},
        "attention_entropy": {k: v for k, v in attn_metrics.items() if k != "per_sample"},
    }
    with open(f"{args.output_dir}/results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.output_dir}/results.json")


if __name__ == "__main__":
    main()
