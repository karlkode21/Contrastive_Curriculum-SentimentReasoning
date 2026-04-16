"""Offline: compute pairwise BERTScore between rationales within each class."""

import argparse
import json
import os

import torch
from bert_score import score as bert_score_fn
from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--output_dir", default="outputs/rationale_sim")
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    ds = load_dataset("leduckhai/Sentiment-Reasoning")
    train = ds["train"]
    rationales = train["human_justification_en"]
    labels = [l.lower() for l in train["label"]]
    label_set = ["negative", "neutral", "positive"]

    all_pairs: list[tuple[int, int, float]] = []
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for label in label_set:
        indices = [i for i, l in enumerate(labels) if l == label]
        class_rationales = [rationales[i] for i in indices]
        n = len(class_rationales)
        print(f"[{label}] {n} samples, {n * (n - 1) // 2} pairs to check")

        # Build all pairs for this class
        cands, refs, pair_map = [], [], []
        for i in range(n):
            for j in range(i + 1, n):
                cands.append(class_rationales[i])
                refs.append(class_rationales[j])
                pair_map.append((indices[i], indices[j]))

        # Score in sub-batches
        for sb_start in range(0, len(cands), args.batch_size):
            sb_end = min(sb_start + args.batch_size, len(cands))
            _, _, F1 = bert_score_fn(
                cands[sb_start:sb_end],
                refs[sb_start:sb_end],
                lang="en",
                verbose=False,
                device=device,
            )
            for k, f1_val in enumerate(F1.tolist()):
                if f1_val >= args.threshold:
                    gi, gj = pair_map[sb_start + k]
                    all_pairs.append((gi, gj, f1_val))

        print(f"  Found {sum(1 for p in all_pairs if labels[p[0]] == label)} pairs above threshold so far")

    os.makedirs(args.output_dir, exist_ok=True)
    output = {
        "threshold": args.threshold,
        "pairs": [(int(i), int(j), round(float(s), 4)) for i, j, s in all_pairs],
        "num_pairs": len(all_pairs),
    }
    with open(f"{args.output_dir}/rationale_sim.json", "w") as f:
        json.dump(output, f)

    print(f"\nTotal: {len(all_pairs)} pairs above threshold {args.threshold}")


if __name__ == "__main__":
    main()
