"""Attention entropy over input tokens -- measures keyword reliance."""

import torch
import numpy as np


def compute_attention_entropy(
    model,
    tokenizer,
    texts: list[str],
    batch_size: int = 8,
    max_length: int = 512,
) -> dict:
    """Compute average attention entropy over input tokens per sample.

    Higher entropy = more distributed attention = less keyword fixation.
    """
    device = next(model.parameters()).device
    model.eval()
    all_entropies = []

    for start in range(0, len(texts), batch_size):
        end = min(start + batch_size, len(texts))
        batch = texts[start:end]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)

        for b in range(len(batch)):
            input_len = int(inputs["attention_mask"][b].sum().item())
            sample_entropies = []
            for layer_attn in outputs.attentions:
                attn = layer_attn[b, :, :input_len, :input_len]  # (heads, L, L)
                entropy = -(attn * torch.log(attn + 1e-10)).sum(dim=-1)  # (heads, L)
                sample_entropies.append(entropy.mean().item())
            all_entropies.append(np.mean(sample_entropies))

    return {
        "mean_entropy": float(np.mean(all_entropies)),
        "std_entropy": float(np.std(all_entropies)),
        "per_sample": all_entropies,
    }
