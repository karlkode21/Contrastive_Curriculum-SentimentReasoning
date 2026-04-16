"""NLI-based faithfulness: does the rationale entail the predicted label?"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


LABEL_HYPOTHESES = {
    "positive": "The sentiment expressed is positive.",
    "neutral": "The sentiment expressed is neutral.",
    "negative": "The sentiment expressed is negative.",
}


def compute_faithfulness(
    labels: list[str],
    rationales: list[str],
    model_name: str = "microsoft/deberta-v3-base-mnli-fever-anli",
    batch_size: int = 32,
) -> dict:
    """Check if each rationale entails its predicted label using NLI.

    Returns dict with "faithfulness_score" (fraction entailing) and "per_sample" (bool list).
    """
    if not labels:
        return {"faithfulness_score": 0.0, "per_sample": []}

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    premises = rationales
    hypotheses = [LABEL_HYPOTHESES[l.lower()] for l in labels]

    entailments = []
    for start in range(0, len(premises), batch_size):
        end = min(start + batch_size, len(premises))
        inputs = tokenizer(
            premises[start:end],
            hypotheses[start:end],
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        ).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits
            # NLI label order: 0=contradiction, 1=neutral, 2=entailment
            preds = logits.argmax(dim=-1).cpu().tolist()
            entailments.extend([p == 2 for p in preds])

    score = sum(entailments) / len(entailments) if entailments else 0.0
    return {"faithfulness_score": float(score), "per_sample": entailments}
