"""Classification metrics: accuracy, per-class F1, macro-F1, confusion matrix."""

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

LABEL_ORDER = ["negative", "neutral", "positive"]


def compute_classification_metrics(
    predictions: list[str],
    references: list[str],
) -> dict:
    preds_clean = [p.lower() if p.lower() in LABEL_ORDER else "INVALID" for p in predictions]
    refs_clean = [r.lower() for r in references]

    acc = accuracy_score(refs_clean, preds_clean)
    per_class_f1 = f1_score(
        refs_clean, preds_clean, labels=LABEL_ORDER, average=None, zero_division=0
    )
    macro_f1_val = f1_score(
        refs_clean, preds_clean, labels=LABEL_ORDER, average="macro", zero_division=0
    )
    cm = confusion_matrix(refs_clean, preds_clean, labels=LABEL_ORDER + ["INVALID"])

    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1_val),
        "f1_negative": float(per_class_f1[0]),
        "f1_neutral": float(per_class_f1[1]),
        "f1_positive": float(per_class_f1[2]),
        "confusion_matrix": cm[:3, :3].tolist(),
    }
