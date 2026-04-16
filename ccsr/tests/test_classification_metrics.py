from src.evaluation.classification_metrics import compute_classification_metrics


def test_perfect():
    preds = ["negative", "neutral", "positive", "neutral"]
    refs = ["negative", "neutral", "positive", "neutral"]
    m = compute_classification_metrics(preds, refs)
    assert m["accuracy"] == 1.0
    assert m["macro_f1"] == 1.0


def test_partial():
    preds = ["negative", "neutral", "positive", "neutral"]
    refs = ["negative", "positive", "positive", "neutral"]
    m = compute_classification_metrics(preds, refs)
    assert 0.0 < m["accuracy"] < 1.0
    assert "f1_negative" in m
    assert "confusion_matrix" in m


def test_handles_invalid():
    preds = ["negative", "unknown", "positive"]
    refs = ["negative", "neutral", "positive"]
    m = compute_classification_metrics(preds, refs)
    assert m["accuracy"] < 1.0
