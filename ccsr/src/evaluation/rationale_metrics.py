"""Rationale quality metrics: ROUGE scores and BERTScore."""

import numpy as np
from rouge_score import rouge_scorer
from bert_score import score as bert_score_fn


def compute_rationale_metrics(
    predictions: list[str],
    references: list[str],
    bert_score_lang: str = "en",
) -> dict:
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL", "rougeLsum"], use_stemmer=True
    )

    rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": [], "rougeLsum": []}
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        for key in rouge_scores:
            rouge_scores[key].append(scores[key].fmeasure)

    P, R, F1 = bert_score_fn(predictions, references, lang=bert_score_lang, verbose=False)

    return {
        "rouge1": float(np.mean(rouge_scores["rouge1"])),
        "rouge2": float(np.mean(rouge_scores["rouge2"])),
        "rougeL": float(np.mean(rouge_scores["rougeL"])),
        "rougeLsum": float(np.mean(rouge_scores["rougeLsum"])),
        "bertscore_precision": float(P.mean()),
        "bertscore_recall": float(R.mean()),
        "bertscore_f1": float(F1.mean()),
    }
