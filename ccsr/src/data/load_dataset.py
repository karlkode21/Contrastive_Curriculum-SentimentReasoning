"""Load and preprocess the English subset of the Sentiment Reasoning dataset."""

from collections import Counter
from datasets import load_dataset as hf_load_dataset, DatasetDict
from src.data.prompt_formatter import format_sample


def load_sentiment_reasoning(
    dataset_name: str = "leduckhai/Sentiment-Reasoning",
    text_col: str = "text_en",
    rationale_col: str = "human_justification_en",
    label_col: str = "label",
    include_rationale: bool = True,
) -> DatasetDict:
    """Load dataset from HuggingFace and add formatted input/target columns."""
    ds = hf_load_dataset(dataset_name)

    def add_formatted_columns(example):
        inp, tgt = format_sample(
            example,
            text_col=text_col,
            rationale_col=rationale_col,
            label_col=label_col,
            include_rationale=include_rationale,
        )
        label_idx = {"negative": 0, "neutral": 1, "positive": 2}[example[label_col].lower()]
        return {"input_text": inp, "target_text": tgt, "label_idx": label_idx}

    ds = ds.map(add_formatted_columns)
    return ds


def get_class_weights(dataset) -> list[float]:
    """Compute inverse-frequency class weights from a dataset split."""
    counts = Counter(dataset["label_idx"])
    total = sum(counts.values())
    weights = [total / (len(counts) * counts[i]) for i in range(len(counts))]
    return weights
