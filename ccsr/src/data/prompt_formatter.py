"""Post-thinking prompt formatter for Sentiment Reasoning."""

LABEL_NAMES = {"negative": "NEGATIVE", "neutral": "NEUTRAL", "positive": "POSITIVE"}
LABEL_FROM_UPPER = {v: k for k, v in LABEL_NAMES.items()}

INSTRUCTION_PREFIX = "Classify the sentiment and provide a rationale: "


def format_input(transcript: str) -> str:
    return f"{INSTRUCTION_PREFIX}{transcript}"


def format_target(label: str, rationale: str | None = None) -> str:
    upper_label = LABEL_NAMES[label.lower()]
    if rationale:
        return f"{upper_label} {rationale}"
    return upper_label


def format_sample(
    sample: dict,
    text_col: str = "text_en",
    rationale_col: str = "human_justification_en",
    label_col: str = "label",
    include_rationale: bool = True,
) -> tuple[str, str]:
    inp = format_input(sample[text_col])
    rat = sample.get(rationale_col) if include_rationale else None
    tgt = format_target(sample[label_col], rat)
    return inp, tgt


def parse_prediction(text: str) -> tuple[str | None, str | None]:
    """Parse model output into (label, rationale). Returns (None, None) on failure."""
    text = text.strip()
    for upper_label in LABEL_FROM_UPPER:
        if text.upper().startswith(upper_label):
            remainder = text[len(upper_label):].strip()
            return LABEL_FROM_UPPER[upper_label], remainder or None
    return None, None
