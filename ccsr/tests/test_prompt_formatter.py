from src.data.prompt_formatter import format_input, format_target, format_sample, parse_prediction


def test_format_input():
    result = format_input("The patient shows signs of recovery")
    assert result == "Classify the sentiment and provide a rationale: The patient shows signs of recovery"


def test_format_target_with_rationale():
    result = format_target("positive", "Signs of recovery indicate improvement")
    assert result == "POSITIVE Signs of recovery indicate improvement"


def test_format_target_label_only():
    result = format_target("negative", rationale=None)
    assert result == "NEGATIVE"


def test_format_sample_with_rationale():
    sample = {
        "text_en": "Patient has diabetes",
        "label": "negative",
        "human_justification_en": "Negative medical condition",
    }
    inp, tgt = format_sample(sample, include_rationale=True)
    assert inp == "Classify the sentiment and provide a rationale: Patient has diabetes"
    assert tgt == "NEGATIVE Negative medical condition"


def test_format_sample_without_rationale():
    sample = {
        "text_en": "Patient has diabetes",
        "label": "negative",
        "human_justification_en": "Negative medical condition",
    }
    _, tgt = format_sample(sample, include_rationale=False)
    assert tgt == "NEGATIVE"


def test_parse_prediction_valid():
    label, rat = parse_prediction("POSITIVE Recovery is going well")
    assert label == "positive"
    assert rat == "Recovery is going well"


def test_parse_prediction_label_only():
    label, rat = parse_prediction("NEGATIVE")
    assert label == "negative"
    assert rat is None


def test_parse_prediction_invalid():
    label, rat = parse_prediction("some random text")
    assert label is None
    assert rat is None
