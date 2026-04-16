from src.data.curriculum import (
    compute_keyword_mismatch_scores,
    combine_difficulty_scores,
    CurriculumSampler,
)


def test_keyword_mismatch_scores():
    transcripts = [
        "patient shows recovery and improvement",
        "patient has severe disease and pain",
        "the doctor explained the procedure",
    ]
    labels = ["positive", "negative", "neutral"]
    rationales = ["signs of recovery", "severe disease condition", "objective explanation"]
    scores = compute_keyword_mismatch_scores(transcripts, labels, rationales)
    assert len(scores) == 3
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_combine_difficulty_scores():
    uncertainty = [0.8, 0.2, 0.5]
    mismatch = [0.3, 0.9, 0.4]
    combined = combine_difficulty_scores(uncertainty, mismatch, u_weight=0.6, m_weight=0.4)
    assert len(combined) == 3
    assert all(0.0 <= s <= 1.0 for s in combined)


def test_curriculum_sampler_phase1():
    difficulty_scores = [0.1, 0.2, 0.5, 0.8, 0.9, 0.3, 0.05, 0.95]
    labels = [0, 1, 2, 0, 1, 2, 0, 1]
    sampler = CurriculumSampler(
        difficulty_scores=difficulty_scores, labels=labels, threshold=0.4, class_balanced=False
    )
    indices = list(sampler)
    for idx in indices:
        assert difficulty_scores[idx] < 0.4


def test_curriculum_sampler_phase3_balanced():
    difficulty_scores = [0.1] * 10
    labels = [0, 0, 0, 0, 0, 1, 1, 2, 2, 2]  # imbalanced
    sampler = CurriculumSampler(
        difficulty_scores=difficulty_scores, labels=labels, threshold=1.0, class_balanced=True
    )
    indices = list(sampler)
    assert len(indices) >= 10  # oversampled
