"""Difficulty scoring and curriculum sampling for CCSR."""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import Sampler


def compute_keyword_mismatch_scores(
    transcripts: list[str],
    labels: list[str],
    rationales: list[str],
) -> list[float]:
    """Score each transcript by keyword-label mismatch using TF-IDF on rationales.

    Builds a per-class lexicon from rationale TF-IDF, then checks how many
    sentiment-associated words in the transcript conflict with its label.
    """
    label_set = sorted(set(labels))
    class_rationale_docs = {}
    for label in label_set:
        class_rats = [r for r, l in zip(rationales, labels) if l == label]
        class_rationale_docs[label] = " ".join(class_rats)

    vectorizer = TfidfVectorizer(max_features=500, stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(class_rationale_docs.values())
    feature_names = vectorizer.get_feature_names_out()
    tfidf_array = tfidf_matrix.toarray()

    word_to_class = {}
    for i, word in enumerate(feature_names):
        class_idx = int(np.argmax(tfidf_array[:, i]))
        if tfidf_array[class_idx, i] > 0.01:
            word_to_class[word] = label_set[class_idx]

    scores = []
    for transcript, label in zip(transcripts, labels):
        words = set(transcript.lower().split())
        sentiment_words = words & set(word_to_class.keys())
        if not sentiment_words:
            scores.append(0.0)
            continue
        mismatches = sum(1 for w in sentiment_words if word_to_class[w] != label)
        scores.append(mismatches / len(sentiment_words))
    return scores


def compute_uncertainty_scores(entropies: list[float]) -> list[float]:
    """Normalize raw entropy values to [0, 1]."""
    arr = np.array(entropies, dtype=np.float64)
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-8:
        return [0.5] * len(entropies)
    return ((arr - mn) / (mx - mn)).tolist()


def combine_difficulty_scores(
    uncertainty: list[float],
    mismatch: list[float],
    u_weight: float = 0.6,
    m_weight: float = 0.4,
) -> list[float]:
    """Combine uncertainty and mismatch into a single difficulty score in [0, 1]."""
    u = np.array(uncertainty, dtype=np.float64)
    m = np.array(mismatch, dtype=np.float64)

    def norm(a):
        mn, mx = a.min(), a.max()
        return (a - mn) / (mx - mn + 1e-8)

    combined = u_weight * norm(u) + m_weight * norm(m)
    combined = norm(combined)
    return combined.tolist()


class CurriculumSampler(Sampler):
    """Sampler that filters by difficulty threshold and optionally class-balances."""

    def __init__(
        self,
        difficulty_scores: list[float],
        labels: list[int],
        threshold: float = 1.0,
        class_balanced: bool = False,
        seed: int = 42,
    ):
        self.difficulty_scores = difficulty_scores
        self.labels = labels
        self.threshold = threshold
        self.class_balanced = class_balanced
        self.rng = np.random.RandomState(seed)

    def __iter__(self):
        eligible = [i for i, d in enumerate(self.difficulty_scores) if d < self.threshold]

        if not self.class_balanced:
            self.rng.shuffle(eligible)
            return iter(eligible)

        class_indices: dict[int, list[int]] = {}
        for i in eligible:
            class_indices.setdefault(self.labels[i], []).append(i)

        max_count = max(len(v) for v in class_indices.values())
        balanced = []
        for lbl, idxs in class_indices.items():
            if len(idxs) < max_count:
                oversampled = list(idxs) + list(
                    self.rng.choice(idxs, max_count - len(idxs), replace=True)
                )
                balanced.extend(oversampled)
            else:
                balanced.extend(idxs)

        self.rng.shuffle(balanced)
        return iter(balanced)

    def __len__(self):
        eligible = [i for i, d in enumerate(self.difficulty_scores) if d < self.threshold]
        if not self.class_balanced:
            return len(eligible)
        class_indices: dict[int, list[int]] = {}
        for i in eligible:
            class_indices.setdefault(self.labels[i], []).append(i)
        if not class_indices:
            return 0
        max_count = max(len(v) for v in class_indices.values())
        return max_count * len(class_indices)
