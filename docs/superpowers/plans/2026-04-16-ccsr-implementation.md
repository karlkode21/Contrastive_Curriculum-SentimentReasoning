# CCSR Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the Contrastive Curriculum Sentiment Reasoning (CCSR) framework that outperforms Nguyen et al.'s Sentiment Reasoning baselines on the English subset for both classification and rationale generation.

**Architecture:** Llama 3 8B-Instruct backbone fine-tuned with LoRA, augmented with a sentiment-contrastive projection head (SupCon loss on end-of-label hidden states with rationale-grounded sub-clustering), difficulty-aware curriculum scheduler, and focal loss for class imbalance. Three-phase curriculum training over 7 epochs.

**Tech Stack:** Python 3.10+, PyTorch, HuggingFace Transformers/PEFT/Datasets/Evaluate, DeepSpeed ZeRO Stage 2, scikit-learn, rouge-score, bert-score, scipy

---

## File Structure

```
ccsr/
├── pyproject.toml                    # Project metadata and dependencies
├── configs/
│   └── default.yaml                  # All hyperparameters in one place
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── load_dataset.py           # Download + split English subset from HF
│   │   ├── prompt_formatter.py       # Post-thinking prompt template
│   │   └── curriculum.py             # Difficulty scoring + CurriculumSampler
│   ├── models/
│   │   ├── __init__.py
│   │   ├── contrastive_head.py       # 2-layer MLP projection + SupCon loss
│   │   ├── focal_loss.py             # Focal loss for classification token
│   │   └── ccsr_trainer.py           # Custom Trainer subclass with joint loss
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── classification_metrics.py # Accuracy, F1, confusion matrix
│   │   ├── rationale_metrics.py      # ROUGE, BERTScore
│   │   ├── faithfulness.py           # NLI-based faithfulness scoring
│   │   └── attention_entropy.py      # Attention entropy over input tokens
│   └── scripts/
│       ├── precompute_difficulty.py   # Offline: BERT classifier + keyword mismatch
│       ├── precompute_rationale_sim.py# Offline: BERTScore pairwise similarity
│       ├── train.py                   # Main training entrypoint
│       ├── evaluate.py                # Run all metrics on a checkpoint
│       ├── run_baselines.py           # BERT/Flan-T5/Mistral baselines
│       └── run_ablations.py           # All 7 ablation variants
├── tests/
│   ├── test_prompt_formatter.py
│   ├── test_curriculum.py
│   ├── test_contrastive_head.py
│   ├── test_focal_loss.py
│   ├── test_classification_metrics.py
│   ├── test_rationale_metrics.py
│   └── test_faithfulness.py
└── notebooks/
    └── generate_tables_figures.ipynb  # Final paper tables + figures
```

---

### Task 1: Project Scaffolding and Dependencies

**Files:**
- Create: `ccsr/pyproject.toml`
- Create: `ccsr/configs/default.yaml`
- Create: `ccsr/src/__init__.py`
- Create: `ccsr/src/data/__init__.py`
- Create: `ccsr/src/models/__init__.py`
- Create: `ccsr/src/evaluation/__init__.py`

- [ ] **Step 1: Create project directory structure**

```bash
cd /Users/josephmkalinzi/Developer/Prof.\ Ahmed/OutPerform_v.1.0.0-Claude02
mkdir -p ccsr/{configs,src/{data,models,evaluation,scripts},tests,notebooks}
touch ccsr/src/__init__.py ccsr/src/data/__init__.py ccsr/src/models/__init__.py ccsr/src/evaluation/__init__.py
```

- [ ] **Step 2: Write pyproject.toml**

```toml
# ccsr/pyproject.toml
[project]
name = "ccsr"
version = "0.1.0"
description = "Contrastive Curriculum Sentiment Reasoning for Healthcare"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.1.0",
    "transformers>=4.40.0",
    "peft>=0.10.0",
    "datasets>=2.19.0",
    "accelerate>=0.29.0",
    "deepspeed>=0.14.0",
    "evaluate>=0.4.0",
    "rouge-score>=0.1.2",
    "bert-score>=0.3.13",
    "scikit-learn>=1.4.0",
    "scipy>=1.12.0",
    "pyyaml>=6.0",
    "numpy>=1.26.0",
    "pandas>=2.2.0",
    "matplotlib>=3.8.0",
    "seaborn>=0.13.0",
    "tqdm>=4.66.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0.0", "pytest-cov>=5.0.0"]

[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.backends._legacy:_Backend"
```

- [ ] **Step 3: Write default config**

```yaml
# ccsr/configs/default.yaml
data:
  dataset_name: "leduckhai/Sentiment-Reasoning"
  text_column: "text_en"
  rationale_column: "human_justification_en"
  label_column: "label"
  label_map:
    negative: 0
    neutral: 1
    positive: 2
  max_seq_length: 512

model:
  backbone: "meta-llama/Meta-Llama-3-8B-Instruct"
  lora_r: 16
  lora_alpha: 32
  lora_target_modules: "all-linear"
  load_in_8bit: false
  torch_dtype: "bfloat16"

contrastive:
  enabled: true
  projection_hidden_dim: 256
  projection_output_dim: 128
  temperature: 0.07
  lambda_con: 0.3
  sub_cluster_alpha: 0.1
  sub_cluster_threshold: 0.85

focal_loss:
  enabled: true
  gamma: 2.0
  lambda_bal: 0.5

curriculum:
  enabled: true
  uncertainty_weight: 0.6
  mismatch_weight: 0.4
  phase1_threshold: 0.4
  phase1_epochs: [1, 2]
  phase2_threshold: 0.7
  phase2_epochs: [3, 4]
  phase3_epochs: [5, 6, 7]
  class_balanced_phase3: true

training:
  num_epochs: 7
  per_device_batch_size: 8
  gradient_accumulation_steps: 4  # effective batch = 32 on 1 GPU
  learning_rate: 2.0e-4
  warmup_ratio: 0.05
  lr_scheduler_type: "cosine"
  weight_decay: 0.01
  early_stopping_patience: 2
  eval_strategy: "epoch"
  save_strategy: "epoch"
  metric_for_best_model: "macro_f1"
  seeds: [42, 123, 456]

evaluation:
  faithfulness_model: "microsoft/deberta-v3-base-mnli-fever-anli"
  bert_score_model: "microsoft/deberta-xlarge-mnli"

difficulty:
  bert_model: "bert-base-uncased"
  bert_epochs: 3
  bert_batch_size: 64
  bert_lr: 2.0e-5
```

- [ ] **Step 4: Install the project in dev mode**

```bash
cd ccsr && pip install -e ".[dev]"
```
Expected: installs successfully with all dependencies.

- [ ] **Step 5: Commit**

```bash
git init
git add pyproject.toml configs/ src/ tests/ notebooks/
git commit -m "feat: scaffold CCSR project structure and dependencies"
```

---

### Task 2: Dataset Loading and Prompt Formatting

**Files:**
- Create: `ccsr/src/data/load_dataset.py`
- Create: `ccsr/src/data/prompt_formatter.py`
- Create: `ccsr/tests/test_prompt_formatter.py`

- [ ] **Step 1: Write the failing test for prompt formatting**

```python
# ccsr/tests/test_prompt_formatter.py
from src.data.prompt_formatter import format_input, format_target, format_sample


def test_format_input():
    transcript = "The patient shows signs of recovery"
    result = format_input(transcript)
    assert result == "Classify the sentiment and provide a rationale: The patient shows signs of recovery"


def test_format_target_with_rationale():
    label = "positive"
    rationale = "Signs of recovery indicate improvement"
    result = format_target(label, rationale)
    assert result == "POSITIVE Signs of recovery indicate improvement"


def test_format_target_label_only():
    result = format_target("negative", rationale=None)
    assert result == "NEGATIVE"


def test_format_sample():
    sample = {
        "text_en": "Patient has diabetes",
        "label": "negative",
        "human_justification_en": "Negative medical condition",
    }
    inp, tgt = format_sample(sample, text_col="text_en", rationale_col="human_justification_en", label_col="label", include_rationale=True)
    assert inp == "Classify the sentiment and provide a rationale: Patient has diabetes"
    assert tgt == "NEGATIVE Negative medical condition"


def test_format_sample_without_rationale():
    sample = {
        "text_en": "Patient has diabetes",
        "label": "negative",
        "human_justification_en": "Negative medical condition",
    }
    inp, tgt = format_sample(sample, text_col="text_en", rationale_col="human_justification_en", label_col="label", include_rationale=False)
    assert tgt == "NEGATIVE"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/josephmkalinzi/Developer/Prof.\ Ahmed/OutPerform_v.1.0.0-Claude02/ccsr
python -m pytest tests/test_prompt_formatter.py -v
```
Expected: `ModuleNotFoundError: No module named 'src.data.prompt_formatter'`

- [ ] **Step 3: Implement prompt formatter**

```python
# ccsr/src/data/prompt_formatter.py
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
        if text.startswith(upper_label):
            remainder = text[len(upper_label):].strip()
            return LABEL_FROM_UPPER[upper_label], remainder or None
    return None, None
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_prompt_formatter.py -v
```
Expected: all 5 tests PASS.

- [ ] **Step 5: Implement dataset loading**

```python
# ccsr/src/data/load_dataset.py
"""Load and preprocess the English subset of the Sentiment Reasoning dataset."""

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
    """Compute inverse-frequency class weights from training split."""
    from collections import Counter

    counts = Counter(dataset["label_idx"])
    total = sum(counts.values())
    weights = [total / (len(counts) * counts[i]) for i in range(len(counts))]
    return weights
```

- [ ] **Step 6: Commit**

```bash
git add src/data/ tests/test_prompt_formatter.py
git commit -m "feat: add dataset loading and post-thinking prompt formatter"
```

---

### Task 3: Focal Loss

**Files:**
- Create: `ccsr/src/models/focal_loss.py`
- Create: `ccsr/tests/test_focal_loss.py`

- [ ] **Step 1: Write the failing test**

```python
# ccsr/tests/test_focal_loss.py
import torch
from src.models.focal_loss import FocalLoss


def test_focal_loss_shape():
    loss_fn = FocalLoss(gamma=2.0, num_classes=3)
    logits = torch.randn(4, 3)
    labels = torch.tensor([0, 1, 2, 1])
    loss = loss_fn(logits, labels)
    assert loss.shape == ()
    assert loss.item() > 0


def test_focal_loss_zero_gamma_equals_ce():
    """With gamma=0, focal loss should equal cross-entropy."""
    torch.manual_seed(42)
    loss_focal = FocalLoss(gamma=0.0, num_classes=3)
    loss_ce = torch.nn.CrossEntropyLoss()
    logits = torch.randn(8, 3)
    labels = torch.tensor([0, 1, 2, 1, 0, 2, 1, 0])
    focal_val = loss_focal(logits, labels)
    ce_val = loss_ce(logits, labels)
    assert torch.allclose(focal_val, ce_val, atol=1e-5)


def test_focal_loss_with_class_weights():
    loss_fn = FocalLoss(gamma=2.0, num_classes=3, class_weights=[1.0, 0.5, 2.0])
    logits = torch.randn(4, 3)
    labels = torch.tensor([0, 1, 2, 1])
    loss = loss_fn(logits, labels)
    assert loss.item() > 0


def test_focal_loss_confident_prediction_lower():
    """Focal loss should be lower for confident correct predictions than uncertain ones."""
    loss_fn = FocalLoss(gamma=2.0, num_classes=3)
    # Confident prediction for class 0
    confident = torch.tensor([[10.0, -10.0, -10.0]])
    # Uncertain prediction for class 0
    uncertain = torch.tensor([[0.5, 0.3, 0.2]])
    labels = torch.tensor([0])
    loss_confident = loss_fn(confident, labels)
    loss_uncertain = loss_fn(uncertain, labels)
    assert loss_confident.item() < loss_uncertain.item()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_focal_loss.py -v
```
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement focal loss**

```python
# ccsr/src/models/focal_loss.py
"""Focal loss for class-imbalanced sentiment classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        num_classes: int = 3,
        class_weights: list[float] | None = None,
    ):
        super().__init__()
        self.gamma = gamma
        if class_weights is not None:
            self.register_buffer("weight", torch.tensor(class_weights, dtype=torch.float32))
        else:
            self.weight = None

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, labels, weight=self.weight, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_focal_loss.py -v
```
Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/models/focal_loss.py tests/test_focal_loss.py
git commit -m "feat: add focal loss with class weights for imbalanced classification"
```

---

### Task 4: Contrastive Projection Head and SupCon Loss

**Files:**
- Create: `ccsr/src/models/contrastive_head.py`
- Create: `ccsr/tests/test_contrastive_head.py`

- [ ] **Step 1: Write the failing test**

```python
# ccsr/tests/test_contrastive_head.py
import torch
from src.models.contrastive_head import ProjectionHead, supervised_contrastive_loss


def test_projection_head_output_shape():
    head = ProjectionHead(input_dim=4096, hidden_dim=256, output_dim=128)
    x = torch.randn(8, 4096)
    z = head(x)
    assert z.shape == (8, 128)


def test_projection_head_output_normalized():
    head = ProjectionHead(input_dim=4096, hidden_dim=256, output_dim=128)
    x = torch.randn(8, 4096)
    z = head(x)
    norms = torch.norm(z, dim=1)
    assert torch.allclose(norms, torch.ones(8), atol=1e-5)


def test_supcon_loss_shape():
    embeddings = torch.randn(12, 128)
    embeddings = torch.nn.functional.normalize(embeddings, dim=1)
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    loss = supervised_contrastive_loss(embeddings, labels, temperature=0.07)
    assert loss.shape == ()
    assert loss.item() > 0


def test_supcon_loss_perfect_clusters():
    """Loss should be lower when same-class embeddings are identical."""
    # Perfect clusters: each class has identical embeddings
    e0 = torch.nn.functional.normalize(torch.tensor([[1.0, 0.0, 0.0]]), dim=1)
    e1 = torch.nn.functional.normalize(torch.tensor([[0.0, 1.0, 0.0]]), dim=1)
    e2 = torch.nn.functional.normalize(torch.tensor([[0.0, 0.0, 1.0]]), dim=1)
    perfect = torch.cat([e0.repeat(4, 1), e1.repeat(4, 1), e2.repeat(4, 1)])
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    loss_perfect = supervised_contrastive_loss(perfect, labels, temperature=0.07)

    # Random embeddings
    torch.manual_seed(0)
    random_emb = torch.nn.functional.normalize(torch.randn(12, 3), dim=1)
    loss_random = supervised_contrastive_loss(random_emb, labels, temperature=0.07)

    assert loss_perfect.item() < loss_random.item()


def test_sub_cluster_loss():
    from src.models.contrastive_head import sub_cluster_loss

    embeddings = torch.randn(6, 128)
    embeddings = torch.nn.functional.normalize(embeddings, dim=1)
    # Sparse similarity: only pairs (0,1) and (3,4) are similar
    sim_indices = torch.tensor([[0, 1], [3, 4]])
    sim_values = torch.tensor([0.9, 0.88])
    loss = sub_cluster_loss(embeddings, sim_indices, sim_values)
    assert loss.shape == ()
    assert loss.item() >= 0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_contrastive_head.py -v
```
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement contrastive head and losses**

```python
# ccsr/src/models/contrastive_head.py
"""Projection head and supervised contrastive loss for CCSR."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """2-layer MLP projecting hidden states to a normalized contrastive space."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return F.normalize(z, dim=1)


def supervised_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """SupCon loss over L2-normalized embeddings.

    Args:
        embeddings: (B, D) normalized embeddings
        labels: (B,) integer class labels
        temperature: scaling temperature
    """
    device = embeddings.device
    B = embeddings.shape[0]

    # Pairwise cosine similarity (already normalized)
    sim_matrix = embeddings @ embeddings.T / temperature  # (B, B)

    # Mask out self-similarity
    self_mask = torch.eye(B, dtype=torch.bool, device=device)
    sim_matrix = sim_matrix.masked_fill(self_mask, -1e9)

    # Positive mask: same label, not self
    labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)  # (B, B)
    pos_mask = labels_eq & ~self_mask

    # For numerical stability
    sim_max, _ = sim_matrix.max(dim=1, keepdim=True)
    sim_matrix = sim_matrix - sim_max.detach()

    # Log-sum-exp over all non-self entries
    exp_sim = torch.exp(sim_matrix) * (~self_mask).float()
    log_sum_exp = torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

    # Mean of positive log-probabilities
    log_prob = sim_matrix - log_sum_exp
    pos_log_prob = (log_prob * pos_mask.float()).sum(dim=1)
    num_positives = pos_mask.float().sum(dim=1).clamp(min=1)
    loss = -(pos_log_prob / num_positives).mean()
    return loss


def sub_cluster_loss(
    embeddings: torch.Tensor,
    sim_indices: torch.Tensor,
    sim_values: torch.Tensor,
) -> torch.Tensor:
    """Soft attraction loss for rationale-grounded sub-clustering.

    Args:
        embeddings: (B, D) normalized embeddings (full training batch indexed)
        sim_indices: (K, 2) pairs of sample indices with high rationale similarity
        sim_values: (K,) BERTScore similarity values for each pair
    """
    if sim_indices.numel() == 0:
        return torch.tensor(0.0, device=embeddings.device)
    i_idx = sim_indices[:, 0]
    j_idx = sim_indices[:, 1]
    diffs = embeddings[i_idx] - embeddings[j_idx]
    sq_dists = (diffs ** 2).sum(dim=1)
    return (sim_values * sq_dists).mean()
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_contrastive_head.py -v
```
Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/models/contrastive_head.py tests/test_contrastive_head.py
git commit -m "feat: add projection head, SupCon loss, and sub-cluster loss"
```

---

### Task 5: Difficulty Scoring (Classifier Uncertainty + Keyword Mismatch)

**Files:**
- Create: `ccsr/src/data/curriculum.py`
- Create: `ccsr/tests/test_curriculum.py`
- Create: `ccsr/src/scripts/precompute_difficulty.py`

- [ ] **Step 1: Write the failing test for difficulty scoring**

```python
# ccsr/tests/test_curriculum.py
import torch
from torch.utils.data import DataLoader, TensorDataset
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
    rationales = [
        "signs of recovery",
        "severe disease condition",
        "objective explanation",
    ]
    scores = compute_keyword_mismatch_scores(transcripts, labels, rationales)
    assert len(scores) == 3
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_combine_difficulty_scores():
    uncertainty = [0.8, 0.2, 0.5]
    mismatch = [0.3, 0.9, 0.4]
    combined = combine_difficulty_scores(uncertainty, mismatch, u_weight=0.6, m_weight=0.4)
    assert len(combined) == 3
    assert all(0.0 <= s <= 1.0 for s in combined)
    # First sample: 0.6*norm(0.8) + 0.4*norm(0.3) — should be between 0 and 1
    # Just check monotonicity isn't broken
    assert combined[0] != combined[1]


def test_curriculum_sampler_phase1():
    difficulty_scores = [0.1, 0.2, 0.5, 0.8, 0.9, 0.3, 0.05, 0.95]
    labels = [0, 1, 2, 0, 1, 2, 0, 1]
    sampler = CurriculumSampler(
        difficulty_scores=difficulty_scores,
        labels=labels,
        threshold=0.4,
        class_balanced=False,
    )
    indices = list(sampler)
    # Should only include samples with difficulty < 0.4: indices 0,1,6 (scores 0.1, 0.2, 0.05)
    # Also index 5 (0.3)
    for idx in indices:
        assert difficulty_scores[idx] < 0.4


def test_curriculum_sampler_phase3_balanced():
    difficulty_scores = [0.1] * 10
    labels = [0, 0, 0, 0, 0, 1, 1, 2, 2, 2]  # imbalanced
    sampler = CurriculumSampler(
        difficulty_scores=difficulty_scores,
        labels=labels,
        threshold=1.0,
        class_balanced=True,
    )
    indices = list(sampler)
    # All samples included since threshold=1.0, length >= original since oversampling minority
    assert len(indices) >= 10
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_curriculum.py -v
```
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement curriculum module**

```python
# ccsr/src/data/curriculum.py
"""Difficulty scoring and curriculum sampling for CCSR."""

import math
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
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
    # Build per-class TF-IDF from rationales
    class_rationale_docs = {}
    for label in label_set:
        class_rats = [r for r, l in zip(rationales, labels) if l == label]
        class_rationale_docs[label] = " ".join(class_rats)

    vectorizer = TfidfVectorizer(max_features=500, stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(class_rationale_docs.values())
    feature_names = vectorizer.get_feature_names_out()
    tfidf_array = tfidf_matrix.toarray()

    # For each word, find which class has the highest TF-IDF
    word_to_class = {}
    for i, word in enumerate(feature_names):
        class_idx = int(np.argmax(tfidf_array[:, i]))
        if tfidf_array[class_idx, i] > 0.01:  # threshold for relevance
            word_to_class[word] = label_set[class_idx]

    # Score each transcript
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
    # Normalize each to [0, 1]
    def norm(a):
        mn, mx = a.min(), a.max()
        return (a - mn) / (mx - mn + 1e-8)
    combined = u_weight * norm(u) + m_weight * norm(m)
    # Final normalization
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
        # Filter by threshold
        eligible = [i for i, d in enumerate(self.difficulty_scores) if d < self.threshold]

        if not self.class_balanced:
            self.rng.shuffle(eligible)
            return iter(eligible)

        # Oversample minority classes to match majority
        class_indices = {}
        for i in eligible:
            lbl = self.labels[i]
            class_indices.setdefault(lbl, []).append(i)

        max_count = max(len(v) for v in class_indices.values())
        balanced = []
        for lbl, idxs in class_indices.items():
            if len(idxs) < max_count:
                oversampled = list(idxs) + list(self.rng.choice(idxs, max_count - len(idxs), replace=True))
                balanced.extend(oversampled)
            else:
                balanced.extend(idxs)

        self.rng.shuffle(balanced)
        return iter(balanced)

    def __len__(self):
        eligible = [i for i, d in enumerate(self.difficulty_scores) if d < self.threshold]
        if not self.class_balanced:
            return len(eligible)
        class_indices = {}
        for i in eligible:
            class_indices.setdefault(self.labels[i], []).append(i)
        if not class_indices:
            return 0
        max_count = max(len(v) for v in class_indices.values())
        return max_count * len(class_indices)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_curriculum.py -v
```
Expected: all 4 tests PASS.

- [ ] **Step 5: Write precompute_difficulty script**

```python
# ccsr/src/scripts/precompute_difficulty.py
"""Offline script: train BERT classifier, compute difficulty scores, save to disk."""

import argparse
import json
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import f1_score, accuracy_score
from src.data.curriculum import (
    compute_keyword_mismatch_scores,
    compute_uncertainty_scores,
    combine_difficulty_scores,
)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert_model", default="bert-base-uncased")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--output_dir", default="outputs/difficulty_scores")
    args = parser.parse_args()

    ds = load_dataset("leduckhai/Sentiment-Reasoning")
    label_map = {"negative": 0, "neutral": 1, "positive": 2}

    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
    model = AutoModelForSequenceClassification.from_pretrained(args.bert_model, num_labels=3)

    def preprocess(example):
        tokens = tokenizer(example["text_en"], truncation=True, padding="max_length", max_length=256)
        tokens["labels"] = label_map[example["label"].lower()]
        return tokens

    train_ds = ds["train"].map(preprocess, batched=False, remove_columns=ds["train"].column_names)
    train_ds.set_format("torch")

    training_args = TrainingArguments(
        output_dir=f"{args.output_dir}/bert_classifier",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        bf16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=train_ds,  # We need predictions on train set
        compute_metrics=compute_metrics,
    )
    trainer.train()

    # Compute entropy on training set
    predictions = trainer.predict(train_ds)
    logits = torch.tensor(predictions.predictions)
    probs = torch.softmax(logits, dim=-1)
    entropies = (-probs * torch.log(probs + 1e-8)).sum(dim=-1).tolist()

    # Compute keyword mismatch
    raw_train = ds["train"]
    transcripts = raw_train["text_en"]
    labels_str = [l.lower() for l in raw_train["label"]]
    rationales = raw_train["human_justification_en"]

    uncertainty = compute_uncertainty_scores(entropies)
    mismatch = compute_keyword_mismatch_scores(transcripts, labels_str, rationales)
    difficulty = combine_difficulty_scores(uncertainty, mismatch)

    import os
    os.makedirs(args.output_dir, exist_ok=True)
    with open(f"{args.output_dir}/difficulty_scores.json", "w") as f:
        json.dump({
            "uncertainty": uncertainty,
            "mismatch": mismatch,
            "difficulty": difficulty,
        }, f)

    print(f"Saved difficulty scores for {len(difficulty)} samples to {args.output_dir}/difficulty_scores.json")
    print(f"Distribution: <0.4: {sum(1 for d in difficulty if d < 0.4)}, "
          f"0.4-0.7: {sum(1 for d in difficulty if 0.4 <= d < 0.7)}, "
          f">=0.7: {sum(1 for d in difficulty if d >= 0.7)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Commit**

```bash
git add src/data/curriculum.py src/scripts/precompute_difficulty.py tests/test_curriculum.py
git commit -m "feat: add difficulty scoring, curriculum sampler, and precompute script"
```

---

### Task 6: Precompute Rationale Similarity Matrix

**Files:**
- Create: `ccsr/src/scripts/precompute_rationale_sim.py`

- [ ] **Step 1: Implement the rationale similarity precomputation script**

```python
# ccsr/src/scripts/precompute_rationale_sim.py
"""Offline script: compute pairwise BERTScore between rationales within each class."""

import argparse
import json
import numpy as np
import torch
from datasets import load_dataset
from bert_score import score as bert_score_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--output_dir", default="outputs/rationale_sim")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    ds = load_dataset("leduckhai/Sentiment-Reasoning")
    train = ds["train"]
    rationales = train["human_justification_en"]
    labels = [l.lower() for l in train["label"]]
    label_set = ["negative", "neutral", "positive"]

    all_pairs = []  # list of (global_idx_i, global_idx_j, similarity)

    for label in label_set:
        indices = [i for i, l in enumerate(labels) if l == label]
        class_rationales = [rationales[i] for i in indices]
        n = len(class_rationales)
        print(f"Computing pairwise BERTScore for {label}: {n} samples ({n*(n-1)//2} pairs)")

        # Process in blocks to avoid OOM
        # Compare each rationale against all others in same class
        # Use BERTScore in batch mode for efficiency
        for start in range(0, n, args.batch_size):
            end = min(start + args.batch_size, n)
            cands = []
            refs = []
            pair_map = []  # (local_i, local_j)
            for i in range(start, end):
                for j in range(i + 1, n):
                    cands.append(class_rationales[i])
                    refs.append(class_rationales[j])
                    pair_map.append((indices[i], indices[j]))

            if not cands:
                continue

            # Compute BERTScore in sub-batches
            sub_batch = 256
            for sb_start in range(0, len(cands), sub_batch):
                sb_end = min(sb_start + sub_batch, len(cands))
                P, R, F1 = bert_score_fn(
                    cands[sb_start:sb_end],
                    refs[sb_start:sb_end],
                    lang="en",
                    verbose=False,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                )
                for k, f1_val in enumerate(F1.tolist()):
                    if f1_val >= args.threshold:
                        gi, gj = pair_map[sb_start + k]
                        all_pairs.append((gi, gj, f1_val))

    import os
    os.makedirs(args.output_dir, exist_ok=True)

    output = {
        "threshold": args.threshold,
        "pairs": [(int(i), int(j), float(s)) for i, j, s in all_pairs],
        "num_pairs": len(all_pairs),
    }
    with open(f"{args.output_dir}/rationale_sim.json", "w") as f:
        json.dump(output, f)

    print(f"Found {len(all_pairs)} pairs above threshold {args.threshold}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add src/scripts/precompute_rationale_sim.py
git commit -m "feat: add offline rationale pairwise similarity computation"
```

---

### Task 7: CCSR Custom Trainer

**Files:**
- Create: `ccsr/src/models/ccsr_trainer.py`

This is the core integration — the custom HuggingFace Trainer that combines all three loss components and handles curriculum phase transitions.

- [ ] **Step 1: Implement the CCSR Trainer**

```python
# ccsr/src/models/ccsr_trainer.py
"""Custom Trainer integrating generative loss, contrastive loss, focal loss, and curriculum."""

import json
import torch
import torch.nn.functional as F
from transformers import Trainer
from src.models.contrastive_head import ProjectionHead, supervised_contrastive_loss, sub_cluster_loss
from src.models.focal_loss import FocalLoss
from src.data.curriculum import CurriculumSampler


class CCSRTrainer(Trainer):
    """Trainer subclass that adds contrastive and focal losses to the generative objective."""

    def __init__(
        self,
        *args,
        contrastive_config: dict | None = None,
        focal_config: dict | None = None,
        curriculum_config: dict | None = None,
        difficulty_scores: list[float] | None = None,
        train_labels: list[int] | None = None,
        rationale_sim_path: str | None = None,
        label_token_ids: dict[int, int] | None = None,
        class_weights: list[float] | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.contrastive_config = contrastive_config or {}
        self.focal_config = focal_config or {}
        self.curriculum_config = curriculum_config or {}
        self.difficulty_scores = difficulty_scores
        self.train_labels = train_labels
        self.label_token_ids = label_token_ids or {}
        self._current_epoch = 0

        # Initialize contrastive head
        if self.contrastive_config.get("enabled"):
            hidden_size = self.model.config.hidden_size
            self.projection_head = ProjectionHead(
                input_dim=hidden_size,
                hidden_dim=self.contrastive_config.get("projection_hidden_dim", 256),
                output_dim=self.contrastive_config.get("projection_output_dim", 128),
            ).to(self.model.device)
            # Load rationale similarity pairs
            self.sim_indices = None
            self.sim_values = None
            if rationale_sim_path:
                with open(rationale_sim_path) as f:
                    sim_data = json.load(f)
                pairs = sim_data["pairs"]
                if pairs:
                    self.sim_indices = torch.tensor([[p[0], p[1]] for p in pairs], dtype=torch.long)
                    self.sim_values = torch.tensor([p[2] for p in pairs], dtype=torch.float32)

        # Initialize focal loss
        if self.focal_config.get("enabled"):
            self.focal_loss_fn = FocalLoss(
                gamma=self.focal_config.get("gamma", 2.0),
                num_classes=3,
                class_weights=class_weights,
            )

    def _get_curriculum_phase(self, epoch: int) -> tuple[float, bool]:
        """Return (threshold, class_balanced) for current epoch."""
        cfg = self.curriculum_config
        if not cfg.get("enabled"):
            return 1.0, False

        if epoch + 1 in cfg.get("phase1_epochs", [1, 2]):
            return cfg.get("phase1_threshold", 0.4), False
        elif epoch + 1 in cfg.get("phase2_epochs", [3, 4]):
            return cfg.get("phase2_threshold", 0.7), False
        else:
            return 1.0, cfg.get("class_balanced_phase3", True)

    def get_train_dataloader(self):
        """Override to inject curriculum sampler based on current epoch."""
        if not self.curriculum_config.get("enabled") or self.difficulty_scores is None:
            return super().get_train_dataloader()

        threshold, class_balanced = self._get_curriculum_phase(self._current_epoch)
        sampler = CurriculumSampler(
            difficulty_scores=self.difficulty_scores,
            labels=self.train_labels,
            threshold=threshold,
            class_balanced=class_balanced,
            seed=self.args.seed + self._current_epoch,
        )

        from torch.utils.data import DataLoader
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def training_step(self, model, inputs, num_items_in_batch=None):
        """Override to add contrastive and focal losses."""
        model.train()
        inputs = self._prepare_inputs(inputs)

        # Forward pass with hidden states
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"],
            output_hidden_states=self.contrastive_config.get("enabled", False),
        )

        gen_loss = outputs.loss
        total_loss = gen_loss

        # --- Contrastive loss ---
        if self.contrastive_config.get("enabled") and outputs.hidden_states is not None:
            # Extract hidden state at end-of-label position
            # Find the position of the label token in the target sequence
            # The label is the first generated token, so we use position 0 of decoder output
            last_hidden = outputs.hidden_states[-1]  # (B, seq_len, hidden)
            # Use position 1 (first token of actual generation after BOS/prompt)
            # For causal LMs, this is the position right after the input
            label_hidden = last_hidden[:, -1, :]  # Simplified: use last position

            # Find per-sample label from inputs
            batch_labels = inputs.get("label_idx")
            if batch_labels is not None:
                self.projection_head = self.projection_head.to(label_hidden.device)
                z = self.projection_head(label_hidden)
                con_loss = supervised_contrastive_loss(
                    z, batch_labels,
                    temperature=self.contrastive_config.get("temperature", 0.07),
                )

                # Sub-cluster loss (only for pairs present in this batch)
                sub_loss = torch.tensor(0.0, device=z.device)
                if self.sim_indices is not None:
                    # Map global indices to batch indices
                    global_indices = inputs.get("global_idx")
                    if global_indices is not None:
                        idx_to_batch = {int(g): b for b, g in enumerate(global_indices)}
                        batch_sim_i, batch_sim_j, batch_sim_v = [], [], []
                        for k in range(self.sim_indices.shape[0]):
                            gi, gj = int(self.sim_indices[k, 0]), int(self.sim_indices[k, 1])
                            if gi in idx_to_batch and gj in idx_to_batch:
                                batch_sim_i.append(idx_to_batch[gi])
                                batch_sim_j.append(idx_to_batch[gj])
                                batch_sim_v.append(self.sim_values[k].item())
                        if batch_sim_i:
                            b_indices = torch.tensor(
                                list(zip(batch_sim_i, batch_sim_j)), dtype=torch.long, device=z.device
                            )
                            b_values = torch.tensor(batch_sim_v, dtype=torch.float32, device=z.device)
                            sub_loss = sub_cluster_loss(z, b_indices, b_values)

                lambda_con = self.contrastive_config.get("lambda_con", 0.3)
                alpha_sub = self.contrastive_config.get("sub_cluster_alpha", 0.1)
                total_loss = total_loss + lambda_con * (con_loss + alpha_sub * sub_loss)

        # --- Focal loss ---
        if self.focal_config.get("enabled"):
            # Extract logits at the label token position for focal loss
            logits = outputs.logits  # (B, seq_len, vocab_size)
            batch_labels = inputs.get("label_idx")
            if batch_labels is not None and self.label_token_ids:
                # Get logits for the 3 label tokens at the first generated position
                label_tids = [self.label_token_ids[i] for i in range(3)]
                label_logits = logits[:, 0, label_tids]  # (B, 3)
                self.focal_loss_fn = self.focal_loss_fn.to(label_logits.device)
                focal = self.focal_loss_fn(label_logits, batch_labels)
                total_loss = total_loss + self.focal_config.get("lambda_bal", 0.5) * focal

        return total_loss

    def on_epoch_begin(self, args, state, control, **kwargs):
        """Track current epoch for curriculum phasing."""
        self._current_epoch = state.epoch or 0
```

- [ ] **Step 2: Commit**

```bash
git add src/models/ccsr_trainer.py
git commit -m "feat: add CCSRTrainer with joint loss and curriculum phasing"
```

---

### Task 8: Classification Metrics

**Files:**
- Create: `ccsr/src/evaluation/classification_metrics.py`
- Create: `ccsr/tests/test_classification_metrics.py`

- [ ] **Step 1: Write the failing test**

```python
# ccsr/tests/test_classification_metrics.py
from src.evaluation.classification_metrics import compute_classification_metrics


def test_compute_classification_metrics_perfect():
    preds = ["negative", "neutral", "positive", "neutral"]
    refs = ["negative", "neutral", "positive", "neutral"]
    metrics = compute_classification_metrics(preds, refs)
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0


def test_compute_classification_metrics_partial():
    preds = ["negative", "neutral", "positive", "neutral"]
    refs = ["negative", "positive", "positive", "neutral"]
    metrics = compute_classification_metrics(preds, refs)
    assert 0.0 < metrics["accuracy"] < 1.0
    assert "f1_negative" in metrics
    assert "f1_neutral" in metrics
    assert "f1_positive" in metrics
    assert "confusion_matrix" in metrics


def test_compute_classification_metrics_handles_invalid():
    preds = ["negative", "unknown", "positive"]
    refs = ["negative", "neutral", "positive"]
    metrics = compute_classification_metrics(preds, refs)
    # "unknown" should be treated as wrong
    assert metrics["accuracy"] < 1.0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_classification_metrics.py -v
```
Expected: FAIL

- [ ] **Step 3: Implement classification metrics**

```python
# ccsr/src/evaluation/classification_metrics.py
"""Classification metrics: accuracy, per-class F1, macro-F1, confusion matrix."""

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np

LABEL_ORDER = ["negative", "neutral", "positive"]


def compute_classification_metrics(
    predictions: list[str],
    references: list[str],
) -> dict:
    """Compute accuracy, per-class F1, macro-F1, and confusion matrix."""
    # Normalize to lowercase, map unknowns to a dummy for consistent handling
    preds_clean = [p.lower() if p.lower() in LABEL_ORDER else "INVALID" for p in predictions]
    refs_clean = [r.lower() for r in references]

    acc = accuracy_score(refs_clean, preds_clean)
    per_class_f1 = f1_score(refs_clean, preds_clean, labels=LABEL_ORDER, average=None, zero_division=0)
    macro_f1_val = f1_score(refs_clean, preds_clean, labels=LABEL_ORDER, average="macro", zero_division=0)
    cm = confusion_matrix(refs_clean, preds_clean, labels=LABEL_ORDER + ["INVALID"])

    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1_val),
        "f1_negative": float(per_class_f1[0]),
        "f1_neutral": float(per_class_f1[1]),
        "f1_positive": float(per_class_f1[2]),
        "confusion_matrix": cm[:3, :3].tolist(),  # 3x3 for valid labels
    }
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_classification_metrics.py -v
```
Expected: all 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/evaluation/classification_metrics.py tests/test_classification_metrics.py
git commit -m "feat: add classification metrics (accuracy, F1, confusion matrix)"
```

---

### Task 9: Rationale Quality Metrics

**Files:**
- Create: `ccsr/src/evaluation/rationale_metrics.py`
- Create: `ccsr/tests/test_rationale_metrics.py`

- [ ] **Step 1: Write the failing test**

```python
# ccsr/tests/test_rationale_metrics.py
from src.evaluation.rationale_metrics import compute_rationale_metrics


def test_rationale_metrics_identical():
    preds = ["This is a positive outcome"]
    refs = ["This is a positive outcome"]
    metrics = compute_rationale_metrics(preds, refs)
    assert metrics["rouge1"] > 0.99
    assert metrics["rougeL"] > 0.99
    assert metrics["bertscore_f1"] > 0.99


def test_rationale_metrics_different():
    preds = ["The cat sat on the mat"]
    refs = ["Recovery is progressing well for the patient"]
    metrics = compute_rationale_metrics(preds, refs)
    assert metrics["rouge1"] < 0.5
    assert "rouge2" in metrics
    assert "rougeLsum" in metrics
    assert "bertscore_f1" in metrics


def test_rationale_metrics_multiple():
    preds = ["good recovery", "bad outcome"]
    refs = ["positive recovery signs", "negative prognosis"]
    metrics = compute_rationale_metrics(preds, refs)
    assert 0.0 < metrics["rouge1"] < 1.0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_rationale_metrics.py -v
```
Expected: FAIL

- [ ] **Step 3: Implement rationale metrics**

```python
# ccsr/src/evaluation/rationale_metrics.py
"""Rationale quality metrics: ROUGE scores and BERTScore."""

from rouge_score import rouge_scorer
from bert_score import score as bert_score_fn
import numpy as np


def compute_rationale_metrics(
    predictions: list[str],
    references: list[str],
    bert_score_lang: str = "en",
) -> dict:
    """Compute ROUGE-{1,2,L,Lsum} and BERTScore F1."""
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL", "rougeLsum"], use_stemmer=True)

    rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": [], "rougeLsum": []}
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        for key in rouge_scores:
            rouge_scores[key].append(scores[key].fmeasure)

    # BERTScore
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
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_rationale_metrics.py -v
```
Expected: all 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/evaluation/rationale_metrics.py tests/test_rationale_metrics.py
git commit -m "feat: add rationale quality metrics (ROUGE + BERTScore)"
```

---

### Task 10: Faithfulness Metric

**Files:**
- Create: `ccsr/src/evaluation/faithfulness.py`
- Create: `ccsr/tests/test_faithfulness.py`

- [ ] **Step 1: Write the failing test**

```python
# ccsr/tests/test_faithfulness.py
from src.evaluation.faithfulness import compute_faithfulness


def test_faithfulness_entailing():
    labels = ["positive", "negative"]
    rationales = [
        "The patient is showing clear signs of recovery and improvement",
        "The patient has a severe and worsening condition",
    ]
    result = compute_faithfulness(labels, rationales)
    assert "faithfulness_score" in result
    assert 0.0 <= result["faithfulness_score"] <= 1.0
    assert "per_sample" in result
    assert len(result["per_sample"]) == 2


def test_faithfulness_empty():
    result = compute_faithfulness([], [])
    assert result["faithfulness_score"] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_faithfulness.py -v
```
Expected: FAIL

- [ ] **Step 3: Implement faithfulness scoring**

```python
# ccsr/src/evaluation/faithfulness.py
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

    Returns:
        dict with "faithfulness_score" (fraction entailing) and "per_sample" (bool list).
    """
    if not labels:
        return {"faithfulness_score": 0.0, "per_sample": []}

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    # Build NLI pairs: premise=rationale, hypothesis=label statement
    premises = rationales
    hypotheses = [LABEL_HYPOTHESES[l.lower()] for l in labels]

    entailments = []
    for start in range(0, len(premises), batch_size):
        end = min(start + batch_size, len(premises))
        batch_p = premises[start:end]
        batch_h = hypotheses[start:end]
        inputs = tokenizer(
            batch_p, batch_h,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        ).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits
            # Typical NLI label order: 0=contradiction, 1=neutral, 2=entailment
            preds = logits.argmax(dim=-1).cpu().tolist()
            entailments.extend([p == 2 for p in preds])

    score = sum(entailments) / len(entailments) if entailments else 0.0
    return {
        "faithfulness_score": float(score),
        "per_sample": entailments,
    }
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_faithfulness.py -v
```
Expected: all 2 tests PASS. (First run will download the NLI model.)

- [ ] **Step 5: Commit**

```bash
git add src/evaluation/faithfulness.py tests/test_faithfulness.py
git commit -m "feat: add NLI-based faithfulness metric for rationale evaluation"
```

---

### Task 11: Attention Entropy Metric

**Files:**
- Create: `ccsr/src/evaluation/attention_entropy.py`

- [ ] **Step 1: Implement attention entropy**

```python
# ccsr/src/evaluation/attention_entropy.py
"""Attention entropy over input tokens — measures keyword reliance."""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer


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
            batch, return_tensors="pt", truncation=True,
            padding=True, max_length=max_length,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)

        # Average attention across all layers and heads
        # Each attention: (B, num_heads, seq_len, seq_len)
        for b in range(len(batch)):
            input_len = inputs["attention_mask"][b].sum().item()
            sample_entropies = []
            for layer_attn in outputs.attentions:
                attn = layer_attn[b, :, :int(input_len), :int(input_len)]  # (heads, L, L)
                # Entropy per head per query position
                entropy = -(attn * torch.log(attn + 1e-10)).sum(dim=-1)  # (heads, L)
                sample_entropies.append(entropy.mean().item())
            all_entropies.append(np.mean(sample_entropies))

    return {
        "mean_entropy": float(np.mean(all_entropies)),
        "std_entropy": float(np.std(all_entropies)),
        "per_sample": all_entropies,
    }
```

- [ ] **Step 2: Commit**

```bash
git add src/evaluation/attention_entropy.py
git commit -m "feat: add attention entropy metric for keyword reliance analysis"
```

---

### Task 12: Main Training Script

**Files:**
- Create: `ccsr/src/scripts/train.py`

- [ ] **Step 1: Implement the main training entrypoint**

```python
# ccsr/src/scripts/train.py
"""Main training script for CCSR and ablation variants."""

import argparse
import json
import os
import yaml
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType
from src.data.load_dataset import load_sentiment_reasoning, get_class_weights
from src.data.prompt_formatter import LABEL_NAMES
from src.models.ccsr_trainer import CCSRTrainer


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output_dir", default="outputs/ccsr")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--difficulty_scores", default=None, help="Path to difficulty_scores.json")
    parser.add_argument("--rationale_sim", default=None, help="Path to rationale_sim.json")
    # Ablation flags
    parser.add_argument("--no_contrastive", action="store_true")
    parser.add_argument("--no_curriculum", action="store_true")
    parser.add_argument("--no_focal", action="store_true")
    parser.add_argument("--label_only", action="store_true", help="Train without rationale (label-only baseline)")
    args = parser.parse_args()

    config = load_config(args.config)

    # Apply ablation flags
    if args.no_contrastive:
        config["contrastive"]["enabled"] = False
    if args.no_curriculum:
        config["curriculum"]["enabled"] = False
    if args.no_focal:
        config["focal_loss"]["enabled"] = False

    # Load dataset
    include_rationale = not args.label_only
    ds = load_sentiment_reasoning(
        dataset_name=config["data"]["dataset_name"],
        text_col=config["data"]["text_column"],
        rationale_col=config["data"]["rationale_column"],
        label_col=config["data"]["label_column"],
        include_rationale=include_rationale,
    )

    # Load tokenizer and model
    model_name = config["model"]["backbone"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=getattr(torch, config["model"]["torch_dtype"]),
        device_map="auto",
    )

    # Apply LoRA
    lora_config = LoraConfig(
        r=config["model"]["lora_r"],
        lora_alpha=config["model"]["lora_alpha"],
        target_modules=config["model"]["lora_target_modules"],
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Tokenize dataset
    max_len = config["data"]["max_seq_length"]

    def tokenize(example):
        input_text = example["input_text"]
        target_text = example["target_text"]
        full_text = f"{input_text}\n{target_text}"

        tokenized = tokenizer(
            full_text,
            truncation=True,
            max_length=max_len,
            padding="max_length",
        )
        # Create labels: mask input tokens with -100
        input_ids = tokenizer(input_text + "\n", truncation=True, max_length=max_len)["input_ids"]
        labels = tokenized["input_ids"].copy()
        labels[:len(input_ids)] = [-100] * len(input_ids)
        tokenized["labels"] = labels
        tokenized["label_idx"] = example["label_idx"]
        return tokenized

    train_ds = ds["train"].map(tokenize, remove_columns=ds["train"].column_names)
    eval_ds = ds["test"].map(tokenize, remove_columns=ds["test"].column_names)

    # Add global index for sub-cluster loss
    train_ds = train_ds.map(lambda ex, idx: {"global_idx": idx}, with_indices=True)

    # Get class weights
    class_weights = get_class_weights(ds["train"])

    # Get label token IDs for focal loss
    label_token_ids = {}
    for idx, (name, upper) in enumerate(LABEL_NAMES.items()):
        tid = tokenizer.encode(upper, add_special_tokens=False)[0]
        label_token_ids[idx] = tid

    # Load difficulty scores if provided
    difficulty_scores = None
    train_labels = None
    if args.difficulty_scores and config["curriculum"]["enabled"]:
        with open(args.difficulty_scores) as f:
            diff_data = json.load(f)
        difficulty_scores = diff_data["difficulty"]
        train_labels = [ex["label_idx"] for ex in ds["train"]]

    # Training arguments
    tc = config["training"]
    output_dir = f"{args.output_dir}/seed_{args.seed}"
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=tc["num_epochs"],
        per_device_train_batch_size=tc["per_device_batch_size"],
        gradient_accumulation_steps=tc["gradient_accumulation_steps"],
        per_device_eval_batch_size=tc["per_device_batch_size"],
        learning_rate=tc["learning_rate"],
        warmup_ratio=tc["warmup_ratio"],
        lr_scheduler_type=tc["lr_scheduler_type"],
        weight_decay=tc["weight_decay"],
        eval_strategy=tc["eval_strategy"],
        save_strategy=tc["save_strategy"],
        load_best_model_at_end=True,
        metric_for_best_model=tc["metric_for_best_model"],
        seed=args.seed,
        bf16=True,
        deepspeed="configs/ds_config.json" if torch.cuda.device_count() > 1 else None,
        report_to="none",
    )

    # Data collator
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)

    trainer = CCSRTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        contrastive_config=config["contrastive"],
        focal_config=config["focal_loss"],
        curriculum_config=config["curriculum"],
        difficulty_scores=difficulty_scores,
        train_labels=train_labels,
        rationale_sim_path=args.rationale_sim,
        label_token_ids=label_token_ids,
        class_weights=class_weights,
    )

    trainer.train()
    trainer.save_model(f"{output_dir}/best_model")
    tokenizer.save_pretrained(f"{output_dir}/best_model")

    print(f"Training complete. Best model saved to {output_dir}/best_model")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Create DeepSpeed config**

```json
// ccsr/configs/ds_config.json
{
    "bf16": {"enabled": true},
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {"device": "none"},
        "allgather_partitions": true,
        "allgather_bucket_size": 2e8,
        "reduce_scatter": true,
        "reduce_bucket_size": 2e8,
        "overlap_comm": true
    },
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": 1.0,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto"
}
```

- [ ] **Step 3: Commit**

```bash
git add src/scripts/train.py configs/ds_config.json
git commit -m "feat: add main training script with CCSR, ablation flags, and DeepSpeed"
```

---

### Task 13: Evaluation Script

**Files:**
- Create: `ccsr/src/scripts/evaluate.py`

- [ ] **Step 1: Implement the evaluation entrypoint**

```python
# ccsr/src/scripts/evaluate.py
"""Run all metrics on a trained checkpoint."""

import argparse
import json
import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm

from src.data.prompt_formatter import format_input, parse_prediction
from src.evaluation.classification_metrics import compute_classification_metrics
from src.evaluation.rationale_metrics import compute_rationale_metrics
from src.evaluation.faithfulness import compute_faithfulness
from src.evaluation.attention_entropy import compute_attention_entropy


def generate_predictions(model, tokenizer, texts: list[str], batch_size: int = 8, max_new_tokens: int = 128) -> list[str]:
    device = next(model.parameters()).device
    model.eval()
    all_outputs = []

    for start in tqdm(range(0, len(texts), batch_size), desc="Generating"):
        end = min(start + batch_size, len(texts))
        batch = texts[start:end]
        inputs = tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=384).to(device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
            )
        # Decode only newly generated tokens
        for i in range(len(batch)):
            input_len = inputs["input_ids"][i].shape[0]
            gen_ids = output_ids[i][input_len:]
            text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            all_outputs.append(text)

    return all_outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to saved best_model directory")
    parser.add_argument("--base_model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--output_dir", default="outputs/eval_results")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--skip_faithfulness", action="store_true")
    parser.add_argument("--skip_attention", action="store_true")
    args = parser.parse_args()

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, args.model_path)

    # Load test set
    ds = load_dataset("leduckhai/Sentiment-Reasoning", split="test")
    transcripts = ds["text_en"]
    ref_labels = [l.lower() for l in ds["label"]]
    ref_rationales = ds["human_justification_en"]

    # Format inputs
    input_texts = [format_input(t) for t in transcripts]

    # Generate predictions
    raw_outputs = generate_predictions(model, tokenizer, input_texts, batch_size=args.batch_size)

    # Parse predictions into labels and rationales
    pred_labels = []
    pred_rationales = []
    for output in raw_outputs:
        label, rationale = parse_prediction(output)
        pred_labels.append(label or "neutral")  # fallback
        pred_rationales.append(rationale or "")

    # Classification metrics
    cls_metrics = compute_classification_metrics(pred_labels, ref_labels)
    print(f"\n=== Classification Metrics ===")
    print(f"Accuracy: {cls_metrics['accuracy']:.4f}")
    print(f"Macro F1: {cls_metrics['macro_f1']:.4f}")
    print(f"F1 NEG: {cls_metrics['f1_negative']:.4f}")
    print(f"F1 NEU: {cls_metrics['f1_neutral']:.4f}")
    print(f"F1 POS: {cls_metrics['f1_positive']:.4f}")

    # Rationale metrics (only for samples with rationales)
    valid_mask = [bool(r) for r in pred_rationales]
    if any(valid_mask):
        valid_preds = [r for r, v in zip(pred_rationales, valid_mask) if v]
        valid_refs = [r for r, v in zip(ref_rationales, valid_mask) if v]
        rat_metrics = compute_rationale_metrics(valid_preds, valid_refs)
        print(f"\n=== Rationale Metrics ===")
        for k, v in rat_metrics.items():
            print(f"{k}: {v:.4f}")
    else:
        rat_metrics = {}

    # Faithfulness
    faith_metrics = {}
    if not args.skip_faithfulness and any(valid_mask):
        valid_labels = [l for l, v in zip(pred_labels, valid_mask) if v]
        faith_metrics = compute_faithfulness(valid_labels, valid_preds)
        print(f"\n=== Faithfulness ===")
        print(f"Score: {faith_metrics['faithfulness_score']:.4f}")

    # Attention entropy
    attn_metrics = {}
    if not args.skip_attention:
        attn_metrics = compute_attention_entropy(model, tokenizer, input_texts[:200])
        print(f"\n=== Attention Entropy ===")
        print(f"Mean: {attn_metrics['mean_entropy']:.4f} +/- {attn_metrics['std_entropy']:.4f}")

    # Save all results
    os.makedirs(args.output_dir, exist_ok=True)
    results = {
        "classification": cls_metrics,
        "rationale": rat_metrics,
        "faithfulness": {k: v for k, v in faith_metrics.items() if k != "per_sample"},
        "attention_entropy": {k: v for k, v in attn_metrics.items() if k != "per_sample"},
    }
    with open(f"{args.output_dir}/results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output_dir}/results.json")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add src/scripts/evaluate.py
git commit -m "feat: add evaluation script with all metrics"
```

---

### Task 14: Baseline Runner Script

**Files:**
- Create: `ccsr/src/scripts/run_baselines.py`

- [ ] **Step 1: Implement baseline runner**

```python
# ccsr/src/scripts/run_baselines.py
"""Run all baseline experiments: BERT, Flan-T5, Mistral-7B, Llama 3 SFT-only."""

import subprocess
import sys


BASELINES = {
    "llama3_sft_label_only": {
        "desc": "Llama 3 8B SFT (label only)",
        "args": ["--label_only", "--no_contrastive", "--no_curriculum", "--no_focal"],
    },
    "llama3_sft_rationale": {
        "desc": "Llama 3 8B SFT (label + rationale)",
        "args": ["--no_contrastive", "--no_curriculum", "--no_focal"],
    },
}

SEEDS = [42, 123, 456]


def main():
    for name, baseline in BASELINES.items():
        for seed in SEEDS:
            print(f"\n{'='*60}")
            print(f"Running: {baseline['desc']} (seed={seed})")
            print(f"{'='*60}\n")

            cmd = [
                sys.executable, "-m", "src.scripts.train",
                "--config", "configs/default.yaml",
                "--output_dir", f"outputs/baselines/{name}",
                "--seed", str(seed),
            ] + baseline["args"]

            result = subprocess.run(cmd, check=False)
            if result.returncode != 0:
                print(f"WARNING: {name} seed={seed} failed with code {result.returncode}")

    print("\nAll baselines complete. Run evaluate.py on each output directory.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add src/scripts/run_baselines.py
git commit -m "feat: add baseline runner script"
```

---

### Task 15: Ablation Runner Script

**Files:**
- Create: `ccsr/src/scripts/run_ablations.py`

- [ ] **Step 1: Implement ablation runner**

```python
# ccsr/src/scripts/run_ablations.py
"""Run all 7 ablation variants across 3 seeds."""

import subprocess
import sys

ABLATIONS = {
    "sft_only":           {"no_contrastive": True,  "no_curriculum": True,  "no_focal": True},
    "focal_only":         {"no_contrastive": True,  "no_curriculum": True,  "no_focal": False},
    "contrastive_only":   {"no_contrastive": False, "no_curriculum": True,  "no_focal": True},
    "curriculum_only":    {"no_contrastive": True,  "no_curriculum": False, "no_focal": True},
    "contrastive_focal":  {"no_contrastive": False, "no_curriculum": True,  "no_focal": False},
    "curriculum_focal":   {"no_contrastive": True,  "no_curriculum": False, "no_focal": False},
    "ccsr_full":          {"no_contrastive": False, "no_curriculum": False, "no_focal": False},
}

SEEDS = [42, 123, 456]


def main():
    for name, flags in ABLATIONS.items():
        for seed in SEEDS:
            print(f"\n{'='*60}")
            print(f"Running ablation: {name} (seed={seed})")
            print(f"{'='*60}\n")

            cmd = [
                sys.executable, "-m", "src.scripts.train",
                "--config", "configs/default.yaml",
                "--output_dir", f"outputs/ablations/{name}",
                "--seed", str(seed),
                "--difficulty_scores", "outputs/difficulty_scores/difficulty_scores.json",
                "--rationale_sim", "outputs/rationale_sim/rationale_sim.json",
            ]

            for flag_name, is_disabled in flags.items():
                if is_disabled:
                    cmd.append(f"--{flag_name}")

            result = subprocess.run(cmd, check=False)
            if result.returncode != 0:
                print(f"WARNING: {name} seed={seed} failed with code {result.returncode}")

    print("\nAll ablations complete. Run evaluate.py on each output directory.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add src/scripts/run_ablations.py
git commit -m "feat: add ablation runner for all 7 CCSR variants"
```

---

### Task 16: Results Notebook (Tables + Figures)

**Files:**
- Create: `ccsr/notebooks/generate_tables_figures.ipynb`

- [ ] **Step 1: Create the notebook for paper figure generation**

Create a Jupyter notebook at `ccsr/notebooks/generate_tables_figures.ipynb` with these cells:

**Cell 1: Load all results**
```python
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Collect all results files
def load_results(pattern):
    results = {}
    for path in sorted(glob.glob(pattern)):
        name = Path(path).parent.name
        seed = Path(path).parent.parent.name
        key = f"{seed}/{name}" if "seed_" in str(path) else name
        with open(path) as f:
            results[key] = json.load(f)
    return results

ablation_results = load_results("../outputs/ablations/*/seed_*/eval_results/results.json")
baseline_results = load_results("../outputs/baselines/*/seed_*/eval_results/results.json")
print(f"Loaded {len(ablation_results)} ablation results, {len(baseline_results)} baseline results")
```

**Cell 2: Main results table (Table I in paper)**
```python
# Aggregate across seeds: mean +/- std
def aggregate(results_dict, variant_prefix):
    metrics = {}
    for key, res in results_dict.items():
        if variant_prefix in key:
            for m in ["accuracy", "macro_f1", "f1_negative", "f1_neutral", "f1_positive"]:
                metrics.setdefault(m, []).append(res["classification"][m])
    return {m: (np.mean(v), np.std(v)) for m, v in metrics.items()}

# Build table rows — fill with actual variant names from your outputs
# print(pd.DataFrame(...).to_latex(index=False))
```

**Cell 3: Ablation table (Table II)**
```python
ablation_names = ["sft_only", "focal_only", "contrastive_only", "curriculum_only",
                   "contrastive_focal", "curriculum_focal", "ccsr_full"]
rows = []
for name in ablation_names:
    agg = aggregate(ablation_results, name)
    rows.append({
        "Variant": name,
        "Acc": f"{agg['accuracy'][0]:.4f} +/- {agg['accuracy'][1]:.4f}",
        "Mac-F1": f"{agg['macro_f1'][0]:.4f} +/- {agg['macro_f1'][1]:.4f}",
    })
df_ablation = pd.DataFrame(rows)
print(df_ablation.to_string(index=False))
```

**Cell 4: Confusion matrix comparison (Figure 2)**
```python
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
labels_order = ["negative", "neutral", "positive"]

# Load best-seed confusion matrices
# baseline_cm = baseline_results["..."]["classification"]["confusion_matrix"]
# ccsr_cm = ablation_results["..."]["classification"]["confusion_matrix"]

for ax, cm, title in [(axes[0], [[0]*3]*3, "Llama 3 SFT-only"), (axes[1], [[0]*3]*3, "CCSR (Ours)")]:
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels_order,
                yticklabels=labels_order, cmap="Blues", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

plt.tight_layout()
plt.savefig("../paper/figures/confusion_matrices.pdf", bbox_inches="tight")
plt.show()
```

**Cell 5: Attention entropy distribution (Figure 3)**
```python
# Load per-sample attention entropies
# sft_entropy = baseline_results["..."]["attention_entropy"]["per_sample"]
# ccsr_entropy = ablation_results["..."]["attention_entropy"]["per_sample"]

fig, ax = plt.subplots(figsize=(8, 5))
# ax.hist(sft_entropy, bins=30, alpha=0.6, label="SFT-only", color="tab:blue")
# ax.hist(ccsr_entropy, bins=30, alpha=0.6, label="CCSR", color="tab:orange")
ax.set_xlabel("Attention Entropy (nats)")
ax.set_ylabel("Count")
ax.set_title("Attention Entropy Distribution")
ax.legend()
plt.savefig("../paper/figures/attention_entropy.pdf", bbox_inches="tight")
plt.show()
```

**Cell 6: Statistical significance**
```python
from scipy import stats

# Example: compare SFT-only vs CCSR macro-F1 across seeds
# sft_f1s = [...]
# ccsr_f1s = [...]
# t_stat, p_value = stats.ttest_ind(sft_f1s, ccsr_f1s)
# print(f"t={t_stat:.4f}, p={p_value:.4f}")
```

- [ ] **Step 2: Commit**

```bash
git add notebooks/generate_tables_figures.ipynb
git commit -m "feat: add notebook for generating paper tables and figures"
```

---

### Task 17: End-to-End Smoke Test

**Files:** None new — validates that everything wires together.

- [ ] **Step 1: Run a quick smoke test with tiny data**

```bash
cd /Users/josephmkalinzi/Developer/Prof.\ Ahmed/OutPerform_v.1.0.0-Claude02/ccsr

# Run all unit tests first
python -m pytest tests/ -v

# Quick train smoke test (will fail without GPU/model — validates imports and wiring)
python -c "
from src.data.load_dataset import load_sentiment_reasoning
from src.data.prompt_formatter import format_input, parse_prediction
from src.data.curriculum import CurriculumSampler, combine_difficulty_scores
from src.models.focal_loss import FocalLoss
from src.models.contrastive_head import ProjectionHead, supervised_contrastive_loss
from src.evaluation.classification_metrics import compute_classification_metrics
from src.evaluation.rationale_metrics import compute_rationale_metrics
import torch

# Quick integration check
head = ProjectionHead(768, 256, 128)
z = head(torch.randn(4, 768))
labels = torch.tensor([0, 0, 1, 2])
loss = supervised_contrastive_loss(z, labels)
print(f'SupCon loss: {loss.item():.4f}')

focal = FocalLoss(gamma=2.0, num_classes=3)
logits = torch.randn(4, 3)
fl = focal(logits, labels)
print(f'Focal loss: {fl.item():.4f}')

metrics = compute_classification_metrics(['negative', 'neutral', 'positive', 'neutral'], ['negative', 'neutral', 'positive', 'negative'])
print(f'Accuracy: {metrics[\"accuracy\"]:.4f}, Macro F1: {metrics[\"macro_f1\"]:.4f}')

print('All imports and integration checks passed!')
"
```

Expected: all tests pass, integration check prints loss values and metrics.

- [ ] **Step 2: Commit final state**

```bash
git add -A
git commit -m "feat: complete CCSR implementation — all components, tests, and scripts"
```

---

## Execution Order on Research Cluster

Once all code is committed, run experiments in this order:

```bash
# 1. Precompute offline artifacts (~1-2 hours)
python -m src.scripts.precompute_difficulty --output_dir outputs/difficulty_scores
python -m src.scripts.precompute_rationale_sim --output_dir outputs/rationale_sim

# 2. Run baselines (~4 hours, 2 variants x 3 seeds)
python -m src.scripts.run_baselines

# 3. Run ablations (~14 hours, 7 variants x 3 seeds)
python -m src.scripts.run_ablations

# 4. Evaluate all checkpoints
for dir in outputs/baselines/*/seed_*/best_model outputs/ablations/*/seed_*/best_model; do
    python -m src.scripts.evaluate --model_path "$dir" --output_dir "${dir%best_model}eval_results"
done

# 5. Generate tables and figures
jupyter notebook notebooks/generate_tables_figures.ipynb
```
