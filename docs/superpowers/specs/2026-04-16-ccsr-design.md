# Contrastive Curriculum Sentiment Reasoning (CCSR) — Design Spec

## Overview

This spec describes the design for a research project that extends Nguyen et al.'s "Sentiment Reasoning for Healthcare" paper. The goal is to outperform their reported metrics on the **English subset** of the Sentiment Reasoning benchmark for both sentiment classification and rationale generation, using a novel framework called **Contrastive Curriculum Sentiment Reasoning (CCSR)**.

---

## Problem Statement

Nguyen et al. introduce Sentiment Reasoning — a task where models predict both a sentiment label (POSITIVE, NEUTRAL, NEGATIVE) and a free-text rationale from medical transcripts. Their best English result is 64.54% accuracy / 64.36% macro-F1 (Mistral-7B with rationale-augmented training).

Their error analysis reveals critical failure modes:

1. **Keyword shortcutting** — words like "helpful" or "diabetes" override contextual reasoning, causing misclassification (Table 9 in the paper)
2. **NEUTRAL confusion** — 23-27% of POSITIVE and NEGATIVE samples are misclassified as NEUTRAL (Figure 2)
3. **No explicit class separation signal** — the generation loss alone doesn't teach the model where class boundaries lie in representation space
4. **Class imbalance** — NEUTRAL dominates at ~50% of the dataset, creating prediction bias
5. **Uniform training** — ambiguous boundary samples are treated identically to clear-cut examples

## Research Contributions

1. A novel **Contrastive Curriculum Sentiment Reasoning (CCSR)** framework that jointly optimizes classification, rationale generation, and sentiment-contrastive alignment
2. A **difficulty-aware curriculum strategy** for medical sentiment that scores sample difficulty using keyword-label mismatch and classifier uncertainty
3. A **rationale-grounded contrastive objective** that creates sub-cluster structure within sentiment classes based on reasoning patterns
4. State-of-the-art results on the English subset of the Sentiment Reasoning benchmark on both classification metrics (accuracy, macro-F1) and rationale quality (ROUGE, BERTScore)

---

## Architecture

### Backbone

**Llama 3 8B-Instruct.** Rationale: one generation ahead of the paper's Mistral-7B, widely reproducible, strong instruction-following for rationale generation.

### Component A: Generative Core (Classification + Rationale)

Standard post-thinking formulation from the paper. Given transcript `w`, the model generates `<LABEL> <RATIONALE>` in a single sequence. Trained with cross-entropy loss `L_gen`.

Prompt template:
```
Input:  "Classify the sentiment and provide a rationale: {transcript}"
Target: "{LABEL} {RATIONALE}"
```

### Component B: Sentiment-Contrastive Representation Module

A lightweight projection head attached to the **last hidden state** of the decoder at the **end-of-label token position** (where the model has committed to a sentiment class but before rationale generation begins).

- **Projection head:** 2-layer MLP (hidden_dim -> 256 -> 128) with ReLU, projecting to a normalized contrastive embedding space
- **Loss:** Supervised Contrastive Loss (SupCon) over the projected embeddings within each training batch
  - Positive pairs: samples sharing the same sentiment label
  - Negative pairs: samples with different labels
- **Rationale-grounded refinement:** Within each class, a secondary soft attraction term pulls together samples whose human rationales have high pairwise BERTScore (>0.85). This creates meaningful sub-clusters (e.g., within NEGATIVE: "disease progression" vs "emotional distress"). The similarity matrix is precomputed offline.

**At inference time, Component B is discarded.** The contrastive training only shapes internal representations during training — zero overhead at test time.

### Component C: Curriculum Scheduler

A training schedule that controls which samples enter each epoch.

**Difficulty scoring (computed once before training):**

1. **Classifier uncertainty:** Train a lightweight BERT-base classifier on the English subset (~3 epochs). For each sample, record the entropy of predicted class probabilities. High entropy = ambiguous sample.
2. **Keyword-label mismatch:** Build a sentiment lexicon from training rationales using TF-IDF (words appearing disproportionately in POS/NEG rationales). Score each transcript by how much its keyword sentiment conflicts with its ground-truth label. High mismatch = deceptive sample.
3. **Combined difficulty score:** `0.6 * normalized_uncertainty + 0.4 * normalized_mismatch`, scaled to [0, 1].

**Curriculum schedule:**

| Phase | Epochs | Difficulty threshold | ~% of data |
|-------|--------|---------------------|------------|
| 1     | 1-2    | < 0.4               | 55-60%     |
| 2     | 3-4    | < 0.7               | ~85%       |
| 3     | 5-7    | All samples          | 100%       |

Phase 3 uses class-balanced sampling with inverse frequency weights.

### Joint Training Objective

```
L_total = L_gen + lambda_con * L_contrastive + lambda_bal * L_focal
```

- `L_gen`: Cross-entropy for sequence generation (label + rationale)
- `L_contrastive`: SupCon loss from Component B (lambda_con = 0.3)
- `L_focal`: Focal loss (gamma=2) applied to the classification token only (lambda_bal = 0.5)
- Lambda values are starting points; tuned via validation macro-F1

---

## Data Pipeline

### Dataset

**English subset of the Sentiment Reasoning dataset (Nguyen et al.)**
- 5,695 train / 2,183 test samples (matching the single-language split from Table 1: 2,844 NEU + 1,694 NEG + 1,157 POS train; 958 NEU + 701 NEG + 524 POS test)
- 3 classes: POSITIVE (~20%), NEUTRAL (~50%), NEGATIVE (~30%)
- Each sample: transcript + sentiment label + human rationale

### Preprocessing (offline, one-time)

1. **Rationale pairwise similarity matrix:** Compute BERTScore between all training rationale pairs within each class. Store as a sparse matrix (threshold > 0.85). Estimated: ~1 hour on single GPU.
2. **Difficulty scoring:** Train BERT-base classifier, record per-sample entropy. Build keyword-label mismatch lexicon via TF-IDF on rationale word frequencies per class. Compute combined scores.
3. **Prompt formatting:** As described in Component A.

---

## Experimental Design

### Baselines

| Model | Type | Purpose |
|-------|------|---------|
| BERT | Encoder | Paper baseline (label only) |
| Flan-T5-base | Enc-Dec | Paper baseline (label + rationale) |
| Mistral-7B | Decoder | Paper baseline (label + rationale) |
| Llama 3 8B + SFT only | Decoder | Our backbone without CCSR (ablation) |
| **Llama 3 8B + CCSR** | **Decoder** | **Our full method** |

### Ablation Study

| Variant | Contrastive | Curriculum | Focal Loss |
|---------|:-----------:|:----------:|:----------:|
| SFT-only (baseline) | - | - | - |
| + Focal | - | - | yes |
| + Contrastive | yes | - | - |
| + Curriculum | - | yes | - |
| + Contrastive + Focal | yes | - | yes |
| + Curriculum + Focal | - | yes | yes |
| **CCSR (full)** | **yes** | **yes** | **yes** |

### Training Configuration

- **Fine-tuning:** LoRA (r=16, alpha=32) on all attention layers
- **Optimizer:** AdamW, lr=2e-4, cosine schedule with 5% warmup
- **Batch size:** 32 (across GPUs, DeepSpeed ZeRO Stage 2)
- **Epochs:** 7 total (curriculum phases as described above)
- **Early stopping:** patience=2 on validation macro-F1, evaluated every epoch
- **Seeds:** 3 random seeds per experiment, report mean +/- std
- **Statistical significance:** Student's t-test, alpha=0.05

### Evaluation Metrics

**Classification (matching the paper):**
- Accuracy, class-wise F1 (NEG, NEU, POS), macro-F1
- Confusion matrix (side-by-side comparison with paper's Figure 2)

**Rationale quality (matching the paper):**
- ROUGE-1, ROUGE-2, ROUGE-L, ROUGE-Lsum
- BERTScore

**New metrics (our contribution):**
- **Faithfulness score:** NLI-based check (DeBERTa-v3 fine-tuned on MultiNLI) measuring % of rationales that entail their predicted label
- **Keyword reliance reduction:** Attention entropy over input tokens — higher entropy = less keyword-fixated. Compare CCSR vs SFT-only.

---

## Expected Results

### Classification (English human transcript)

| Model | Acc. | Mac-F1 | F1-NEG | F1-NEU | F1-POS |
|-------|------|--------|--------|--------|--------|
| Mistral-7B + Rationale (paper) | 64.54 | 64.36 | 67.68 | 63.64 | 61.76 |
| Llama 3 8B + SFT only | ~66.5 | ~66.0 | ~69.0 | ~65.5 | ~63.5 |
| **Llama 3 8B + CCSR (full)** | **~69-71** | **~68-70** | **~72-73** | **~68-69** | **~65-67** |

- +4-6% macro-F1 over the paper's best English result
- POS F1 sees the largest relative gain due to curriculum's focused exposure
- NEUTRAL confusion drops significantly (confusion matrix comparison)
- Ablation: contrastive loss ~60% of gain, curriculum ~25%, focal loss ~15%

### Rationale Quality

| Model | ROUGE-1 | BERTScore | Faithfulness |
|-------|---------|-----------|-------------|
| Mistral-7B + Rationale (paper) | ~0.39 | ~0.81 | not reported |
| Llama 3 8B + SFT only | ~0.41 | ~0.82 | ~78% |
| **Llama 3 8B + CCSR (full)** | **~0.44-0.46** | **~0.84-0.85** | **~86-89%** |

---

## Paper Structure

**Working title:** *"Contrastive Curriculum Learning for Sentiment Reasoning in Healthcare: Addressing Keyword Bias and Class Ambiguity"*

| Section | Content | Est. pages |
|---------|---------|-----------|
| 1. Introduction | Motivation, gap, CCSR overview, contributions | 1.5 |
| 2. Related Work | Healthcare sentiment, CoT distillation, contrastive learning for NLP, curriculum learning for NLP, gap statement | 1.5 |
| 3. Method | Problem formulation (reuse paper's notation), Components A/B/C, joint objective, training procedure | 2.5 |
| 4. Experimental Setup | Dataset, baselines, ablation design, metrics, training config | 1.5 |
| 5. Results and Analysis | Main results, ablation table, confusion matrix comparison, rationale quality + faithfulness, attention entropy analysis, statistical significance | 2.0 |
| 6. Discussion | Why contrastive works here, generalizability, when curriculum helps vs doesn't | 0.5 |
| 7. Conclusion & Future Work | Summary, limitations, next steps | 0.5 |

**Total:** ~10 pages (IEEE two-column format)

**Target venues:** IEEE JBHI, IEEE TAFFC, IEEE/ACM TASLP, or ACL/EMNLP (*ACL format variant)

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| DPO-free contrastive loss on decoder hidden states is architecturally non-trivial | Medium | Extract hidden state at a fixed token position (end-of-label); well-defined extraction point |
| English subset is small (~5K train) — contrastive learning needs batch diversity | High | Use large batch size (32+), ensure each batch has all 3 classes via stratified sampling |
| Curriculum schedule hyperparameters (thresholds) may need tuning | Low | Validate on held-out set; ablation already planned |
| Llama 3 8B may already beat baselines just from model scale | Medium | SFT-only ablation isolates model-scale gains from methodological gains |
| Rationale similarity matrix is expensive for large datasets | Low | English subset is small enough; sparse storage above threshold keeps memory manageable |
