# Contrastive Curriculum Sentiment Reasoning (CCSR)

A novel framework for healthcare sentiment analysis that jointly improves classification accuracy and rationale quality. Extends [Nguyen et al.'s Sentiment Reasoning](https://github.com/leduckhai/Sentiment-Reasoning) work by addressing keyword bias and class ambiguity through contrastive learning and curriculum training.

## Motivation

Current sentiment reasoning models in healthcare suffer from three problems:
1. **Keyword shortcutting** — models fixate on individual terms ("diabetes", "helpful") rather than contextual reasoning
2. **NEUTRAL confusion** — 23-27% of polar samples are misclassified as NEUTRAL
3. **Class imbalance** — NEUTRAL dominates at ~50% of training data

CCSR addresses all three with a unified training framework.

## Method

CCSR augments standard rationale-augmented fine-tuning with three components:

| Component | What it does |
|-----------|-------------|
| **Sentiment-Contrastive Module** | Projects decoder hidden states into a contrastive embedding space (SupCon loss), forcing class separation. Includes rationale-grounded sub-clustering for within-class structure. |
| **Difficulty-Aware Curriculum** | Scores each sample by classifier uncertainty + keyword-label mismatch. Trains easy-to-hard across three phases. |
| **Focal Loss** | Rebalances the class prior to counteract NEUTRAL dominance. |

At inference, only the generative core is active — zero overhead compared to standard fine-tuning.

### Architecture

```
Input Transcript
       │
       ▼
┌──────────────────────┐
│   Llama 3 8B + LoRA  │
│   (Shared Backbone)  │
└──────┬───────────────┘
       │
       ├──► Component A: Generate <LABEL> <RATIONALE>  →  L_gen (cross-entropy)
       │
       ├──► Component B: Project end-of-label hidden    →  L_contrastive (SupCon)
       │    state → 128-d normalized embedding               + L_sub_cluster
       │
       └──► Component C: Curriculum scheduler controls
            which samples enter each epoch
            + L_focal on classification token

L_total = L_gen + 0.3 * L_contrastive + 0.5 * L_focal
```

## Dataset

[Sentiment Reasoning](https://huggingface.co/datasets/leduckhai/Sentiment-Reasoning) English subset — 5,610 train / 2,180 test samples of medical doctor-patient transcripts, each labeled with sentiment (POSITIVE / NEUTRAL / NEGATIVE) and a human-written rationale.

## Project Structure

```
ccsr/
├── configs/
│   ├── default.yaml              # All hyperparameters
│   └── ds_config.json            # DeepSpeed ZeRO Stage 2
├── src/
│   ├── data/                     # Dataset loading, prompt formatting, curriculum
│   ├── models/                   # Focal loss, contrastive head, CCSRTrainer
│   ├── evaluation/               # Classification, ROUGE, BERTScore, faithfulness, attention entropy
│   └── scripts/                  # train, evaluate, precompute, run_baselines, run_ablations
├── tests/                        # 25 unit tests
├── slurm/                        # SLURM job scripts for cluster execution
├── EXECUTION_GUIDE.md            # Detailed cluster run instructions
└── pyproject.toml
paper/
└── main.tex                      # IEEE-format paper draft
docs/
├── superpowers/specs/            # Design specification
└── superpowers/plans/            # Implementation plan
```

## Quick Start

### Install

```bash
cd ccsr
pip install -e ".[dev]"
```

### Run tests

```bash
python -m pytest tests/ -v
```

### Train (single GPU, quick test)

```bash
python -m src.scripts.train \
    --config configs/default.yaml \
    --output_dir outputs/test_run \
    --seed 42 \
    --no_contrastive --no_curriculum --no_focal
```

### Full experiment pipeline (SLURM cluster)

See [EXECUTION_GUIDE.md](ccsr/EXECUTION_GUIDE.md) for step-by-step instructions covering precomputation, baselines, ablations, evaluation, and paper figure generation.

## Ablation Variants

| Variant | Contrastive | Curriculum | Focal |
|---------|:-----------:|:----------:|:-----:|
| SFT-only (baseline) | | | |
| + Focal | | | x |
| + Contrastive | x | | |
| + Curriculum | | x | |
| + Contrastive + Focal | x | | x |
| + Curriculum + Focal | | x | x |
| **CCSR (full)** | **x** | **x** | **x** |

All variants are trained with 3 random seeds (42, 123, 456) and evaluated with Student's t-test at alpha=0.05.

## Evaluation Metrics

**Classification** (matching Nguyen et al.): accuracy, per-class F1, macro-F1, confusion matrix

**Rationale quality** (matching Nguyen et al.): ROUGE-1/2/L/Lsum, BERTScore

**New metrics introduced by this work:**
- **Faithfulness** — NLI-based check (DeBERTa-v3) measuring whether generated rationales logically entail their predicted label
- **Keyword reliance** — attention entropy over input tokens; higher = more distributed attention = less shortcutting

## Requirements

- Python 3.10+
- PyTorch 2.1+
- GPU with 40GB+ VRAM (A40, A100, H100, H200)
- HuggingFace access to [Llama 3 8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)

## References

```bibtex
@article{nguyen2024sentiment,
  title={Sentiment Reasoning for Healthcare},
  author={Nguyen, Khai-Nguyen and Le-Duc, Khai and Tat, Bach Phan and Le, Duy and Vo-Dang, Long and Hy, Truong-Son},
  journal={arXiv preprint arXiv:2407.21054v5},
  year={2025}
}
```

## License

MIT
