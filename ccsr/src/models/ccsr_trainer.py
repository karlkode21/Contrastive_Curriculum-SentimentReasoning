"""Custom Trainer integrating generative loss, contrastive loss, focal loss, and curriculum."""

import json
import torch
from transformers import Trainer
from torch.utils.data import DataLoader

from src.models.contrastive_head import (
    ProjectionHead,
    supervised_contrastive_loss,
    sub_cluster_loss,
)
from src.models.focal_loss import FocalLoss
from src.data.curriculum import CurriculumSampler


class CCSRTrainer(Trainer):
    """Trainer subclass adding contrastive and focal losses to the generative objective."""

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

        # -- Contrastive head --
        if self.contrastive_config.get("enabled"):
            hidden_size = self.model.config.hidden_size
            self.projection_head = ProjectionHead(
                input_dim=hidden_size,
                hidden_dim=self.contrastive_config.get("projection_hidden_dim", 256),
                output_dim=self.contrastive_config.get("projection_output_dim", 128),
            )
            self.sim_indices = None
            self.sim_values = None
            if rationale_sim_path:
                with open(rationale_sim_path) as f:
                    sim_data = json.load(f)
                pairs = sim_data["pairs"]
                if pairs:
                    self.sim_indices = torch.tensor(
                        [[p[0], p[1]] for p in pairs], dtype=torch.long
                    )
                    self.sim_values = torch.tensor(
                        [p[2] for p in pairs], dtype=torch.float32
                    )

        # -- Focal loss --
        if self.focal_config.get("enabled"):
            self.focal_loss_fn = FocalLoss(
                gamma=self.focal_config.get("gamma", 2.0),
                num_classes=3,
                class_weights=class_weights,
            )

    # ---- Curriculum ----

    def _get_curriculum_phase(self, epoch: int) -> tuple[float, bool]:
        cfg = self.curriculum_config
        if not cfg.get("enabled"):
            return 1.0, False
        if epoch + 1 in cfg.get("phase1_epochs", [1, 2]):
            return cfg.get("phase1_threshold", 0.4), False
        elif epoch + 1 in cfg.get("phase2_epochs", [3, 4]):
            return cfg.get("phase2_threshold", 0.7), False
        else:
            return 1.0, cfg.get("class_balanced_phase3", True)

    def get_train_dataloader(self) -> DataLoader:
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
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    # ---- Joint loss ----

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Override compute_loss to inject contrastive + focal losses."""
        # Extract our custom columns before passing to model
        label_idx = inputs.pop("label_idx", None)
        global_idx = inputs.pop("global_idx", None)

        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"],
            output_hidden_states=self.contrastive_config.get("enabled", False),
        )
        total_loss = outputs.loss  # L_gen

        # -- Contrastive loss --
        if (
            self.contrastive_config.get("enabled")
            and label_idx is not None
            and outputs.hidden_states is not None
        ):
            last_hidden = outputs.hidden_states[-1]  # (B, seq_len, hidden)
            # Use the last non-padding position of the input (approx. end-of-label)
            # For simplicity, average-pool over sequence positions with attention
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

            self.projection_head = self.projection_head.to(pooled.device)
            z = self.projection_head(pooled)

            con_loss = supervised_contrastive_loss(
                z,
                label_idx,
                temperature=self.contrastive_config.get("temperature", 0.07),
            )

            # Sub-cluster loss
            sub_loss_val = torch.tensor(0.0, device=z.device)
            if self.sim_indices is not None and global_idx is not None:
                idx_to_batch = {int(g): b for b, g in enumerate(global_idx)}
                batch_pairs_i, batch_pairs_j, batch_sims = [], [], []
                for k in range(self.sim_indices.shape[0]):
                    gi, gj = int(self.sim_indices[k, 0]), int(self.sim_indices[k, 1])
                    if gi in idx_to_batch and gj in idx_to_batch:
                        batch_pairs_i.append(idx_to_batch[gi])
                        batch_pairs_j.append(idx_to_batch[gj])
                        batch_sims.append(self.sim_values[k].item())
                if batch_pairs_i:
                    b_indices = torch.tensor(
                        list(zip(batch_pairs_i, batch_pairs_j)),
                        dtype=torch.long,
                        device=z.device,
                    )
                    b_values = torch.tensor(batch_sims, dtype=torch.float32, device=z.device)
                    sub_loss_val = sub_cluster_loss(z, b_indices, b_values)

            lambda_con = self.contrastive_config.get("lambda_con", 0.3)
            alpha_sub = self.contrastive_config.get("sub_cluster_alpha", 0.1)
            total_loss = total_loss + lambda_con * (con_loss + alpha_sub * sub_loss_val)

        # -- Focal loss --
        if self.focal_config.get("enabled") and label_idx is not None and self.label_token_ids:
            logits = outputs.logits  # (B, seq_len, vocab)
            label_tids = [self.label_token_ids[i] for i in range(3)]
            # Logits at the first generated position for the 3 label tokens
            label_logits = logits[:, 0, label_tids]  # (B, 3)
            self.focal_loss_fn = self.focal_loss_fn.to(label_logits.device)
            focal = self.focal_loss_fn(label_logits, label_idx)
            total_loss = total_loss + self.focal_config.get("lambda_bal", 0.5) * focal

        return (total_loss, outputs) if return_outputs else total_loss

    # ---- Epoch tracking ----

    def on_epoch_begin(self, args, state, control, **kwargs):
        self._current_epoch = int(state.epoch) if state.epoch else 0
