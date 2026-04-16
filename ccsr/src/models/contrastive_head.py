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

    sim_matrix = embeddings @ embeddings.T / temperature  # (B, B)

    self_mask = torch.eye(B, dtype=torch.bool, device=device)
    sim_matrix = sim_matrix.masked_fill(self_mask, -1e9)

    labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)  # (B, B)
    pos_mask = labels_eq & ~self_mask

    # Numerical stability
    sim_max, _ = sim_matrix.max(dim=1, keepdim=True)
    sim_matrix = sim_matrix - sim_max.detach()

    exp_sim = torch.exp(sim_matrix) * (~self_mask).float()
    log_sum_exp = torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

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
        embeddings: (B, D) normalized embeddings
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
