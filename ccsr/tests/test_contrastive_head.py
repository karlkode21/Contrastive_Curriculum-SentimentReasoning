import torch
import torch.nn.functional as F
from src.models.contrastive_head import ProjectionHead, supervised_contrastive_loss, sub_cluster_loss


def test_projection_head_output_shape():
    head = ProjectionHead(input_dim=4096, hidden_dim=256, output_dim=128)
    z = head(torch.randn(8, 4096))
    assert z.shape == (8, 128)


def test_projection_head_output_normalized():
    head = ProjectionHead(input_dim=4096, hidden_dim=256, output_dim=128)
    z = head(torch.randn(8, 4096))
    norms = torch.norm(z, dim=1)
    assert torch.allclose(norms, torch.ones(8), atol=1e-5)


def test_supcon_loss_shape():
    embeddings = F.normalize(torch.randn(12, 128), dim=1)
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    loss = supervised_contrastive_loss(embeddings, labels, temperature=0.07)
    assert loss.shape == ()
    assert loss.item() > 0


def test_supcon_loss_perfect_clusters():
    e0 = F.normalize(torch.tensor([[1.0, 0.0, 0.0]]), dim=1)
    e1 = F.normalize(torch.tensor([[0.0, 1.0, 0.0]]), dim=1)
    e2 = F.normalize(torch.tensor([[0.0, 0.0, 1.0]]), dim=1)
    perfect = torch.cat([e0.repeat(4, 1), e1.repeat(4, 1), e2.repeat(4, 1)])
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    loss_perfect = supervised_contrastive_loss(perfect, labels, temperature=0.07)

    torch.manual_seed(0)
    random_emb = F.normalize(torch.randn(12, 3), dim=1)
    loss_random = supervised_contrastive_loss(random_emb, labels, temperature=0.07)
    assert loss_perfect.item() < loss_random.item()


def test_sub_cluster_loss():
    embeddings = F.normalize(torch.randn(6, 128), dim=1)
    sim_indices = torch.tensor([[0, 1], [3, 4]])
    sim_values = torch.tensor([0.9, 0.88])
    loss = sub_cluster_loss(embeddings, sim_indices, sim_values)
    assert loss.shape == ()
    assert loss.item() >= 0


def test_sub_cluster_loss_empty():
    embeddings = F.normalize(torch.randn(4, 128), dim=1)
    loss = sub_cluster_loss(embeddings, torch.tensor([]).long().reshape(0, 2), torch.tensor([]))
    assert loss.item() == 0.0
