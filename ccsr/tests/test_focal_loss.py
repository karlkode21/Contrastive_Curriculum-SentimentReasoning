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
    torch.manual_seed(42)
    loss_focal = FocalLoss(gamma=0.0, num_classes=3)
    loss_ce = torch.nn.CrossEntropyLoss()
    logits = torch.randn(8, 3)
    labels = torch.tensor([0, 1, 2, 1, 0, 2, 1, 0])
    assert torch.allclose(loss_focal(logits, labels), loss_ce(logits, labels), atol=1e-5)


def test_focal_loss_with_class_weights():
    loss_fn = FocalLoss(gamma=2.0, num_classes=3, class_weights=[1.0, 0.5, 2.0])
    logits = torch.randn(4, 3)
    labels = torch.tensor([0, 1, 2, 1])
    assert loss_fn(logits, labels).item() > 0


def test_focal_loss_confident_lower():
    loss_fn = FocalLoss(gamma=2.0, num_classes=3)
    confident = torch.tensor([[10.0, -10.0, -10.0]])
    uncertain = torch.tensor([[0.5, 0.3, 0.2]])
    labels = torch.tensor([0])
    assert loss_fn(confident, labels).item() < loss_fn(uncertain, labels).item()
