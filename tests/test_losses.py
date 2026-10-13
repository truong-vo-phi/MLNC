import torch

from src.loss import CrossEntropyLoss


def test_cross_entropy_matches_manual():
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    target = torch.tensor([0, 1])
    expected = -torch.log_softmax(logits, dim=-1)[torch.arange(2), target].mean()
    assert torch.allclose(CrossEntropyLoss()(logits, target), expected)
