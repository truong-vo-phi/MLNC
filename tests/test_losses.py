import torch

from src.loss import CRDLoss, CrossEntropyLoss, KDLoss


def test_kd_is_zero_for_identical_logits():
    logits = torch.randn(4, 10)
    assert KDLoss(temperature=4.0)(logits, logits).abs() < 1e-5


def test_kd_has_no_gradient_to_teacher():
    student = torch.randn(4, 10, requires_grad=True)
    teacher = torch.randn(4, 10, requires_grad=True)
    KDLoss()(student, teacher).backward()
    assert teacher.grad is None
    assert student.grad is not None


def test_cross_entropy_matches_manual():
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    target = torch.tensor([0, 1])
    expected = -torch.log_softmax(logits, dim=-1)[torch.arange(2), target].mean()
    assert torch.allclose(CrossEntropyLoss()(logits, target), expected)


def test_crd_prefers_aligned_pairs():
    torch.manual_seed(0)
    crd = CRDLoss(student_dim=16, teacher_dim=16, proj_dim=8)
    with torch.no_grad():
        crd.proj_s.weight.copy_(crd.proj_t.weight)
        crd.proj_s.bias.copy_(crd.proj_t.bias)
    features = torch.randn(6, 5, 16)
    aligned = crd(features, features)
    shuffled = crd(features, features.roll(1, dims=0))
    assert aligned < shuffled
