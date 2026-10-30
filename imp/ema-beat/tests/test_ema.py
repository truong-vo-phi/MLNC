from __future__ import annotations

import torch
from torch import nn

from ema_beat.ema import BEATsModelEMA


class ToyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.bn = nn.BatchNorm1d(20)
        self.fc2 = nn.Linear(20, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.bn(self.fc1(x)))

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self(x), dim=-1)


def test_ema_initialization() -> None:
    model = ToyClassifier()
    ema = BEATsModelEMA(model, decay=0.999, warmup_steps=100)

    # Tất cả tham số của shadow model phải tắt requires_grad
    for param in ema.ema_model.parameters():
        assert not param.requires_grad

    # Giá trị tham số ban đầu phải hoàn toàn bằng nhau
    assert ema.weight_distance(model) == 0.0


def test_ema_warmup_decay() -> None:
    model = ToyClassifier()
    ema = BEATsModelEMA(model, decay=0.999, warmup_steps=100, warmup_decay_start=0.90)

    assert ema.get_decay() == 0.90
    for _ in range(50):
        ema.update(model)
    # Tại step 50/100: decay phải ở giữa 0.90 và 0.999
    expected_mid = 0.90 + (0.999 - 0.90) * 0.5
    assert abs(ema.get_decay() - expected_mid) < 1e-4

    for _ in range(50):
        ema.update(model)
    # Sau step 100: decay phải đạt target 0.999
    assert abs(ema.get_decay() - 0.999) < 1e-6


def test_ema_numerical_update() -> None:
    model = ToyClassifier()
    decay = 0.9
    ema = BEATsModelEMA(model, decay=decay, warmup_steps=0)

    with torch.no_grad():
        initial_w = ema.ema_model.fc1.weight.clone()
        # Giả lập 1 bước gradient update thay đổi model.fc1.weight
        model.fc1.weight.add_(torch.ones_like(model.fc1.weight))

    ema.update(model)

    expected_w = initial_w * decay + model.fc1.weight * (1.0 - decay)
    assert torch.allclose(ema.ema_model.fc1.weight, expected_w, atol=1e-5)


def test_ema_weight_distance() -> None:
    model = ToyClassifier()
    ema = BEATsModelEMA(model, decay=0.99, warmup_steps=0)

    dist_before = ema.weight_distance(model)
    assert dist_before == 0.0

    with torch.no_grad():
        model.fc2.weight.add_(torch.full_like(model.fc2.weight, 2.0))

    dist_after_change = ema.weight_distance(model)
    assert dist_after_change > 0.0

    ema.update(model)
    dist_after_update = ema.weight_distance(model)
    # Sau khi EMA cập nhật, khoảng cách phải thu hẹp lại
    assert dist_after_update < dist_after_change


def test_ema_state_dict_and_export() -> None:
    model = ToyClassifier()
    ema = BEATsModelEMA(model, decay=0.95, warmup_steps=10)

    for _ in range(5):
        ema.update(model)

    state = ema.state_dict()
    assert state["step_count"] == 5
    assert "ema_model" in state

    new_ema = BEATsModelEMA(model, decay=0.95, warmup_steps=10)
    new_ema.load_state_dict(state)
    assert new_ema.step_count == 5

    exported = ema.export_model()
    assert isinstance(exported, ToyClassifier)
    for p in exported.parameters():
        assert not p.requires_grad


def test_ema_with_separate_teacher_model() -> None:
    student = ToyClassifier()
    teacher = ToyClassifier()
    with torch.no_grad():
        teacher.fc1.weight.add_(2.0)

    ema = BEATsModelEMA(student, teacher_model=teacher, decay=0.99)
    assert ema.weight_distance(student) > 0.0
    assert torch.allclose(ema.ema_model.fc1.weight, teacher.fc1.weight)
    for p in ema.ema_model.parameters():
        assert not p.requires_grad
