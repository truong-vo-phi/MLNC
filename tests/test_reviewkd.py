from __future__ import annotations

import logging
from dataclasses import replace

import pytest
import torch
from torch import Tensor, nn

from scripts.run_reviewkd import (
    RESULT_COLUMNS,
    build_review,
    build_student,
    check_grids,
    read_reviewkd_config,
    student_grid,
    teacher_grid,
)
from src.loss import HierarchicalContextLoss
from src.model.baseline import CNN10, BaselineConfig, BaselineModel
from src.model.beat import BEATClassifier, BEATConfig
from src.model.reviewkd import ReviewKDConfig, ReviewModule, ReviewTeacher, resize_map, tokens_to_grid
from src.train import ReviewKDTrainConfig, ReviewKDTrainer
from src.train.reviewkd_trainer import snr_scores

NUM_CLASSES = 3
CHANNELS = (4, 8, 16, 32)


def tiny_student(num_frames: int = 64) -> BaselineModel:
    config = BaselineConfig(
        arch="cnn10", channels=CHANNELS, num_classes=NUM_CLASSES, num_frames=num_frames, dropout=0.0
    )
    return BaselineModel(config)


def tiny_beats() -> BEATClassifier:
    config = BEATConfig(
        embed_dim=16,
        encoder_embed_dim=24,
        encoder_layers=3,
        encoder_attention_heads=2,
        encoder_ffn_embed_dim=48,
        conv_pos=4,
        conv_pos_groups=2,
        num_buckets=8,
        max_distance=16,
        num_classes=NUM_CLASSES,
    )
    return BEATClassifier(config)


class FixedTeacher(nn.Module):
    def __init__(self, shape: tuple[int, ...]) -> None:
        super().__init__()
        self.shape = shape
        self.calls = 0

    def forward(self, waveform: Tensor) -> Tensor:
        self.calls += 1
        return torch.randn(*self.shape[:1], waveform.shape[0], *self.shape[1:])


def batch_of(size: int = 4) -> dict[str, Tensor]:
    labels = torch.arange(size) % NUM_CLASSES
    return {
        "waveform": torch.randn(size, 16000),
        "target": torch.nn.functional.one_hot(labels, NUM_CLASSES).float(),
        "snr_db": torch.tensor([0.0, 5.0] * (size // 2)),
    }


def make_trainer(config: ReviewKDTrainConfig, teacher: nn.Module | None, tmp_path) -> ReviewKDTrainer:
    review = ReviewModule(CHANNELS, out_channels=10, out_size=(8, 4), mid_channels=16)
    return ReviewKDTrainer(
        tiny_student(),
        review,
        teacher,
        config,
        torch.device("cpu"),
        logging.getLogger("test-reviewkd"),
        "tiny",
        tmp_path / "ckpt.pt",
    )


def test_tokens_to_grid_keeps_time_major_layout() -> None:
    freq, time_steps, dim = 8, 5, 3
    tokens = torch.zeros(2, time_steps * freq, dim)
    for t in range(time_steps):
        for f in range(freq):
            tokens[:, t * freq + f, :] = t * 100 + f
    grid = tokens_to_grid(tokens, freq)
    assert grid.shape == (2, dim, freq, time_steps)
    assert grid[1, 2, 3, 4] == 4 * 100 + 3


def test_tokens_to_grid_rejects_indivisible_counts() -> None:
    with pytest.raises(ValueError, match="do not divide"):
        tokens_to_grid(torch.zeros(1, 10, 4), 8)


def test_resize_map_modes() -> None:
    x = torch.arange(64.0).reshape(1, 1, 8, 8)
    area = resize_map(x, (2, 2), "area")
    nearest = resize_map(x, (2, 2), "nearest")
    assert area.shape == nearest.shape == (1, 1, 2, 2)
    assert torch.isclose(area[0, 0, 0, 0], x[0, 0, :4, :4].mean())
    assert resize_map(x, (8, 8), "area") is x
    assert resize_map(x, (16, 16), "area").shape == (1, 1, 16, 16)


def test_review_module_outputs_teacher_sized_stack() -> None:
    module = ReviewModule(CHANNELS, out_channels=10, out_size=(2, 6), mid_channels=16)
    sizes = [(16, 48), (8, 24), (4, 12), (2, 6)]
    features = [torch.randn(3, c, *size) for c, size in zip(CHANNELS, sizes, strict=True)]
    stacked = module(features)
    assert stacked.shape == (4, 3, 10, 2, 6)
    stacked.sum().backward()
    assert all(parameter.grad is not None for parameter in module.parameters())


def test_review_module_rejects_wrong_stage_count() -> None:
    module = ReviewModule(CHANNELS, out_channels=10, out_size=(2, 6))
    with pytest.raises(ValueError, match="expected 4 stage features"):
        module([torch.randn(1, 4, 4, 4)])


def test_hcl_zero_for_identical_and_positive_otherwise() -> None:
    loss = HierarchicalContextLoss()
    student = torch.randn(4, 2, 5, 8, 24)
    assert loss(student, student).item() == 0.0
    teacher = student + 0.5
    values = loss.per_level(student, teacher)
    assert values.shape == (4,)
    assert (values > 0).all()
    assert torch.isclose(loss(student, teacher), values.sum())


def test_hcl_skips_levels_not_below_height() -> None:
    loss = HierarchicalContextLoss()
    student = torch.randn(1, 2, 3, 2, 6)
    teacher = torch.randn(1, 2, 3, 2, 6)
    full = torch.nn.functional.mse_loss(student[0], teacher[0])
    pooled = torch.nn.functional.mse_loss(
        torch.nn.functional.adaptive_avg_pool2d(student[0], (1, 1)),
        torch.nn.functional.adaptive_avg_pool2d(teacher[0], (1, 1)),
    )
    expected = (full + pooled * 0.5) / 1.5
    assert torch.isclose(loss(student, teacher), expected, atol=1e-6)


def test_hcl_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="matching"):
        HierarchicalContextLoss()(torch.zeros(2, 1, 3, 4, 4), torch.zeros(2, 1, 3, 4, 5))


def test_cnn10_stage_features_match_forward() -> None:
    config = BaselineConfig(arch="cnn10", channels=CHANNELS, num_classes=NUM_CLASSES, dropout=0.0)
    net = CNN10(config).eval()
    x = torch.randn(2, 1, 128, 64)
    logits, stages = net.forward_stages(x)
    assert torch.allclose(logits, net(x), atol=1e-6)
    assert [tuple(s.shape[1:]) for s in stages] == [(4, 64, 32), (8, 32, 16), (16, 16, 8), (32, 8, 4)]


def test_baseline_return_features_and_unsupported_backbone() -> None:
    model = tiny_student().eval()
    output = model(torch.randn(2, 16000), return_features=True)
    assert len(output.features) == 4
    assert model(torch.randn(2, 16000)).features == ()
    crnn = BaselineModel(BaselineConfig(arch="crnn", channels=(4, 8), hidden=8, layers=1, num_classes=3, num_frames=64))
    with pytest.raises(ValueError, match="does not expose stage features"):
        crnn(torch.randn(1, 16000), return_features=True)


def test_beats_hidden_layers_and_last_equals_features() -> None:
    beats = tiny_beats().eval()
    fbank = torch.randn(1, 32, 128)
    with torch.no_grad():
        default = beats(fbank, is_waveform=False)
        output = beats(fbank, is_waveform=False, layers=(1, 3))
    assert default.hidden == ()
    assert len(output.hidden) == 2
    assert torch.allclose(output.hidden[-1], output.features)
    assert torch.allclose(default.features, output.features, atol=1e-6)
    with pytest.raises(ValueError, match="outside"):
        beats(fbank, is_waveform=False, layers=(4,))


def test_review_teacher_grid_and_validation() -> None:
    class Identity(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = ReviewKDConfig()

        def forward(self, waveform: Tensor) -> Tensor:
            return waveform

    separator = Identity()
    separator.freeze = lambda: separator
    beats = tiny_beats()
    config = ReviewKDConfig(teacher_layers=(1, 2, 3), noise_from_residual=False)
    with pytest.raises(ValueError, match="target_rms must be positive"):
        ReviewTeacher(separator, beats, config, 0.0)
    with pytest.raises(ValueError, match="must be resolved"):
        ReviewTeacher(separator, beats, replace(config, noise_from_residual=None), 0.1)
    teacher = ReviewTeacher(separator, beats, config, 0.1)
    grid = teacher(torch.randn(2, 16000))
    assert grid.shape == (3, 2, 24, 8, 6)


def test_config_validation() -> None:
    with pytest.raises(ValueError, match="resize"):
        ReviewKDConfig(resize="bilinear")
    with pytest.raises(ValueError, match="strictly increasing"):
        ReviewKDConfig(teacher_layers=(6, 3))
    with pytest.raises(ValueError, match="review_weight"):
        ReviewKDTrainConfig(review_weight=-1.0)


def test_review_weight_warmup_schedule(tmp_path) -> None:
    config = ReviewKDTrainConfig(review_weight=2.0, review_warmup_epochs=5)
    trainer = make_trainer(config, None, tmp_path)
    assert trainer.review_weight(1) == 0.0
    assert trainer.review_weight(4) == pytest.approx(2.0 * 3 / 5)
    assert trainer.review_weight(6) == 2.0
    assert make_trainer(replace(config, review_warmup_epochs=0), None, tmp_path).review_weight(1) == 2.0


def test_training_step_updates_student_and_review_module(tmp_path) -> None:
    config = ReviewKDTrainConfig(review_weight=1.0, review_warmup_epochs=0, epochs=1)
    teacher = FixedTeacher((4, 10, 8, 4))
    trainer = make_trainer(config, teacher, tmp_path)
    trainer.configure_optimizer(steps_per_epoch=1)
    trainer.model.train()
    trainer.review.train()
    trainer._epoch = 1
    step = trainer.training_step(batch_of())
    assert torch.isfinite(step.loss)
    step.loss.backward()
    assert teacher.calls == 1
    assert trainer.review.abfs[0].conv1[0].weight.grad is not None
    assert trainer.model.backbone.blocks[0].layers[0].weight.grad is not None
    summary = trainer.epoch_summary()
    assert {"ce", "review", "review_l1", "review_l4", "teacher_s_per_step"} <= set(summary)
    ids = {id(p) for p in trainer.trainable_parameters}
    assert all(id(p) in ids for p in trainer.review.parameters())


def test_zero_weight_skips_teacher_and_matches_ce_only(tmp_path) -> None:
    config = ReviewKDTrainConfig(review_weight=0.0, epochs=1)
    teacher = FixedTeacher((4, 10, 8, 4))
    trainer = make_trainer(config, teacher, tmp_path)
    trainer._epoch = 1
    step = trainer.training_step(batch_of())
    assert teacher.calls == 0
    assert torch.isfinite(step.loss)
    assert trainer.epoch_summary()["review"] == 0.0


def test_snr_scores_keys() -> None:
    probabilities = torch.softmax(torch.randn(12, NUM_CLASSES), dim=-1).numpy()
    labels = (torch.arange(12) % NUM_CLASSES).numpy()
    snrs = torch.tensor([0.0, 5.0] * 6).numpy()
    scores = snr_scores(probabilities, labels, snrs, NUM_CLASSES)
    assert {"macro_acc", "acc_snr_0", "acc_snr_5", "balanced_snr_5", "mAP_macro", "macro_f1"} <= set(scores)


def test_shipped_config_grids_and_columns() -> None:
    raw = read_reviewkd_config()
    assert check_grids(raw) == (8, 24)
    assert teacher_grid(4.0, raw["frontend"]) == student_grid(raw)
    assert raw["model"]["pretrained"] is True
    student = build_student({**raw, "model": {**raw["model"], "pretrained": False}})
    review = build_review(raw, ReviewKDConfig.from_dict(raw["review"]), (8, 24))
    assert 4.5 <= sum(p.numel() for p in student.parameters()) / 1e6 <= 5.5
    assert review.out_size == (8, 24)
    assert {"macro_acc", "review_weight", "top1_accuracy"} <= set(RESULT_COLUMNS)


def test_check_grids_rejects_cropped_student() -> None:
    raw = read_reviewkd_config()
    raw["model"]["num_frames"] = 384
    with pytest.raises(ValueError, match="must equal"):
        check_grids(raw)


def _loader(batches: int = 2) -> list[dict[str, Tensor]]:
    return [batch_of(4) for _ in range(batches)]


class _ListLoader(list):
    """DataLoader tối thiểu: có len() và duyệt được nhiều lần."""


def test_resume_continues_from_saved_epoch_and_keeps_history(tmp_path) -> None:
    resume = tmp_path / "resume.pt"
    first = make_trainer(ReviewKDTrainConfig(review_weight=1.0, review_warmup_epochs=0, epochs=1, patience=5), FixedTeacher((4, 10, 8, 4)), tmp_path)
    first.resume_path = resume
    result_a = first.fit(_ListLoader(_loader()), _ListLoader(_loader(1)))
    assert result_a.epochs_run == 1 and resume.is_file()

    config_b = ReviewKDTrainConfig(review_weight=1.0, review_warmup_epochs=0, epochs=3, patience=5)
    second = make_trainer(config_b, FixedTeacher((4, 10, 8, 4)), tmp_path)
    second.resume_path = resume
    second.load_resume(resume)
    assert second._completed_epochs == 1
    for left, right in zip(first.model.state_dict().values(), second.model.state_dict().values(), strict=True):
        assert torch.equal(left, right)
    result_b = second.fit(_ListLoader(_loader()), _ListLoader(_loader(1)))
    assert [row["epoch"] for row in result_b.history] == [1, 2, 3]  # epoch 1 từ lần trước, 2-3 mới chạy
    assert result_b.epochs_run == 3
    assert second.skip_epoch(1) and not second.skip_epoch(2)


def test_resume_restores_stale_counter_and_best(tmp_path) -> None:
    resume = tmp_path / "resume.pt"
    trainer = make_trainer(ReviewKDTrainConfig(review_weight=0.0, epochs=1, patience=9), None, tmp_path)
    trainer.resume_path = resume
    trainer.fit(_ListLoader(_loader()), _ListLoader(_loader(1)))
    saved = torch.load(resume, map_location="cpu", weights_only=False)
    assert {"model", "review", "optimizer", "scheduler", "best_state", "epoch_records", "stale", "epoch"} <= set(saved)
    saved["stale"] = 4
    torch.save(saved, resume)
    again = make_trainer(ReviewKDTrainConfig(review_weight=0.0, epochs=2, patience=9), None, tmp_path)
    again.load_resume(resume)
    assert again.initial_stale() == 4 and again._best_epoch == saved["best_epoch"]


class _LogitTrainer(ReviewKDTrainer):
    """Trainer giả có teacher trả cả lưới đặc trưng lẫn logit (như NoiseKDTrainer)."""

    def _call_teacher(self, waveform: Tensor) -> Tensor:
        self._teacher_logits = torch.randn(waveform.shape[0], NUM_CLASSES)
        return self.teacher(waveform)


def _make_logit_trainer(config: ReviewKDTrainConfig, tmp_path) -> _LogitTrainer:
    review = ReviewModule(CHANNELS, out_channels=10, out_size=(8, 4), mid_channels=16)
    return _LogitTrainer(
        tiny_student(), review, FixedTeacher((4, 10, 8, 4)), config, torch.device("cpu"),
        logging.getLogger("test-logit"), "tiny", tmp_path / "ckpt.pt",
    )


def test_logit_kd_adds_a_positive_term_and_is_reported(tmp_path) -> None:
    config = ReviewKDTrainConfig(review_weight=1.0, review_warmup_epochs=0, logit_weight=1.0, logit_temperature=4.0, epochs=1)
    trainer = _make_logit_trainer(config, tmp_path)
    trainer.configure_optimizer(steps_per_epoch=1)
    trainer.model.train()
    trainer._epoch = 1
    step = trainer.training_step(batch_of())
    step.loss.backward()
    summary = trainer.epoch_summary()
    assert summary["logit_kd"] > 0 and summary["logit_weight"] == pytest.approx(1.0)
    assert trainer.model.backbone.blocks[0].layers[0].weight.grad is not None


def test_logit_only_still_calls_the_teacher_when_review_weight_is_zero(tmp_path) -> None:
    config = ReviewKDTrainConfig(review_weight=0.0, review_warmup_epochs=0, logit_weight=1.0, epochs=1)
    trainer = _make_logit_trainer(config, tmp_path)
    trainer._epoch = 1
    step = trainer.training_step(batch_of())
    assert trainer.teacher.calls == 1 and torch.isfinite(step.loss)
    assert trainer.epoch_summary()["review"] == 0.0 and trainer.epoch_summary()["logit_kd"] > 0


def test_logit_kd_without_teacher_logits_fails_loudly(tmp_path) -> None:
    config = ReviewKDTrainConfig(review_weight=1.0, review_warmup_epochs=0, logit_weight=1.0, epochs=1)
    trainer = make_trainer(config, FixedTeacher((4, 10, 8, 4)), tmp_path)  # trainer thường: không có logit
    trainer._epoch = 1
    with pytest.raises(ValueError, match="logit"):
        trainer.training_step(batch_of())


def test_logit_weight_follows_warmup_and_off_by_default(tmp_path) -> None:
    config = ReviewKDTrainConfig(logit_weight=2.0, review_warmup_epochs=4)
    trainer = make_trainer(config, None, tmp_path)
    assert trainer.logit_weight(1) == 0.0 and trainer.logit_weight(3) == pytest.approx(1.0) and trainer.logit_weight(10) == 2.0
    assert ReviewKDTrainConfig().logit_weight == 0.0
    with pytest.raises(ValueError, match="logit_weight"):
        ReviewKDTrainConfig(logit_weight=-1.0)
    with pytest.raises(ValueError, match="logit_temperature"):
        ReviewKDTrainConfig(logit_temperature=0.0)
