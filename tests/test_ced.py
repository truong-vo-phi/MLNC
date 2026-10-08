import logging

import pytest
import torch
from safetensors.torch import save_file
from torch.optim import SGD
from torch.utils.data import DataLoader

from src.core import BaseConfig, BaseModel, LoadReport
from src.model.beat import BEATConfig, BEATTeacher
from src.model.ced import CEDClassifier, CEDConfig
from src.train import BEATTrainer, CEDTrainConfig, CEDTrainer, Trainer
from src.train.beat_trainer import cosine_with_warmup

NUM_CLASSES = 5
MIDS = [f"/m/{index}" for index in range(8)]


def tiny_ced_config(**overrides):
    values = {"embed_dim": 32, "depth": 2, "num_heads": 2, "target_length": 64, "num_classes": NUM_CLASSES}
    return CEDConfig(**{**values, **overrides})


def tiny_beat_config():
    return BEATConfig(
        encoder_layers=2,
        encoder_attention_heads=4,
        encoder_embed_dim=64,
        encoder_ffn_embed_dim=128,
        conv_pos=16,
        conv_pos_groups=4,
    )


def beat_checkpoint(donor, path):
    model = {}
    for key, value in donor.state_dict().items():
        if key.startswith("backbone."):
            name = key.removeprefix("backbone.")
            name = name.replace("patch_embed.patch_embedding.", "patch_embedding.")
            name = name.replace(".ffn.fc1.", ".fc1.").replace(".ffn.fc2.", ".fc2.")
            model[name] = value
    model["predictor.weight"], model["predictor.bias"] = donor.head.fc.weight, donor.head.fc.bias
    torch.save({"model": model, "label_dict": dict(enumerate(MIDS))}, path)
    return path


@pytest.fixture(scope="module")
def tiny_ced():
    return CEDClassifier(tiny_ced_config()).eval()


@pytest.fixture(scope="module")
def donor_teacher():
    torch.manual_seed(0)
    return BEATTeacher(tiny_beat_config()).eval()


@pytest.fixture
def teacher(donor_teacher, tmp_path):
    model = BEATTeacher(tiny_beat_config())
    model.load_pretrained(beat_checkpoint(donor_teacher, tmp_path / "beat.pt"))
    return model.select_classes(MIDS[:NUM_CLASSES]).eval()


def test_forward_shapes_from_spectrogram(tiny_ced):
    with torch.no_grad():
        out = tiny_ced(torch.randn(2, 64, 48), is_waveform=False)
    assert out.logits.shape == (2, NUM_CLASSES)
    assert out.features.shape == (2, 4 * 3, 32)
    assert torch.isfinite(out.logits).all()


def test_forward_from_waveform(tiny_ced):
    with torch.no_grad():
        out = tiny_ced(torch.randn(2, 8000) * 0.1)
    assert out.logits.shape == (2, NUM_CLASSES)
    assert torch.equal(tiny_ced.predict_proba(torch.randn(2, 8000) * 0.1).sum(dim=-1).round(), torch.ones(2))


def test_rejects_input_longer_than_positional_embedding(tiny_ced):
    with pytest.raises(ValueError):
        tiny_ced(torch.randn(1, 64, 80), is_waveform=False)


def test_config_from_yaml(tmp_path):
    path = tmp_path / "ced.yaml"
    path.write_text("model:\n  depth: 3\n  num_classes: 7\n  unknown_key: 1\n", encoding="utf-8")
    config = CEDConfig.from_yaml(path)
    assert config.depth == 3
    assert config.num_classes == 7
    model = CEDClassifier.from_yaml(path)
    assert model.config.depth == 3


def test_default_param_count_matches_ced_mini():
    millions = CEDClassifier(CEDConfig()).num_parameters() / 1e6
    assert 9.3 < millions < 9.7


def test_load_pretrained_from_safetensors(tmp_path):
    donor, model = CEDClassifier(tiny_ced_config()), CEDClassifier(tiny_ced_config())
    state = {
        f"encoder.{key.removeprefix('backbone.')}": value
        for key, value in donor.state_dict().items()
        if "backbone." in key
    }
    state["outputlayer.1.weight"] = torch.zeros(527, 32)
    save_file(state, tmp_path / "model.safetensors")
    report = model.load_pretrained(tmp_path / "model.safetensors")
    assert report == LoadReport()
    assert torch.equal(model.backbone.blocks[0].attn.qkv.weight, donor.backbone.blocks[0].attn.qkv.weight)
    assert torch.equal(model.backbone.time_pos_embed, donor.backbone.time_pos_embed)


def test_teacher_selects_classes_by_mid(donor_teacher, teacher):
    fbank = torch.randn(2, 64, 128)
    with torch.no_grad():
        full = donor_teacher(fbank, is_waveform=False).logits
        selected = teacher(fbank, is_waveform=False).logits
    assert full.shape == (2, 527)
    assert selected.shape == (2, NUM_CLASSES)
    assert torch.allclose(selected, full[:, :NUM_CLASSES], atol=1e-5)


def test_teacher_select_classes_validates(donor_teacher, tmp_path):
    model = BEATTeacher(tiny_beat_config())
    with pytest.raises(ValueError):
        model.select_classes(MIDS)
    model.load_pretrained(beat_checkpoint(donor_teacher, tmp_path / "beat.pt"))
    with pytest.raises(KeyError):
        model.select_classes(["/m/unknown"])


def test_ced_follows_core_interfaces():
    assert issubclass(CEDClassifier, BaseModel)
    assert issubclass(CEDConfig, BaseConfig) and issubclass(CEDTrainConfig, BaseConfig)
    assert issubclass(CEDTrainer, Trainer) and issubclass(CEDTrainer, BEATTrainer)
    assert issubclass(BEATTeacher, BaseModel)


def test_min_lr_ratio_floors_cosine_schedule():
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = SGD([parameter], lr=1.0)
    scheduler = cosine_with_warmup(optimizer, total_steps=100, warmup_ratio=0.1, min_lr_ratio=0.1)
    for _ in range(100):
        optimizer.step()
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.1, abs=1e-3)


def test_distillation_smoke(teacher, tmp_path):
    torch.manual_seed(0)
    samples = [
        {
            "waveform": torch.randn(8000) * 0.1,
            "target": torch.nn.functional.one_hot(torch.tensor(i % NUM_CLASSES), NUM_CLASSES).float(),
        }
        for i in range(10)
    ]
    loader = DataLoader(samples, batch_size=5)
    config = CEDTrainConfig(epochs=1, patience=1, batch_size=5, eval_batch_size=5, kd_temperature=2.0)
    trainer = CEDTrainer(
        CEDClassifier(tiny_ced_config()),
        teacher,
        config,
        torch.device("cpu"),
        logging.getLogger("test"),
        "ced-test",
        tmp_path / "best.pt",
    )
    result = trainer.fit(loader, loader)
    assert result.epochs_run == 1
    assert torch.isfinite(torch.tensor(result.history[0]["train_loss"]))
    assert {"val_macro_f1", "val_mAP", "mAP"} <= set(result.history[0])
    assert (tmp_path / "best.pt").is_file()
    assert not any(parameter.requires_grad for parameter in trainer.teacher.parameters())
    assert {"top1_accuracy", "mAP"} <= set(trainer.evaluate_teacher(loader))
