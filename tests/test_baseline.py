from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from scripts.run_baselines import (
    BASELINES_CONFIG,
    COMPARISON_COLUMNS,
    RESULT_COLUMNS,
    available_frames,
    baseline_names,
    best_epoch_scores,
    best_models,
    check_frames,
    comparison_table,
    describe_baselines,
    evaluate_baselines,
    load_confusions,
    load_results,
    markdown_table,
    plot_baseline_confusions,
    read_baselines_config,
    train_baseline,
    write_comparison,
    write_result_csv,
    write_variant_config,
)
from src.model.baseline import BaselineConfig, BaselineModel
from src.train import BEATTrainConfig
from src.utils.paths import RunPaths

NUM_CLASSES = 40
PARAM_BANDS = {
    "efficientat-mnv3": (4.0, 4.6),
    "panns-cnn10": (4.7, 5.3),
    "resnet18": (10.8, 11.6),
    "efficientnet-b2": (7.4, 8.1),
    "convnextv2-t": (27.5, 28.3),
    "crnn-bigru": (9.8, 10.8),
    "bilstm": (11.3, 12.2),
    "deit-tiny": (5.2, 5.8),
    "vit-s16-ced": (21.2, 22.0),
    "mobilevit-s": (4.7, 5.3),
    "conformer-m": (26.5, 27.5),
    "swin-t-htsat": (27.1, 28.0),
    "efficientat-mn04": (0.68, 0.78),
    "efficientat-mn05": (1.06, 1.18),
    "efficientat-mn20": (16.4, 16.9),
    "ced-mini": (9.5, 9.8),
}


def tiny_model(num_classes: int = 3) -> BaselineModel:
    config = BaselineConfig(
        arch="crnn", channels=(4, 8), hidden=8, layers=1, num_classes=num_classes, num_frames=64, dropout=0.0
    )
    return BaselineModel(config)


def synthetic_loader(samples: int = 12, num_classes: int = 3) -> DataLoader:
    items = [
        {
            "waveform": torch.randn(16000),
            "target": torch.nn.functional.one_hot(torch.tensor(i % num_classes), num_classes).float(),
        }
        for i in range(samples)
    ]
    return DataLoader(items, batch_size=4)


def test_yaml_lists_all_baselines() -> None:
    assert sorted(baseline_names()) == sorted(PARAM_BANDS)


@pytest.mark.parametrize("name", sorted(PARAM_BANDS))
def test_parameter_count_in_band(name: str) -> None:
    model = BaselineModel.from_yaml(BASELINES_CONFIG, name)
    millions = sum(parameter.numel() for parameter in model.parameters()) / 1e6
    low, high = PARAM_BANDS[name]
    assert low <= millions <= high, f"{name}: {millions:.2f}M outside [{low}, {high}]"
    assert millions < 50


@pytest.mark.parametrize("name", sorted(PARAM_BANDS))
def test_forward_shape(name: str) -> None:
    raw = read_baselines_config()
    config = BaselineConfig.from_dict({**raw["model"], **raw["models"][name], "num_classes": NUM_CLASSES})
    model = BaselineModel(config, raw["frontend"]).eval()
    with torch.no_grad():
        output = model(torch.randn(1, 160000))
        probabilities = model.predict_proba(torch.randn(1, 160000))
    assert output.logits.shape == (1, NUM_CLASSES)
    assert torch.isfinite(output.logits).all()
    assert torch.allclose(probabilities.sum(dim=-1), torch.ones(1), atol=1e-5)


def test_from_yaml_needs_name_when_several_models() -> None:
    with pytest.raises(ValueError, match="pass name="):
        BaselineModel.from_yaml(BASELINES_CONFIG)
    with pytest.raises(KeyError):
        BaselineModel.from_yaml(BASELINES_CONFIG, "does-not-exist")


def test_custom_arch_rejects_pretrained() -> None:
    with pytest.raises(ValueError, match="no pretrained weights"):
        BaselineModel(BaselineConfig(arch="crnn", pretrained=True))


def test_frontend_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="does not match"):
        BaselineModel(BaselineConfig(arch="crnn", num_frames=64), {"num_frames": 128})


def test_available_frames_and_check() -> None:
    frontend = read_baselines_config()["frontend"]
    assert available_frames(10.0, frontend) == 998
    assert available_frames(4.0, frontend) == 398
    check_frames(10.0, frontend, 960)
    with pytest.raises(ValueError, match="exceeds"):
        check_frames(4.0, frontend, 960)


def test_describe_baselines_reports_params() -> None:
    rows = describe_baselines(num_classes=NUM_CLASSES)
    assert {row["model"] for row in rows} == set(PARAM_BANDS)
    for row in rows:
        low, high = PARAM_BANDS[row["model"]]
        assert low <= row["params_m"] <= high


def test_best_epoch_scores_strips_prefix() -> None:
    history = [{"epoch": 1, "val_macro_f1": 0.1}, {"epoch": 2, "val_macro_f1": 0.4, "val_mAP": 0.5}]
    assert best_epoch_scores(history, 2) == {"macro_f1": 0.4, "mAP": 0.5}
    with pytest.raises(ValueError):
        best_epoch_scores(history, 3)


def test_write_result_csv_header(tmp_path: Path) -> None:
    rows = [dict.fromkeys(RESULT_COLUMNS, 0.0)]
    path = write_result_csv(rows, tmp_path / "out" / "result.csv")
    assert path.read_text(encoding="utf-8").splitlines()[0].split(",") == list(RESULT_COLUMNS)


def test_train_baseline_writes_outputs_and_skips_when_done(tmp_path: Path) -> None:
    run = RunPaths.new("baselines", tmp_path).create()
    logger = logging.getLogger("baseline-test")
    config = BEATTrainConfig(epochs=2, patience=2, lr=1e-3, batch_size=4, eval_batch_size=4)
    loaders = {"train": synthetic_loader(), "validation": synthetic_loader()}

    row = train_baseline("tiny", tiny_model(), loaders, config, run, torch.device("cpu"), logger)

    assert set(RESULT_COLUMNS) <= set(row)
    assert 1 <= row["best_epoch"] <= 2
    assert run.checkpoint_path("tiny").is_file()
    assert run.history_path("tiny").is_file()

    again = train_baseline("tiny", tiny_model(), loaders, config, run, torch.device("cpu"), logger)
    assert again == row


def make_rows() -> list[dict]:
    rows = []
    for index, (name, family) in enumerate((("resnet18", "CNN"), ("deit-tiny", "ViT"), ("crnn-bigru", "RNN"))):
        row = {"model": name, "family": family, "architecture": name, "params_m": 5.0 + index}
        row.update({column: 0.1 * (index + 1) for column in RESULT_COLUMNS if column not in row})
        rows.append(row)
    return rows


def test_comparison_table_ranks_by_macro_f1() -> None:
    table = comparison_table(make_rows())
    assert [row["model"] for row in table] == ["crnn-bigru", "deit-tiny", "resnet18"]
    assert [row["rank"] for row in table] == [1, 2, 3]
    with pytest.raises(KeyError):
        comparison_table(make_rows(), sort_by="nope")


def test_best_models_returns_leader_per_metric() -> None:
    best = best_models(make_rows())
    assert set(best) == set(RESULT_COLUMNS[-9:])
    assert set(best.values()) == {"crnn-bigru"}


def test_write_comparison_header(tmp_path: Path) -> None:
    run = RunPaths.new("baselines", tmp_path)
    path = write_comparison(make_rows(), run)
    assert path == run.result_dir / "comparison.csv"
    assert path.read_text(encoding="utf-8").splitlines()[0].split(",") == list(COMPARISON_COLUMNS)


def test_load_results_and_confusions_roundtrip_and_plot(tmp_path: Path) -> None:
    run = RunPaths.new("baselines", tmp_path).create()
    matrix = [[3, 1, 0], [0, 4, 0], [1, 0, 2]]
    for row in make_rows():
        payload = {"row": row, "history": [], "confusion_matrix": matrix}
        run.history_path(row["model"]).write_text(json.dumps(payload), encoding="utf-8")

    assert {row["model"] for row in load_results(run)} == {"resnet18", "deit-tiny", "crnn-bigru"}
    matrices = load_confusions(run)
    assert matrices["resnet18"].shape == (3, 3)
    paths = plot_baseline_confusions(matrices, ["a", "b", "c"], run)
    assert set(paths) == set(matrices)
    assert all(path.is_file() and path.parent == run.figure_dir for path in paths.values())


def test_evaluate_baselines_reports_missing_checkpoint(tmp_path: Path) -> None:
    run = RunPaths.new("baselines", tmp_path).create()
    with pytest.raises(FileNotFoundError, match="resnet18"):
        evaluate_baselines(run, torch.device("cpu"), logging.getLogger("baseline-test"), names=["resnet18"])


def test_markdown_table_formats_floats() -> None:
    table = markdown_table([{"model": "a", "macro_f1": 0.123456, "rank": 1}], ("rank", "model", "macro_f1"), digits=3)
    assert table.splitlines() == ["| rank | model | macro_f1 |", "| --- | --- | --- |", "| 1 | a | 0.123 |"]


def test_write_variant_config_switches_dataset(tmp_path: Path) -> None:
    path = write_variant_config(
        tmp_path / "out" / "baselines.yaml", dataset_name="mixture-36l", duration=4.0, num_frames=384
    )
    raw = read_baselines_config(path)
    assert raw["dataset"] == {"name": "mixture-36l", "duration": 4.0}
    assert raw["frontend"]["num_frames"] == raw["model"]["num_frames"] == 384
    assert set(raw["models"]) == set(PARAM_BANDS)
    model = BaselineModel.from_yaml(path, "deit-tiny")
    assert model.config.num_frames == 384
    with pytest.raises(ValueError, match="exceeds"):
        write_variant_config(tmp_path / "bad.yaml", dataset_name="mixture-36l", duration=4.0, num_frames=960)


def _fake_panns_cnn10() -> dict[str, torch.Tensor]:
    widths = (1, 64, 128, 256, 512)
    state: dict[str, torch.Tensor] = {"bn0.weight": torch.ones(64), "fc_audioset.weight": torch.zeros(527, 512)}
    for i in range(4):
        for conv, bn, (cin, cout) in (("conv1", "bn1", (widths[i], widths[i + 1])), ("conv2", "bn2", (widths[i + 1],) * 2)):
            state[f"conv_block{i + 1}.{conv}.weight"] = torch.full((cout, cin, 3, 3), 0.5 + i)
            for suffix, value in (("weight", 1.0), ("bias", 0.0), ("running_mean", 0.0), ("running_var", 1.0)):
                state[f"conv_block{i + 1}.{bn}.{suffix}"] = torch.full((cout,), value)
            state[f"conv_block{i + 1}.{bn}.num_batches_tracked"] = torch.tensor(7)
    state["fc1.weight"] = torch.full((512, 512), 0.25)
    state["fc1.bias"] = torch.full((512,), 0.125)
    return state


def test_cnn10_loads_panns_weights(tmp_path: Path) -> None:
    path = tmp_path / "Cnn10.pth"
    torch.save({"model": _fake_panns_cnn10(), "iteration": 1}, path)
    model = BaselineModel(BaselineConfig(arch="cnn10", num_classes=36, pretrained=True, freq_bins=128, num_frames=398))
    head_before = model.backbone.head.weight.clone()
    report = model.backbone.load_panns(path)
    assert report.is_clean
    assert model.backbone.blocks[2].layers[0].weight.mean().item() == pytest.approx(2.5)
    assert model.backbone.fc.bias.mean().item() == pytest.approx(0.125)
    assert torch.equal(model.backbone.head.weight, head_before)


def test_cnn10_panns_rejects_other_channels(tmp_path: Path) -> None:
    path = tmp_path / "Cnn10.pth"
    torch.save({"model": _fake_panns_cnn10()}, path)
    model = BaselineModel(BaselineConfig(arch="cnn10", channels=(32, 64, 128, 256), freq_bins=128, num_frames=398))
    with pytest.raises(ValueError, match="channels"):
        model.backbone.load_panns(path)
