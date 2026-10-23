import numpy as np
import pytest
import torch

from scripts.sweep_frcrn import (
    SELECTION_METRIC,
    best_epoch_scores,
    evaluate_by_snr,
    load_sweep_config,
    sweep_table,
    trial_train_config,
)
from src.loss import FRCRNLoss
from src.metrics import SnrBucketMeter
from src.model.frcrn import FRCRNConfig, FRCRNSeparator
from src.train import FRCRNTrainConfig, FRCRNTrainer


def _batches(count: int = 2, size: int = 4, length: int = 8000):
    generator = torch.Generator().manual_seed(0)
    levels = torch.tensor([-5.0, 0.0, 15.0, 20.0])
    for _ in range(count):
        noise = torch.randn(size, length, generator=generator) * 0.05
        clean = torch.randn(size, length, generator=generator) * 0.1
        yield {"waveform": clean + noise, "noise": noise, "snr_db": levels[:size]}


def test_meter_buckets_and_high_mean():
    meter = SnrBucketMeter(high_levels=(15, 20))
    meter.update(torch.tensor([1.0, 3.0, 10.0, 20.0]), torch.tensor([0.0, 0.0, 15.0, 20.0]))
    scores = meter.compute("m")
    assert scores["m_0"] == pytest.approx(2.0)
    assert scores["m_15"] == pytest.approx(10.0)
    assert scores["m_hi"] == pytest.approx(15.0)
    assert scores["m"] == pytest.approx(8.5)


def test_meter_rejects_empty_and_mismatched():
    meter = SnrBucketMeter()
    with pytest.raises(ValueError):
        meter.compute("m")
    with pytest.raises(ValueError):
        meter.update(torch.zeros(3), torch.zeros(2))


def test_mask_weight_scales_mask_term_only():
    config = FRCRNConfig()
    mixture, target = torch.randn(1, 8000), torch.randn(1, 8000) * 0.5
    estimate = target + 0.1 * torch.randn(1, 8000)
    mask = torch.randn(1, 2 * config.num_bins, (8000 - config.win_len) // config.win_inc + 1)
    base = FRCRNLoss(config)(mixture, target, estimate, mask)
    zero = FRCRNLoss(config, mask_weight=0.0)(mixture, target, estimate, mask)
    double = FRCRNLoss(config, mask_weight=2.0)(mixture, target, estimate, mask)
    mask_term = base - zero
    assert mask_term > 0
    assert double - zero == pytest.approx(2.0 * mask_term.item(), rel=1e-4)
    assert FRCRNLoss(config, mask_weight=1.0)(mixture, target, estimate, mask) == base
    with pytest.raises(ValueError):
        FRCRNLoss(config, mask_weight=-1.0)


def test_train_config_validation():
    with pytest.raises(ValueError):
        FRCRNTrainConfig(selection_metric="accuracy")
    with pytest.raises(ValueError):
        FRCRNTrainConfig(lr_decay=1.5)
    assert FRCRNTrainConfig().selection_metric == "noise_si_snr"


def test_trial_train_config_overrides_and_rejects_unknown():
    config = trial_train_config(FRCRNTrainConfig(), {"lr": 3e-4, "lr_decay": 0.98, "batch_size": 12, "seed": 1})
    assert (config.lr, config.lr_decay, config.batch_size) == (3e-4, 0.98, 12)
    assert config.selection_metric == SELECTION_METRIC
    assert config.high_snr_levels == (15.0, 20.0)
    with pytest.raises(ValueError):
        trial_train_config(FRCRNTrainConfig(), {"hidden": 64})


def test_sweep_config_is_consistent():
    config = load_sweep_config()
    assert config["sweep"]["metric"]["name"] == SELECTION_METRIC
    names = set(config["sweep"]["parameters"])
    assert names <= {
        "lr",
        "lr_decay",
        "halve_lr_after",
        "batch_size",
        "segment_seconds",
        "grad_clip",
        "mask_weight",
        "target",
        "epochs",
        "patience",
    }


def test_best_epoch_scores_and_table():
    result = {
        "best_epoch": 2,
        "history": [{"epoch": 1, "val_x": 1.0}, {"epoch": 2, "val_x": 2.0, "val_speech_si_snr_15": 5.0}],
    }
    assert best_epoch_scores(result) == {"x": 2.0, "speech_si_snr_15": 5.0}
    rows = sweep_table({"a": {SELECTION_METRIC: 1.0}, "b": {SELECTION_METRIC: 3.0}, "c": {}})
    assert [row["run"] for row in rows] == ["b", "a", "c"]
    assert np.isnan(rows[-1][SELECTION_METRIC])


def test_trainer_evaluate_reports_per_snr_and_selection(tmp_path):
    import logging

    separator = FRCRNSeparator(FRCRNConfig())
    config = FRCRNTrainConfig(selection_metric="noise_si_snr_hi", epochs=1)
    seen = []
    trainer = FRCRNTrainer(
        separator,
        config,
        torch.device("cpu"),
        logging.getLogger("t"),
        "frcrn",
        tmp_path / "best.pt",
        epoch_callback=lambda epoch, scores: seen.append((epoch, scores)),
    )
    assert trainer.selection_metric == "noise_si_snr_hi"
    scores = trainer.evaluate(list(_batches()))
    keys = ("loss", "noise_si_snr", "noise_si_snr_hi", "noise_si_snr_-5", "noise_si_snr_20", "speech_si_snr_hi")
    for key in keys:
        assert key in scores
        assert np.isfinite(scores[key])
    assert scores["noise_si_snr_hi"] == pytest.approx((scores["noise_si_snr_15"] + scores["noise_si_snr_20"]) / 2)
    assert seen == []


def test_evaluate_by_snr_includes_improvement():
    separator = FRCRNSeparator(FRCRNConfig())
    result = evaluate_by_snr(separator, list(_batches(count=1)), torch.device("cpu"))
    assert result["speech_si_snri_20"] == pytest.approx(result["speech_si_snr_20"] - result["input_si_snr_20"])


def test_noise_target_separator_trains_and_evaluates(tmp_path):
    import logging

    separator = FRCRNSeparator(FRCRNConfig(target="noise"))
    batch = next(_batches(count=1, size=2))
    mixture, noise = batch["waveform"], batch["noise"]
    assert separator.training_target(mixture, noise) is noise
    assert torch.isfinite(separator.loss(mixture, noise))
    trainer = FRCRNTrainer(
        separator,
        FRCRNTrainConfig(selection_metric="noise_si_snr_hi"),
        torch.device("cpu"),
        logging.getLogger("t"),
        "frcrn",
        tmp_path / "best.pt",
    )
    scores = trainer.evaluate(list(_batches(count=1)))
    assert np.isfinite(scores["noise_si_snr_hi"])


def test_run_tracked_trial_logs_epochs_and_summary(monkeypatch):
    import sys
    import types

    import scripts.sweep_frcrn as sweep

    events = {"log": [], "metrics": [], "summary": {}, "finished": False, "init": None}

    class FakeRun:
        summary = types.SimpleNamespace(update=lambda values: events["summary"].update(values))

        def define_metric(self, *args, **kwargs):
            events["metrics"].append((args, kwargs))

        def log(self, values):
            events["log"].append(values)

        def finish(self):
            events["finished"] = True

    def fake_init(**kwargs):
        events["init"] = kwargs
        return FakeRun()

    monkeypatch.setitem(sys.modules, "wandb", types.SimpleNamespace(init=fake_init))

    def fake_trial(settings, on_epoch=None, **kwargs):
        on_epoch(1, {"noise_si_snr_hi": -1.0})
        on_epoch(2, {"noise_si_snr_hi": 0.5})
        return {"best_epoch": 2, "best_score": 0.5, "history": [{"epoch": 2, "val_noise_si_snr_15": 1.0}]}

    monkeypatch.setattr(sweep, "run_trial", fake_trial)
    result = sweep.run_tracked_trial({"lr": 1e-4}, project="p", run_name="n", run_id="abc", trial_dir="x")
    assert result["best_epoch"] == 2
    assert events["init"]["id"] == "abc" and events["init"]["resume"] == "allow"
    assert [row["epoch"] for row in events["log"]] == [1, 2]
    assert events["summary"]["noise_si_snr_hi"] == 0.5 and events["summary"]["noise_si_snr_15"] == 1.0
    assert events["finished"] is True


def test_run_from_spec_dispatches_plain_and_tracked(tmp_path, monkeypatch):
    import json

    import scripts.sweep_frcrn as sweep

    calls = {}
    monkeypatch.setattr(
        sweep, "run_trial", lambda settings, **kw: calls.update(plain=(settings, kw)) or {"mode": "plain"}
    )
    monkeypatch.setattr(
        sweep, "run_tracked_trial", lambda settings, **kw: calls.update(tracked=(settings, kw)) or {"mode": "tracked"}
    )
    spec = {
        "settings": {"lr": 1e-4},
        "trial_dir": str(tmp_path),
        "stage_name": "s",
        "device": "cpu",
        "max_samples": 8,
        "num_workers": 0,
        "smoke": True,
    }
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    assert sweep.run_from_spec(path) == {"mode": "plain"}
    assert calls["plain"][1]["fs"] is None and calls["plain"][1]["stage_name"] == "s"
    path.write_text(json.dumps({**spec, "wandb_project": "p", "run_name": "n", "run_id": "i"}), encoding="utf-8")
    assert sweep.run_from_spec(path) == {"mode": "tracked"}
    assert calls["tracked"][1]["project"] == "p" and calls["tracked"][1]["run_id"] == "i"


def test_read_training_result_reports_log_tail(tmp_path):
    import scripts.sweep_frcrn as sweep

    (tmp_path / "train.log").write_text("line one\nboom\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="boom"):
        sweep.read_training_result(tmp_path / "result.json", tmp_path)
    (tmp_path / "result.json").write_text('{"best_epoch": 3}', encoding="utf-8")
    assert sweep.read_training_result(tmp_path / "result.json", tmp_path) == {"best_epoch": 3}


@pytest.mark.skipif(not __import__("pathlib").Path("/proc/self/cmdline").exists(), reason="needs /proc")
def test_training_pid_matches_marker_only(tmp_path, monkeypatch):
    import os

    import scripts.sweep_frcrn as sweep

    pid_file = tmp_path / "train.pid"
    assert sweep.training_pid(pid_file) is None
    pid_file.write_text(str(os.getpid()), encoding="utf-8")
    monkeypatch.setattr(sweep, "PROCESS_MARKER", "definitely-not-in-cmdline")
    assert sweep.training_pid(pid_file) is None
    monkeypatch.setattr(sweep, "PROCESS_MARKER", "pytest")
    assert sweep.training_pid(pid_file) == os.getpid()
    pid_file.write_text("not a pid", encoding="utf-8")
    assert sweep.training_pid(pid_file) is None


def test_drive_put_retries_then_succeeds(tmp_path, monkeypatch):
    import scripts.drive_sync as drive

    class FlakyFs:
        def __init__(self):
            self.failures, self.puts = 2, 0

        def makedirs(self, *args, **kwargs):
            pass

        def put_file(self, local, remote):
            self.puts += 1
            if self.failures:
                self.failures -= 1
                raise OSError("Resumable upload not finalized: server returned 308")

    fs = FlakyFs()
    monkeypatch.setattr(drive.time, "sleep", lambda seconds: None)
    drive._put(fs, tmp_path / "x", "a/b", "a/b/x")
    assert fs.puts == 3
    fs.failures = 5
    with pytest.raises(OSError):
        drive._put(fs, tmp_path / "x", "a/b", "a/b/x", attempts=2)


def test_resume_skips_finished_epochs_and_keeps_history(tmp_path):
    import logging
    from dataclasses import replace

    config = FRCRNTrainConfig(epochs=2, patience=5, selection_metric="noise_si_snr_hi", halve_lr_after=1)
    batches = list(_batches(count=1, size=4))

    def make(epochs, load=False):
        trainer = FRCRNTrainer(
            FRCRNSeparator(FRCRNConfig(target="speech")),
            replace(config, epochs=epochs),
            torch.device("cpu"),
            logging.getLogger("t"),
            "frcrn",
            tmp_path / "best.pt",
            resume_path=tmp_path / "resume.pt",
        )
        if load:
            trainer.load_resume(tmp_path / "resume.pt")
        return trainer

    first = make(2).fit(batches, batches)
    assert [row["epoch"] for row in first.history] == [1, 2]

    resumed = make(3, load=True)
    trained = []
    original = resumed.training_step
    resumed.training_step = lambda batch: trained.append(1) or original(batch)
    result = resumed.fit(batches, batches)
    assert len(trained) == len(batches)
    assert [row["epoch"] for row in result.history] == [1, 2, 3]
    assert result.epochs_run == 3


def test_epoch_callback_includes_train_loss(tmp_path):
    import logging

    seen = []
    trainer = FRCRNTrainer(
        FRCRNSeparator(FRCRNConfig()),
        FRCRNTrainConfig(epochs=1),
        torch.device("cpu"),
        logging.getLogger("t"),
        "frcrn",
        tmp_path / "best.pt",
        epoch_callback=lambda epoch, scores: seen.append((epoch, scores)),
    )
    trainer.configure_optimizer(1)
    trainer._last_scores = {"loss": 3.0, "noise_si_snr_hi": 1.5}
    trainer._last_train = {"train_loss": 4.0}
    trainer.end_epoch(2)
    epoch, scores = seen[0]
    assert epoch == 2
    assert scores["train_loss"] == 4.0 and scores["loss"] == 3.0 and scores["noise_si_snr_hi"] == 1.5
    assert scores["lr"] == pytest.approx(trainer.lr)


def test_run_from_spec_passes_on_epoch_and_audioset_loaders(tmp_path, monkeypatch):
    import json

    import scripts.sweep_frcrn as sweep

    calls = {}
    monkeypatch.setattr(sweep, "run_trial", lambda settings, **kw: calls.update(kw=kw) or {"mode": "plain"})
    spec = {
        "settings": {"target": "noise"},
        "trial_dir": str(tmp_path),
        "stage_name": "s",
        "device": "cpu",
        "audioset_noise": {"offsets_dir": str(tmp_path), "cache_dir": str(tmp_path), "max_error": 0.01},
    }
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    def callback(epoch, scores):
        return None

    sweep.run_from_spec(spec_path, callback)
    assert calls["kw"]["on_epoch"] is callback
    assert callable(calls["kw"]["make_loaders"])
    calls.clear()
    sweep.run_from_spec(spec_path)
    assert "on_epoch" not in calls["kw"]
