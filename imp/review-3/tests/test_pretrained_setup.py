"""Pretrained routing and checkpoint backup integrity, without GPU or downloads."""
import io
import json
import logging
from pathlib import Path

import pytest
import yaml

from scripts.run_all_models import CANONICAL_MODELS, resolve_model_config
from src.trainer import Trainer


def test_pretrained_profiles_keep_group_hyperparameters():
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "config/pretrained.yaml").read_text())
    for name in CANONICAL_MODELS:
        effective = resolve_model_config(name, config)
        assert effective["ce_only"]["pretrained"] is True
        assert effective["training"]["epochs"] == 50
        assert effective["training"]["label_smoothing"] == .1
    assert resolve_model_config("ced_tiny", config)["training"]["layer_decay"] == .3
    assert resolve_model_config("beats", config)["training"]["lr"] == 1e-5
    assert config["logging"]["checkpoint_dir"] == "checkpoints/pretrained"


class MemoryDrive:
    def __init__(self):
        self.files = {}

    def exists(self, path):
        return path in self.files

    def invalidate_cache(self, path):
        pass

    def ls(self, path):
        return [name for name in self.files if name.startswith(path + "/")]

    def open(self, path, mode):
        assert mode == "rb"
        return io.BytesIO(self.files[path])

    def put_file(self, local, remote):
        self.files[remote] = Path(local).read_bytes()

    def get_file(self, remote, local):
        Path(local).write_bytes(self.files[remote])

    def rm(self, remote):
        del self.files[remote]


def backup_trainer(tmp_path):
    trainer = Trainer.__new__(Trainer)
    trainer.checkpoint_dir = tmp_path
    trainer.drive_fs = MemoryDrive()
    trainer.drive_dir = "own-run/beats"
    trainer.config = {"logging": {"backup_every_epochs": 5}}
    trainer.best_epoch = 1
    trainer.logger = logging.getLogger("backup-test")
    return trainer


def test_drive_resume_pair_and_checksum_guard(tmp_path):
    trainer = backup_trainer(tmp_path)
    for name in ("best.pt", "last.pt"):
        (tmp_path / name).write_bytes(name.encode())
    trainer._backup_drive(1, improved=True)
    assert json.loads((tmp_path / "backup_status.json").read_text())["ok"]
    for name in ("best.pt", "last.pt"):
        (tmp_path / name).unlink()
    trainer._restore_drive()
    assert (tmp_path / "best.pt").read_bytes() == b"best.pt"
    assert (tmp_path / "last.pt").read_bytes() == b"last.pt"
    for name in ("best.pt", "last.pt"):
        (tmp_path / name).unlink()
    trainer.drive_fs.files["own-run/beats/last_e001.pt"] = b"corrupt"
    with pytest.raises(ValueError, match="SHA256"):
        trainer._restore_drive()
    assert not (tmp_path / "last.pt").exists()
    assert not (tmp_path / "best.pt").exists()


def test_failed_upload_retains_previous_manifest(tmp_path):
    trainer = backup_trainer(tmp_path)
    for name in ("best.pt", "last.pt"):
        (tmp_path / name).write_bytes(name.encode())
    trainer._backup_drive(1, improved=True)
    original = trainer.drive_fs.files["own-run/beats/resume_manifest.json"]
    put = trainer.drive_fs.put_file
    def fail(local, remote):
        if remote.endswith("resume_best_e005.pt"):
            raise OSError("Disconnected")
        put(local, remote)
    trainer.drive_fs.put_file = fail
    trainer._backup_drive(5)
    assert trainer.drive_fs.files["own-run/beats/resume_manifest.json"] == original
    assert not json.loads((tmp_path / "backup_status.json").read_text())["ok"]


def test_final_metrics_survive_large_checkpoint_upload_failure(tmp_path):
    trainer = backup_trainer(tmp_path)
    for name in ("results.json", "metrics_per_snr.csv", "validation_predictions.npz", "best.pt", "last.pt"):
        (tmp_path / name).write_bytes(name.encode())
    put = trainer.drive_fs.put_file
    def fail(local, remote):
        if remote.endswith("last_e005.pt"):
            raise OSError("Large checkpoint transfer failed")
        put(local, remote)
    trainer.drive_fs.put_file = fail
    trainer._backup_drive(5, final=True)
    for name in ("results.json", "metrics_per_snr.csv", "validation_predictions.npz"):
        assert trainer.drive_fs.files["own-run/beats/" + name] == name.encode()
