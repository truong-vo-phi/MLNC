import logging
import zipfile
from datetime import datetime

import pytest

from scripts.download_dataset import extract_zip, read_token, resolve_slug
from src.utils.logger import setup_logger
from src.utils.paths import CHECKPOINT_FILENAME, RESULT_FILENAME, RunPaths


def test_run_paths_follow_guide(tmp_path):
    run = RunPaths.new("experiment-01", tmp_path, now=datetime(2026, 10, 2, 22, 36, 9))
    assert run.run_id == "experiment-01-2026-10-02-22-36-09"
    assert run.log_file == tmp_path / "log" / "experiment-01-2026-10-02-22-36-09.log"
    assert run.checkpoint_path("frcrn") == (
        tmp_path / "checkpoint" / "experiment-01-2026-10-02-22-36-09" / "frcrn" / CHECKPOINT_FILENAME
    )
    assert run.result_file == tmp_path / "result" / "experiment-01-2026-10-02-22-36-09" / RESULT_FILENAME


def test_run_paths_create_and_resume(tmp_path):
    run = RunPaths.new("frcrn-beat", tmp_path).create()
    assert run.log_file.parent.is_dir()
    assert run.history_dir.is_dir() and run.figure_dir.is_dir() and run.checkpoint_dir.is_dir()
    assert RunPaths.from_run_id(run.run_id, tmp_path) == run


def test_run_id_must_have_timestamp(tmp_path):
    with pytest.raises(ValueError):
        RunPaths.from_run_id("experiment-01", tmp_path)


def test_logger_writes_to_given_file(tmp_path):
    log_file = tmp_path / "log" / "run.log"
    logger = setup_logger("paths-test", log_file)
    logger.info("hello")
    for handler in logger.handlers:
        handler.flush()
    assert "hello" in log_file.read_text(encoding="utf-8")
    assert isinstance(logger, logging.Logger)


def test_kaggle_token_from_environment(monkeypatch):
    monkeypatch.delenv("KAGGLE_ACCESS_TOKEN", raising=False)
    with pytest.raises(RuntimeError):
        read_token()
    monkeypatch.setenv("KAGGLE_ACCESS_TOKEN", "token")
    assert read_token() == "token"


def test_resolve_slug_and_extract(tmp_path):
    config = tmp_path / "url.yaml"
    config.write_text("kaggle-x: owner/data\nempty:\n", encoding="utf-8")
    assert resolve_slug("kaggle-x", config) == "owner/data"
    with pytest.raises(KeyError):
        resolve_slug("empty", config)

    archive = tmp_path / "a.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("x.txt", "1")
    assert (extract_zip(archive, tmp_path / "out") / "x.txt").read_text() == "1"


def test_find_root_handles_nested_extract(tmp_path):
    from scripts.download_dataset import find_root

    nested = tmp_path / "audioset" / "AudioSet" / "metadata"
    nested.mkdir(parents=True)
    (nested / "manifest.csv").write_text("x", encoding="utf-8")
    assert find_root(tmp_path / "audioset", "metadata/manifest.csv") == tmp_path / "audioset" / "AudioSet"
    with pytest.raises(FileNotFoundError):
        find_root(tmp_path, "missing.txt")


def test_ensure_dataset_uses_existing_extract_without_token(tmp_path, monkeypatch):
    from scripts.download_dataset import ensure_dataset
    from scripts.load_dataset import MIXTURE_KEY, dataset_env_name, resolve_dataset_root

    monkeypatch.delenv("KAGGLE_ACCESS_TOKEN", raising=False)
    monkeypatch.setenv(dataset_env_name(MIXTURE_KEY), "")
    config = tmp_path / "url.yaml"
    config.write_text(f"{MIXTURE_KEY}: '{tmp_path / 'absent'}'\nkaggle-mix-dataset: owner/mix\n", encoding="utf-8")
    extracted = tmp_path / "mix-dataset" / "mix-dataset"
    extracted.mkdir(parents=True)
    (extracted / "labels.txt").write_text("a", encoding="utf-8")

    root = ensure_dataset(MIXTURE_KEY, tmp_path, config)
    assert root == extracted
    assert resolve_dataset_root(MIXTURE_KEY, config) == extracted
