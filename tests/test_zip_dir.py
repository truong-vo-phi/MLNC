from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from scripts.zip_dir import zip_dir


def test_zip_dir_keeps_relative_paths_and_content(tmp_path: Path) -> None:
    src = tmp_path / "nhat"
    (src / "sub").mkdir(parents=True)
    (src / "train.done").write_text("3", encoding="utf-8")
    (src / "sub" / "data.bin").write_bytes(bytes(range(256)) * 100)
    dst = tmp_path / "out" / "nhat.zip"
    names = zip_dir(src, dst)
    assert names == ["sub/data.bin", "train.done"]
    with zipfile.ZipFile(dst) as archive:
        assert sorted(archive.namelist()) == names
        assert archive.read("sub/data.bin") == bytes(range(256)) * 100
        assert archive.read("train.done") == b"3"


def test_zip_dir_rejects_missing_source(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        zip_dir(tmp_path / "missing", tmp_path / "x.zip")
