from __future__ import annotations

from pathlib import Path

import pytest
import requests

from scripts import load_pretrain

PAYLOAD = bytes(range(256)) * 40


class _Response:
    def __init__(self, data: bytes, status: int, fail_after: int | None = None) -> None:
        self.data, self.status_code, self.fail_after = data, status, fail_after
        self.headers = {"content-length": str(len(data))}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, size: int):
        for index in range(0, len(self.data), 1000):
            if self.fail_after is not None and index >= self.fail_after:
                raise requests.exceptions.ChunkedEncodingError("Connection broken")
            yield self.data[index : index + 1000]


def test_interrupted_download_resumes_from_partial(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []

    def fake_get(url, stream, timeout, headers):
        calls.append(dict(headers))
        if not headers:
            return _Response(PAYLOAD, 200, fail_after=3000)
        start = int(headers["Range"].removeprefix("bytes=").removesuffix("-"))
        return _Response(PAYLOAD[start:], 206)

    monkeypatch.setattr(load_pretrain.requests, "get", fake_get)
    path = load_pretrain.download_checkpoint("https://example.com/a/model.pth", tmp_path)
    assert path.read_bytes() == PAYLOAD
    assert calls[0] == {} and calls[1] == {"Range": "bytes=3000-"}


def test_download_gives_up_after_retries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(load_pretrain.requests, "get", lambda *a, **k: _Response(PAYLOAD, 200, fail_after=0))
    monkeypatch.setattr(load_pretrain, "DOWNLOAD_RETRIES", 2)
    with pytest.raises(requests.exceptions.ChunkedEncodingError):
        load_pretrain.download_checkpoint("https://example.com/a/model.pth", tmp_path)
