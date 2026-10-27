from __future__ import annotations

import logging
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DRIVE_FRCRN_DIR = "FA26_Capstone/Truong/FRCRN"
DRIVE_FRCRN_VLSP_DIR = "FA26_Capstone/Truong/FRCRN_vlsp"
BEST_FILENAME = "best.pt"


UPLOAD_ATTEMPTS = 3
UPLOAD_BACKOFF_SECONDS = 5.0


def _put(
    fs: Any,
    local_path: str | Path,
    remote_dir: str,
    remote_path: str,
    attempts: int = UPLOAD_ATTEMPTS,
    backoff: float = UPLOAD_BACKOFF_SECONDS,
) -> None:
    for attempt in range(1, attempts + 1):
        try:
            fs.makedirs(remote_dir, exist_ok=True)
            fs.put_file(str(local_path), remote_path)
            return
        except Exception:
            if attempt == attempts:
                raise
            time.sleep(backoff * attempt)


def utc_stamp(now: datetime | None = None) -> str:
    """Mốc thời gian UTC dùng trong tên tệp: 20261019T021500Z."""
    return (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")


class DriveCheckpointUploader:
    """Callback cho Trainer.on_checkpoint: mỗi lần có best mới thì ghi đè best.pt trên Google Drive.

    keep_history=True: ngoài best.pt (bản mới nhất) còn giữ một bản best_e<epoch>_<UTC>.pt cho từng lần cải thiện.
    """

    def __init__(
        self,
        fs: Any,
        remote_dir: str = DRIVE_FRCRN_DIR,
        logger: logging.Logger | None = None,
        keep_history: bool = False,
    ) -> None:
        self.fs = fs
        self.remote_dir = remote_dir.strip("/")
        self.logger = logger or logging.getLogger(__name__)
        self.keep_history = keep_history

    def history_path(self, epoch: int, stamp: str) -> str:
        return f"{self.remote_dir}/{Path(BEST_FILENAME).stem}_e{epoch:03d}_{stamp}{Path(BEST_FILENAME).suffix}"

    @property
    def remote_path(self) -> str:
        return f"{self.remote_dir}/{BEST_FILENAME}"

    def __call__(self, local_path: str | Path, epoch: int, score: float) -> None:
        # Lỗi mạng/Drive không được làm hỏng quá trình train; checkpoint local vẫn còn.
        try:
            _put(self.fs, local_path, self.remote_dir, self.remote_path)
            self.logger.info("drive: epoch %d (%.3f) -> %s", epoch, score, self.remote_path)
            if self.keep_history:
                history = self.history_path(epoch, utc_stamp())
                _put(self.fs, local_path, self.remote_dir, history)
                self.logger.info("drive: lưu bản lịch sử -> %s", history)
        except Exception as error:
            self.logger.warning("drive: upload failed at epoch %d: %r", epoch, error)


class DriveFileBackup:
    """Ghi đè một file local (resume, history...) lên Drive mỗi lần gọi; lỗi mạng không làm gãy train."""

    def __init__(self, fs: Any, remote_dir: str, filename: str, logger: logging.Logger | None = None) -> None:
        self.fs = fs
        self.remote_dir = remote_dir.strip("/")
        self.filename = filename.strip("/")
        self.logger = logger or logging.getLogger(__name__)

    @property
    def remote_path(self) -> str:
        return f"{self.remote_dir}/{self.filename}"

    def __call__(self, local_path: str | Path) -> None:
        try:
            _put(self.fs, local_path, self.remote_dir, self.remote_path)
            self.logger.info("drive backup: %s -> %s", local_path, self.remote_path)
        except Exception as error:
            self.logger.warning("drive backup failed for %s: %r", local_path, error)


class DriveLogUploader(logging.Handler):
    """Handler gắn vào logger của lần chạy: định kỳ ghi đè file .log lên Google Drive trong lúc train."""

    def __init__(self, fs: Any, log_file: str | Path, remote_dir: str, interval: float = 30.0) -> None:
        super().__init__(level=logging.INFO)
        self.fs = fs
        self.log_file = Path(log_file)
        self.remote_dir = remote_dir.strip("/")
        self.interval = interval
        self._last = float("-inf")
        self._timer: threading.Timer | None = None

    @property
    def remote_path(self) -> str:
        return f"{self.remote_dir}/{self.log_file.name}"

    def emit(self, record: logging.LogRecord) -> None:
        wait = self.interval - (time.monotonic() - self._last)
        if wait <= 0:
            self.upload()
        elif self._timer is None:
            self._timer = threading.Timer(wait, self._upload_pending)
            self._timer.daemon = True
            self._timer.start()

    def _upload_pending(self) -> None:
        self._timer = None
        self.upload()

    def close(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        super().close()

    def upload(self) -> bool:
        self._last = time.monotonic()
        # Không dùng logger của lần chạy ở đây (đệ quy); lỗi Drive chỉ in ra stderr.
        try:
            _put(self.fs, self.log_file, self.remote_dir, self.remote_path)
        except Exception as error:
            print(f"drive: log upload failed: {error!r}", file=sys.stderr)
            return False
        return True
