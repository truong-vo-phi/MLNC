from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

DRIVE_FRCRN_DIR = "FA26_Capstone/Truong/FRCRN"
BEST_FILENAME = "best.pt"


class DriveCheckpointUploader:
    """Callback cho Trainer.on_checkpoint: mỗi lần có best mới thì ghi đè best.pt trên Google Drive."""

    def __init__(self, fs: Any, remote_dir: str = DRIVE_FRCRN_DIR, logger: logging.Logger | None = None) -> None:
        self.fs = fs
        self.remote_dir = remote_dir.strip("/")
        self.logger = logger or logging.getLogger(__name__)

    @property
    def remote_path(self) -> str:
        return f"{self.remote_dir}/{BEST_FILENAME}"

    def __call__(self, local_path: str | Path, epoch: int, score: float) -> None:
        # Lỗi mạng/Drive không được làm hỏng quá trình train; checkpoint local vẫn còn.
        try:
            self.fs.makedirs(self.remote_dir, exist_ok=True)
            self.fs.put_file(str(local_path), self.remote_path)
            self.logger.info("drive: epoch %d (%.3f) -> %s", epoch, score, self.remote_path)
        except Exception as error:
            self.logger.warning("drive: upload failed at epoch %d: %r", epoch, error)
