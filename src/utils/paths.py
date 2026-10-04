from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

TIMESTAMP_FORMAT = "%Y-%m-%d-%H-%M-%S"
CHECKPOINT_FILENAME = "last_best_checkpoint.pt"
RESULT_FILENAME = "result.csv"
_RUN_ID = re.compile(r"^(?P<experiment>.+)-(?P<timestamp>\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})$")


@dataclass(frozen=True)
class RunPaths:
    experiment: str
    timestamp: str
    base_dir: Path

    @classmethod
    def new(cls, experiment: str, base_dir: str | Path, now: datetime | None = None) -> RunPaths:
        return cls(experiment, (now or datetime.now()).strftime(TIMESTAMP_FORMAT), Path(base_dir))

    @classmethod
    def from_run_id(cls, run_id: str, base_dir: str | Path) -> RunPaths:
        match = _RUN_ID.match(run_id)
        if match is None:
            raise ValueError(f"run id must look like '<experiment>-YYYY-MM-DD-HH-MM-SS', got {run_id!r}")
        return cls(match["experiment"], match["timestamp"], Path(base_dir))

    @property
    def run_id(self) -> str:
        return f"{self.experiment}-{self.timestamp}"

    @property
    def log_file(self) -> Path:
        return self.base_dir / "log" / f"{self.run_id}.log"

    @property
    def checkpoint_dir(self) -> Path:
        return self.base_dir / "checkpoint" / self.run_id

    @property
    def result_dir(self) -> Path:
        return self.base_dir / "result" / self.run_id

    @property
    def result_file(self) -> Path:
        return self.result_dir / RESULT_FILENAME

    @property
    def history_dir(self) -> Path:
        return self.result_dir / "history"

    @property
    def figure_dir(self) -> Path:
        return self.base_dir / "figure" / self.run_id

    def checkpoint_path(self, model_name: str) -> Path:
        return self.checkpoint_dir / model_name / CHECKPOINT_FILENAME

    def history_path(self, name: str) -> Path:
        return self.history_dir / f"{name}.json"

    def create(self) -> RunPaths:
        for directory in (self.log_file.parent, self.checkpoint_dir, self.history_dir, self.figure_dir):
            directory.mkdir(parents=True, exist_ok=True)
        return self
