"""Hồ sơ của một lần chạy trên molab: mốc thời gian từng epoch, vị trí Drive, mã băm của code đã dùng.

Ghi `run_manifest.json` cạnh kết quả và (nếu có Drive) ở gốc thư mục Drive của lần chạy, để sandbox mất kết nối vẫn biết
lần chạy dùng code nào, lưu ở đâu và nối tiếp thế nào. `archive_code` đóng gói đúng các tệp đã dùng để tái triển khai.
"""

from __future__ import annotations

import hashlib
import io
import json
import platform
import subprocess
import tarfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CODE_PATTERNS = (("src", "**/*.py"), ("scripts", "*.py"), ("config", "*"))
EXCLUDE = {"src/utils/download.py"}  # chứa token viết thẳng trong mã: không đóng gói, không đẩy lên sandbox
OFFSET_DIR = Path("output") / "audioset-ref"
MANIFEST_NAME = "run_manifest.json"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def md5_of(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def code_files(root: Path = ROOT, extra: Iterable[str] = ()) -> list[str]:
    """Đường dẫn tương đối (posix) của code cần để chạy lại, cộng thêm các tệp `extra` nếu có thật."""
    found: set[str] = set()
    for folder, pattern in CODE_PATTERNS:
        for path in (root / folder).glob(pattern):
            if path.is_file() and "__pycache__" not in path.parts:
                found.add(path.relative_to(root).as_posix())
    found.update(rel for rel in extra if (root / rel).is_file())
    return sorted(found - EXCLUDE)


def offset_files(root: Path = ROOT) -> list[str]:
    folder = root / OFFSET_DIR
    return sorted(p.relative_to(root).as_posix() for p in folder.glob("offsets_*.csv")) if folder.is_dir() else []


def fingerprint(root: Path = ROOT, files: Sequence[str] | None = None) -> dict[str, str]:
    files = files if files is not None else code_files(root, offset_files(root))
    return {rel: md5_of((root / rel).read_bytes()) for rel in files}


def code_hash(prints: Mapping[str, str]) -> str:
    return md5_of(json.dumps(dict(sorted(prints.items())), sort_keys=True).encode("utf-8"))


def archive_code(out_path: Path, root: Path = ROOT, files: Sequence[str] | None = None) -> dict[str, str]:
    """Đóng gói các tệp thành tar.gz (cùng đường dẫn tương đối); trả về mã băm từng tệp."""
    files = files if files is not None else code_files(root, offset_files(root))
    prints: dict[str, str] = {}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out_path, "w:gz", compresslevel=9) as tar:
        for rel in files:
            data = (root / rel).read_bytes()
            prints[rel] = md5_of(data)
            info = tarfile.TarInfo(rel)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return prints


def read_archive(archive: Path) -> dict[str, bytes]:
    with tarfile.open(archive, "r:gz") as tar:
        return {m.name: tar.extractfile(m).read() for m in tar.getmembers() if m.isfile()}


def git_state(root: Path = ROOT) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            out = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, timeout=20)
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "uncommitted_files": None if status is None else len(status.splitlines()),
    }


def environment() -> dict[str, Any]:
    info: dict[str, Any] = {"python": platform.python_version(), "host": platform.node()}
    try:
        import torch

        info["torch"] = torch.__version__
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:  # noqa: BLE001 - thông tin phụ, không được làm gãy lần chạy
        pass
    return info


class RunManifest:
    """Hồ sơ lần chạy: tạo lúc bắt đầu, thêm một dòng mỗi epoch, đóng lại khi xong; ghi local và sao lưu Drive."""

    def __init__(
        self,
        path: Path,
        data: dict[str, Any],
        backup: Callable[[Path], None] | None = None,
        list_drive: Callable[[], list[dict[str, Any]]] | None = None,
    ) -> None:
        self.path = Path(path)
        self.data = data
        self.backup = backup
        self.list_drive = list_drive

    @classmethod
    def start(
        cls,
        path: Path,
        *,
        spec: Mapping[str, Any],
        drive_dir: str | None,
        stage_name: str,
        command: Sequence[str] = (),
        root: Path = ROOT,
        backup: Callable[[Path], None] | None = None,
        list_drive: Callable[[], list[dict[str, Any]]] | None = None,
    ) -> RunManifest:
        prints = fingerprint(root)
        started = now_iso()
        stage_dir = f"{drive_dir.strip('/')}/{stage_name}" if drive_dir else None
        data = {
            "run_id": "frcrn-noise-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
            "status": "running",
            "started_at": started,
            "finished_at": None,
            "command": list(command),
            "environment": environment(),
            "git": git_state(root),
            "code_hash": code_hash(prints),
            "code_files": prints,
            "spec": dict(spec),
            "drive": {
                "dir": drive_dir,
                "stage_dir": stage_dir,
                "manifest": f"{drive_dir.strip('/')}/{MANIFEST_NAME}" if drive_dir else None,
                "best": f"{stage_dir}/best.pt" if stage_dir else None,
                "resume": f"{stage_dir}/resume.pt" if stage_dir else None,
                "result": f"{stage_dir}/result.json" if stage_dir else None,
                "files": [],
            },
            "resume_hint": (
                "Chạy lại đúng lệnh trong `command` với cùng `drive.dir`: script tự khôi phục resume.pt/best.pt từ Drive "
                "và nối tiếp; nếu result.json đã có thì lần chạy coi như xong."
            ),
            "epochs": [],
            "result": None,
        }
        manifest = cls(path, data, backup, list_drive)
        manifest.save()
        return manifest

    def add_epoch(self, epoch: int, scores: Mapping[str, float]) -> None:
        keep = {k: round(float(v), 6) for k, v in scores.items() if isinstance(v, (int, float))}
        self.data["epochs"].append({"epoch": epoch, "time": now_iso(), **keep})
        self.save()

    def finish(self, result: Mapping[str, Any] | None, status: str = "finished") -> None:
        self.data["status"] = status
        self.data["finished_at"] = now_iso()
        if result is not None:
            self.data["result"] = {k: result.get(k) for k in ("selection_metric", "best_score", "best_epoch", "epochs_run")}
        self.save()

    def save(self) -> None:
        if self.list_drive is not None:
            try:
                self.data["drive"]["files"] = self.list_drive()
            except Exception:  # noqa: BLE001 - Drive lỗi không được làm gãy lần chạy
                pass
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2, ensure_ascii=False), encoding="utf-8")
        if self.backup is not None:
            self.backup(self.path)


def drive_listing(fs: Any, stage_dir: str) -> list[dict[str, Any]]:
    """Tên, kích thước, thời gian sửa của các tệp trong thư mục Drive của lần chạy."""
    rows = []
    for entry in fs.ls(stage_dir, detail=True):
        rows.append(
            {
                "name": str(entry.get("name", "")).rsplit("/", 1)[-1],
                "size": entry.get("size"),
                "modified": str(entry.get("mtime") or entry.get("modified") or entry.get("modifiedTime") or ""),
            }
        )
    return sorted(rows, key=lambda r: r["name"])
