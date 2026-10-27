"""Lưu hồ sơ và tái triển khai lần chạy lên molab (dùng khi sandbox cũ mất kết nối hoặc hết hạn).

    python -m scripts.molab_deploy record --url URL --drive-dir DIR [-- <đối số run_frcrn_noise>]
    python -m scripts.molab_deploy push   --url URL --archive output/molab-runs/<id>/code.tar.gz [--force]
    python -m scripts.molab_deploy launch --url URL [--kaggle] -- <đối số run_frcrn_noise>
    python -m scripts.molab_deploy status --url URL [--work-dir /marimo/output/frcrn-noise]

Token marimo đọc từ biến môi trường MARIMO_TOKEN (không đưa vào dòng lệnh). Với `launch --kaggle`, token Kaggle lấy từ biến
môi trường KAGGLE_ACCESS_TOKEN của máy này và chỉ truyền vào môi trường tiến trình trên sandbox (không in, không ghi tệp).
Cần đúng một phiên marimo đang mở trên sandbox (tải lại tab nếu chưa có).
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import zlib
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_manifest import (  # noqa: E402
    archive_code,
    code_files,
    code_hash,
    git_state,
    md5_of,
    now_iso,
    offset_files,
    read_archive,
)

DEFAULT_OUT_DIR = ROOT / "output" / "molab-runs"
REMOTE_ROOT = "/marimo"
DEFAULT_WORK_DIR = f"{REMOTE_ROOT}/output/frcrn-noise"
CHUNK_BYTES = 3_000_000  # kích thước nén tối đa mỗi lần gửi
RUN_MODULE = ["python", "-m", "scripts.run_frcrn_noise"]


class MolabError(RuntimeError):
    pass


@dataclass
class StreamResult:
    stdout: str = ""
    stderr: str = ""
    done: bool = False
    success: bool = True


def parse_stream(lines: Iterable[str]) -> StreamResult:
    """Đọc luồng SSE của /api/kernel/execute: sự kiện stdout, stderr và done."""
    result, event = StreamResult(), ""
    for line in lines:
        line = line.rstrip("\r")
        if line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            try:
                payload = json.loads(line[5:].strip())
            except json.JSONDecodeError:
                continue
            if event == "stdout":
                result.stdout += payload.get("data", "")
            elif event == "stderr":
                result.stderr += payload.get("data", "")
            elif event == "done":
                result.done = True
                result.success = payload.get("success") is not False
                break
    return result


class MolabClient:
    def __init__(self, url: str, token: str, timeout: float = 900.0) -> None:
        self.base = url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {token}"}
        self.timeout = timeout

    def session_id(self) -> str:
        response = requests.get(f"{self.base}/api/sessions", headers=self.headers, timeout=30)
        response.raise_for_status()
        ids = list(response.json())
        if len(ids) != 1:
            raise MolabError(f"cần đúng 1 phiên marimo đang mở, hiện có {len(ids)}: hãy tải lại tab marimo của sandbox")
        return ids[0]

    def run(self, code: str) -> str:
        headers = {**self.headers, "Marimo-Session-Id": self.session_id()}
        with requests.post(
            f"{self.base}/api/kernel/execute", json={"code": code}, headers=headers, stream=True, timeout=self.timeout
        ) as response:
            response.encoding = "utf-8"
            result = parse_stream(response.iter_lines(decode_unicode=True))
        if not result.done:
            raise MolabError("không nhận được kết quả (phiên có thể đã đổi, thử lại)")
        if not result.success:
            raise MolabError(result.stderr[-1500:] or "mã trên sandbox báo lỗi")
        return result.stdout


def remote_md5s(client: MolabClient, rels: list[str]) -> dict[str, str | None]:
    code = (
        "import hashlib, json\nfrom pathlib import Path\n"
        f"rels = json.loads({json.dumps(json.dumps(rels))})\n"
        f'root = Path("{REMOTE_ROOT}")\n'
        "print(json.dumps({r: (hashlib.md5((root / r).read_bytes()).hexdigest() if (root / r).is_file() else None) for r in rels}))\n"
    )
    return json.loads(client.run(code).strip().splitlines()[-1])


def batches(files: dict[str, bytes], limit: int = CHUNK_BYTES) -> Iterator[dict[str, str]]:
    current: dict[str, str] = {}
    size = 0
    for rel, data in files.items():
        blob = base64.b64encode(zlib.compress(data, 9)).decode("ascii")
        if current and size + len(blob) > limit:
            yield current
            current, size = {}, 0
        current[rel] = blob
        size += len(blob)
    if current:
        yield current


def write_remote(client: MolabClient, files: dict[str, bytes], force: bool) -> list[str]:
    done: list[str] = []
    for batch in batches(files):
        code = (
            "import base64, hashlib, json, zlib\nfrom pathlib import Path\n"
            f"force = {force!r}\nroot = Path(\"{REMOTE_ROOT}\")\nout = []\n"
            f"for rel, blob in json.loads({json.dumps(json.dumps(batch))}).items():\n"
            "    data = zlib.decompress(base64.b64decode(blob)); dest = root / rel\n"
            "    if dest.exists() and not force and hashlib.md5(dest.read_bytes()).hexdigest() != hashlib.md5(data).hexdigest():\n"
            "        out.append('SKIP ' + rel); continue\n"
            "    dest.parent.mkdir(parents=True, exist_ok=True); dest.write_bytes(data); out.append('OK ' + rel)\n"
            "print(json.dumps(out))\n"
        )
        done += json.loads(client.run(code).strip().splitlines()[-1])
    return done


INSTALL_CODE = '''
import importlib.util, subprocess, sys
need = [m for m in ("soundfile", "torchaudio", "timm") if importlib.util.find_spec(m) is None]
if need:
    import torch
    pins = {"torchaudio": "torchaudio==" + torch.__version__.split("+")[0]}
    pkgs = [pins.get(m, m) for m in need]
    r = subprocess.run(["uv", "pip", "install", "--no-deps", "--python", sys.executable, *pkgs], capture_output=True, text=True)
    print("cai dat", pkgs, "exit", r.returncode); print((r.stdout + r.stderr)[-400:])
else:
    print("da du module: soundfile, torchaudio, timm")
r = subprocess.run([sys.executable, "-c", "import torch, torchaudio, timm, soundfile, scripts.sweep_frcrn; print('torch', torch.__version__, '| import repo ok')"],
                   cwd="/marimo", capture_output=True, text=True)
print((r.stdout + r.stderr).strip()[-300:])
'''

PRETRAIN_CODE = '''
import sys
sys.path.insert(0, "/marimo")
from scripts.load_pretrain import download_checkpoint, resolve_url
from scripts.sweep_frcrn import ORIGINAL_CHECKPOINT
path = ORIGINAL_CHECKPOINT if ORIGINAL_CHECKPOINT.is_file() else download_checkpoint(resolve_url("frcrn-pretrain"), ORIGINAL_CHECKPOINT.parent)
print("pretrain:", path, round(path.stat().st_size / 1e6, 1), "MB")
'''


MSYS_PREFIXES = ("C:/Program Files/Git", "C:/Program Files (x86)/Git")


def fix_msys_path(value: str) -> str:
    """Git Bash đổi đối số `/marimo/x` thành `C:/Program Files/Git/marimo/x`: trả lại đường dẫn Linux của sandbox."""
    for prefix in MSYS_PREFIXES:
        if value.startswith(prefix + "/"):
            return value[len(prefix) :]
    return value


def work_dir_from(args: list[str]) -> str:
    return args[args.index("--work-dir") + 1] if "--work-dir" in args else DEFAULT_WORK_DIR


def launch_code(
    args: list[str], kaggle_token: str | None, module: str = "scripts.run_frcrn_noise", work_dir: str | None = None
) -> str:
    env_line = f'env["KAGGLE_ACCESS_TOKEN"] = {kaggle_token!r}\n' if kaggle_token else ""
    return (
        "import os, subprocess, sys\nfrom pathlib import Path\n"
        f"args = {json.dumps(args)}\n"
        f"work = Path({work_dir or work_dir_from(args)!r}); work.mkdir(parents=True, exist_ok=True)\n"
        'pid_file = work / "launcher.pid"\n'
        "if pid_file.exists():\n"
        "    try:\n        os.kill(int(pid_file.read_text()), 0); raise SystemExit('da co tien trinh dang chay: ' + pid_file.read_text())\n"
        "    except (ProcessLookupError, ValueError):\n        pass\n"
        'env = dict(os.environ, PYTHONUNBUFFERED="1", TQDM_DISABLE="1", WANDB_SILENT="true")\n'
        f"{env_line}"
        'with open(work / "stdout.log", "ab") as out:\n'
        f'    p = subprocess.Popen([sys.executable, "-m", {module!r}, *args], cwd="/marimo", env=env,\n'
        "                         stdout=out, stderr=subprocess.STDOUT, start_new_session=True)\n"
        "pid_file.write_text(str(p.pid)); print('da khoi chay pid', p.pid, '| log:', work / 'stdout.log')\n"
    )


def status_code(work_dir: str) -> str:
    return (
        "import json, os\nfrom pathlib import Path\n"
        f"work = Path({work_dir!r})\n"
        "pid = int((work / 'launcher.pid').read_text()) if (work / 'launcher.pid').exists() else None\n"
        "alive = False\n"
        "if pid:\n    try: os.kill(pid, 0); alive = True\n    except OSError: pass\n"
        "print('pid', pid, 'dang chay' if alive else 'khong chay')\n"
        "m = work / 'run_manifest.json'\n"
        "if m.exists():\n"
        "    d = json.loads(m.read_text(encoding='utf-8'))\n"
        "    print('trang thai:', d['status'], '| bat dau:', d['started_at'], '| ket thuc:', d['finished_at'])\n"
        "    for e in d['epochs'][-3:]: print('  epoch', e['epoch'], e['time'], {k: e[k] for k in ('train_loss', 'loss', 'noise_si_snr', 'noise_si_snr_hi') if k in e})\n"
        "    print('drive:', d['drive']['stage_dir'], [(f['name'], f['size']) for f in d['drive']['files']])\n"
        "log = work / 'stdout.log'\n"
        "if log.exists():\n    print('--- log (cuoi):')\n"
        "    for l in log.read_text(encoding='utf-8', errors='ignore').splitlines()[-6:]: print(l[:200].encode('ascii', 'replace').decode())\n"
    )


def drive_paths(drive_dir: str, stage: str = "frcrn") -> dict[str, str]:
    base = drive_dir.strip("/")
    return {
        "dir": base,
        "manifest": f"{base}/run_manifest.json",
        "stage_dir": f"{base}/{stage}",
        "best": f"{base}/{stage}/best.pt",
        "resume": f"{base}/{stage}/resume.pt",
        "result": f"{base}/{stage}/result.json",
        "history": f"{base}/{stage}/best_e<epoch>_<UTC>.pt",
    }


REDEPLOY_TEMPLATE = """# Tái triển khai lần chạy FRCRN (target noise) lên molab

Hồ sơ lưu lúc {created_at}. Sandbox ghi nhận: `{url}`.

## Vị trí trên Google Drive
- Thư mục: `{dir}`
- Hồ sơ lần chạy (tiến độ, mốc thời gian từng epoch): `{manifest}`
- Bản tốt nhất mới nhất: `{best}`
- Bản lịch sử có mốc thời gian: `{history}`
- Trạng thái để nối tiếp (trọng số, optimizer, lịch sử epoch): `{resume}`
- Kết quả cuối (chỉ có khi chạy xong): `{result}`

## Code đã dùng
- Commit local: `{commit}` ({dirty} tệp chưa commit); mã băm code: `{code_hash}`.
- Bản đóng gói đúng các tệp đã dùng (kèm bảng offset): `{archive}`.

## Các bước
1. Tạo sandbox molab mới, mở notebook, chạy cell đăng nhập Google Drive (token được lưu trên sandbox).
2. Trên máy này:
   ```bash
   export MARIMO_TOKEN=<token của sandbox mới>
   python -m scripts.molab_deploy push --url <URL sandbox mới> --archive {archive}
   export KAGGLE_ACCESS_TOKEN=<token Kaggle>   # chỉ cần nếu sandbox chưa có dữ liệu
   python -m scripts.molab_deploy launch --url <URL sandbox mới> --kaggle -- {run_args}
   python -m scripts.molab_deploy status --url <URL sandbox mới>
   ```
3. Cùng thư mục Drive nên script tự khôi phục `resume.pt`/`best.pt` và nối tiếp từ epoch đã xong. Nếu `result.json` đã có
   thì lần chạy coi như xong; muốn chạy lại từ đầu hãy dùng thư mục Drive mới.
"""


def build_record(
    root: Path,
    out_dir: Path,
    *,
    url: str,
    drive_dir: str,
    run_args: list[str],
    stage: str = "frcrn",
    git: dict[str, Any] | None = None,
) -> Path:
    stamp = now_iso().replace(":", "").replace("-", "").split("+")[0] + "Z"
    record_dir = out_dir / f"frcrn-noise-{stamp}"
    archive = record_dir / "code.tar.gz"
    files = code_files(root, offset_files(root))
    prints = archive_code(archive, root, files)
    paths = drive_paths(drive_dir, stage)
    git = git if git is not None else git_state(root)
    record = {
        "created_at": now_iso(),
        "sandbox_url": url,
        "drive": paths,
        "run_command": [*RUN_MODULE, *run_args],
        "git": git,
        "code_hash": code_hash(prints),
        "code_files": prints,
        "archive": archive.name,
    }
    (record_dir / "record.json").write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    (record_dir / "REDEPLOY.md").write_text(
        REDEPLOY_TEMPLATE.format(
            created_at=record["created_at"],
            url=url,
            commit=git.get("commit"),
            dirty=git.get("uncommitted_files"),
            code_hash=record["code_hash"],
            archive=archive.as_posix(),
            run_args=" ".join(run_args),
            **paths,
        ),
        encoding="utf-8",
    )
    return record_dir


def client_from_env(url: str) -> MolabClient:
    token = os.environ.get("MARIMO_TOKEN")
    if not token:
        raise SystemExit("Đặt biến môi trường MARIMO_TOKEN (token của sandbox); không truyền token qua dòng lệnh")
    return MolabClient(url, token)


def split_remainder(argv: list[str]) -> tuple[list[str], list[str]]:
    return (argv[: argv.index("--")], argv[argv.index("--") + 1 :]) if "--" in argv else (argv, [])


def main(argv: list[str] | None = None) -> None:
    own, run_args = split_remainder([fix_msys_path(a) for a in (sys.argv[1:] if argv is None else argv)])
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    record = sub.add_parser("record", help="lưu hồ sơ và đóng gói code xuống máy này")
    record.add_argument("--url", required=True)
    record.add_argument("--drive-dir", required=True)
    record.add_argument("--stage", default="frcrn")
    record.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    push = sub.add_parser("push", help="đẩy code đã đóng gói lên sandbox, cài module, tải checkpoint pretrain")
    push.add_argument("--url", required=True)
    push.add_argument("--archive", type=Path, required=True)
    push.add_argument("--force", action="store_true", help="ghi đè cả tệp đã có nhưng khác nội dung")
    launch = sub.add_parser("launch", help="khởi chạy script chạy nền trên sandbox (đối số sau --)")
    launch.add_argument("--url", required=True)
    launch.add_argument("--module", default="scripts.run_frcrn_noise", help="module python chạy trên sandbox")
    launch.add_argument("--work-dir", default=None, help="thư mục chứa launcher.pid và stdout.log")
    launch.add_argument("--kaggle", action="store_true", help="truyền KAGGLE_ACCESS_TOKEN của máy này vào tiến trình")
    status = sub.add_parser("status", help="xem tiến độ lần chạy trên sandbox")
    status.add_argument("--url", required=True)
    status.add_argument("--work-dir", default=DEFAULT_WORK_DIR)
    args = parser.parse_args(own)

    if args.command == "record":
        record_dir = build_record(ROOT, args.out_dir, url=args.url, drive_dir=args.drive_dir, run_args=run_args, stage=args.stage)
        print("đã lưu hồ sơ:", record_dir)
        return
    client = client_from_env(args.url)
    if args.command == "push":
        files = read_archive(args.archive)
        remote = remote_md5s(client, list(files))
        todo = {rel: data for rel, data in files.items() if remote.get(rel) != md5_of(data)}
        print(f"{len(files)} tệp trong gói: {len(files) - len(todo)} đã khớp, {len(todo)} cần ghi")
        if todo:
            results = write_remote(client, todo, args.force)
            skipped = [r for r in results if r.startswith("SKIP")]
            print(f"đã ghi {len(results) - len(skipped)}, bỏ qua {len(skipped)} tệp khác nội dung (dùng --force để ghi đè):", skipped[:5])
        print(client.run(INSTALL_CODE).strip())
        print(client.run(PRETRAIN_CODE).strip())
    elif args.command == "launch":
        token = os.environ.get("KAGGLE_ACCESS_TOKEN") if args.kaggle else None
        if args.kaggle and not token:
            raise SystemExit("--kaggle cần biến môi trường KAGGLE_ACCESS_TOKEN trên máy này")
        print(client.run(launch_code(run_args, token, args.module, fix_msys_path(args.work_dir) if args.work_dir else None)).strip())
    else:
        print(client.run(status_code(args.work_dir)).strip())


if __name__ == "__main__":
    main()
