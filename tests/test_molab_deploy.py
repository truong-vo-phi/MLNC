import json

import pytest

import scripts.molab_deploy as deploy
from scripts.run_manifest import (
    RunManifest,
    archive_code,
    code_files,
    code_hash,
    fingerprint,
    md5_of,
    offset_files,
    read_archive,
)


def make_repo(tmp_path):
    files = {
        "src/a.py": "x = 1\n",
        "src/utils/download.py": 'KAGGLE_ACCESS_TOKEN = "KGAT_secret"\n',
        "src/__pycache__/junk.py": "junk\n",
        "scripts/b.py": "y = 2\n",
        "config/c.yaml": "k: v\n",
        "output/audioset-ref/offsets_train.csv": "sample_id,offset,error\ns1,5,0.001\n",
        "tests/t.py": "not shipped\n",
    }
    for rel, text in files.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return tmp_path


def test_parse_stream_collects_stdout_stderr_and_done():
    lines = [
        "event: stdout",
        'data: {"data": "xin "}',
        "",
        "event: stdout",
        'data: {"data": "chao\\n"}',
        "",
        "event: stderr",
        'data: {"data": "warn"}',
        "",
        "event: done",
        'data: {"success": true}',
    ]
    result = deploy.parse_stream(lines)
    assert result.stdout == "xin chao\n" and result.stderr == "warn" and result.done and result.success


def test_parse_stream_reports_failure_and_missing_done():
    failed = deploy.parse_stream(["event: done", 'data: {"success": false}'])
    assert failed.done and not failed.success
    assert not deploy.parse_stream(["event: stdout", 'data: {"data": "x"}']).done


def test_code_files_excludes_secret_cache_and_tests(tmp_path):
    root = make_repo(tmp_path)
    files = code_files(root, offset_files(root))
    assert files == ["config/c.yaml", "output/audioset-ref/offsets_train.csv", "scripts/b.py", "src/a.py"]


def test_archive_roundtrip_and_fingerprint(tmp_path):
    root = make_repo(tmp_path)
    prints = archive_code(tmp_path / "out" / "code.tar.gz", root)
    assert prints == fingerprint(root)
    contents = read_archive(tmp_path / "out" / "code.tar.gz")
    assert set(contents) == set(prints) and all(md5_of(data) == prints[rel] for rel, data in contents.items())
    assert b"KGAT_secret" not in b"".join(contents.values())
    assert code_hash(prints) == code_hash(dict(reversed(list(prints.items()))))


def test_run_manifest_records_epochs_and_backs_up_every_save(tmp_path):
    root = make_repo(tmp_path)
    saved = []
    manifest = RunManifest.start(
        tmp_path / "run" / "run_manifest.json",
        spec={"settings": {"target": "noise"}},
        drive_dir="/a/b/",
        stage_name="frcrn",
        command=["python", "-m", "x"],
        root=root,
        backup=lambda path: saved.append(path.name),
        list_drive=lambda: [{"name": "best.pt", "size": 5, "modified": ""}],
    )
    manifest.add_epoch(1, {"train_loss": 2.0, "loss": 1.5, "noise_si_snr_hi": -3.0, "tag": "bo qua"})
    manifest.finish({"best_epoch": 1, "best_score": -3.0, "selection_metric": "noise_si_snr_hi", "history": []})
    data = json.loads((tmp_path / "run" / "run_manifest.json").read_text(encoding="utf-8"))
    assert data["status"] == "finished" and data["finished_at"] and data["run_id"].startswith("frcrn-noise-")
    assert data["epochs"][0]["epoch"] == 1 and data["epochs"][0]["time"] and "tag" not in data["epochs"][0]
    assert data["result"]["best_epoch"] == 1 and "history" not in data["result"]
    assert data["drive"]["best"] == "a/b/frcrn/best.pt" and data["drive"]["files"][0]["name"] == "best.pt"
    assert data["code_hash"] == code_hash(fingerprint(root)) and "src/utils/download.py" not in data["code_files"]
    assert saved == ["run_manifest.json"] * 3


def test_run_manifest_survives_drive_listing_errors(tmp_path):
    def broken():
        raise OSError("drive offline")

    manifest = RunManifest.start(
        tmp_path / "m.json", spec={}, drive_dir="d", stage_name="s", root=make_repo(tmp_path), list_drive=broken
    )
    manifest.finish(None, status="failed")
    assert json.loads((tmp_path / "m.json").read_text(encoding="utf-8"))["status"] == "failed"


def test_build_record_writes_archive_record_and_redeploy_notes(tmp_path):
    root = make_repo(tmp_path / "repo")
    record_dir = deploy.build_record(
        root,
        tmp_path / "records",
        url="https://sb-x.sb.molab.run/",
        drive_dir="FA26/Truong/frcrn-noise",
        run_args=["--epochs", "15", "--use-drive"],
        git={"commit": "abc123", "uncommitted_files": 2},
    )
    record = json.loads((record_dir / "record.json").read_text(encoding="utf-8"))
    assert record["drive"]["resume"] == "FA26/Truong/frcrn-noise/frcrn/resume.pt"
    assert record["run_command"] == ["python", "-m", "scripts.run_frcrn_noise", "--epochs", "15", "--use-drive"]
    assert record["git"]["commit"] == "abc123" and (record_dir / "code.tar.gz").is_file()
    notes = (record_dir / "REDEPLOY.md").read_text(encoding="utf-8")
    assert "FA26/Truong/frcrn-noise/frcrn/best.pt" in notes and "--epochs 15 --use-drive" in notes
    assert "KGAT_" not in json.dumps(record) + notes


def test_generated_remote_code_compiles_and_token_only_when_given():
    plain = deploy.launch_code(["--smoke", "--work-dir", "/tmp/w"], None)
    compile(plain, "launch", "exec")
    assert "KAGGLE_ACCESS_TOKEN" not in plain and "/tmp/w" in plain
    with_token = deploy.launch_code(["--smoke"], "tok-en")
    compile(with_token, "launch", "exec")
    assert "tok-en" in with_token
    for code in (deploy.status_code("/marimo/output/x"), deploy.INSTALL_CODE, deploy.PRETRAIN_CODE):
        compile(code, "remote", "exec")


def test_batches_split_by_size():
    import random

    files = {f"f{i}.bin": random.Random(i).randbytes(20_000) for i in range(6)}  # không nén được
    parts = list(deploy.batches(files, limit=5_000))
    assert sum(len(p) for p in parts) == 6 and all(len(p) == 1 for p in parts)
    assert len(list(deploy.batches(files, limit=10**9))) == 1


class FakeResponse:
    def __init__(self, lines=None, payload=None):
        self._lines, self._payload, self.encoding = lines or [], payload, None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload

    def iter_lines(self, decode_unicode=True):
        return iter(self._lines)


def test_client_runs_code_with_single_session(monkeypatch):
    seen = {}
    monkeypatch.setattr(deploy.requests, "get", lambda url, **kw: FakeResponse(payload={"s_1": {}}))

    def post(url, **kw):
        seen.update(url=url, headers=kw["headers"], code=kw["json"]["code"])
        return FakeResponse(lines=["event: stdout", 'data: {"data": "ok"}', "event: done", 'data: {"success": true}'])

    monkeypatch.setattr(deploy.requests, "post", post)
    client = deploy.MolabClient("https://sb.example/", "TOKEN")
    assert client.run("print(1)") == "ok"
    assert seen["url"] == "https://sb.example/api/kernel/execute" and seen["headers"]["Marimo-Session-Id"] == "s_1"
    assert seen["headers"]["Authorization"] == "Bearer TOKEN" and seen["code"] == "print(1)"


def test_client_requires_exactly_one_session(monkeypatch):
    monkeypatch.setattr(deploy.requests, "get", lambda url, **kw: FakeResponse(payload={}))
    with pytest.raises(deploy.MolabError, match="1 phiên"):
        deploy.MolabClient("https://sb.example", "T").session_id()


def test_fix_msys_path_restores_sandbox_paths():
    assert deploy.fix_msys_path("C:/Program Files/Git/marimo/output/x") == "/marimo/output/x"
    assert deploy.fix_msys_path("/marimo/output/x") == "/marimo/output/x"
    assert deploy.fix_msys_path("--epochs") == "--epochs"
