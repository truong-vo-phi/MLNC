# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo", "gdrive-fsspec"]
# ///
import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import json
    import os
    from pathlib import Path
    import subprocess
    import sys
    mo.md("""
    # Review-3: 9 pretrained CE-only models
    Fork this notebook into your Molab workspace and enable GPU in notebook specs.
    Run the numbered buttons in order. Data: raw mixture v2, train/eval only.
    No KD, MoE or FRCRN. Training starts only when you press **5. Train**.
    """)
    return Path, json, mo, os, subprocess, sys


@app.cell
def _(mo):
    setup = mo.ui.run_button(label="1. Set up code and dependencies")
    setup
    return (setup,)


@app.cell
def _(Path, mo, setup, subprocess, sys):
    mo.stop(not setup.value)
    repo = Path("/tmp/mlnc-review3/MLNC")
    branch = "vuong_mix_v2_group_config_20261102"
    if not repo.exists():
        repo.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", "--branch", branch,
                        "https://github.com/truong-vo-phi/MLNC.git", str(repo)], check=True)
    assert (repo / "imp/review-3/config/pretrained.yaml").is_file(), "Unexpected existing checkout"
    work = repo / "imp/review-3"
    import shutil
    import torch
    import importlib.util
    installer = ([shutil.which("uv"), "pip", "install", "--python", sys.executable]
                 if shutil.which("uv") else [sys.executable, "-m", "pip", "install"])
    constraint = work / "logs/pretrained/molab_torch_constraint.txt"
    constraint.parent.mkdir(parents=True, exist_ok=True)
    constraint.write_text("torch==" + torch.__version__ + "\n")
    subprocess.run(installer + ["-c", str(constraint), "-r", str(work / "requirements-molab.txt")], check=True)
    if importlib.util.find_spec("torchaudio") is None:
        cuda = torch.version.cuda
        assert cuda, "Molab GPU PyTorch build required"
        index = "https://download.pytorch.org/whl/cu" + cuda.replace(".", "")
        subprocess.run(installer + ["--no-deps", "--index-url", index,
                                   "torchaudio==" + torch.__version__.split("+")[0]], check=True)
    import torchaudio
    import torchvision
    print("Audio/vision:", torchaudio.__version__, torchvision.__version__)
    assert torch.cuda.is_available(), "Enable Molab GPU and rerun setup; keep its compatible PyTorch stack"
    print("GPU:", torch.cuda.get_device_name(0), "PyTorch:", torch.__version__)
    print("Code commit:", subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip())
    print("Working directory:", work)
    return (work,)


@app.cell
def _(mo):
    prepare = mo.ui.run_button(label="2. Download clean raw data and verified weights")
    prepare
    return (prepare,)


@app.cell
def _(mo, prepare, subprocess, sys, work):
    mo.stop(not prepare.value)
    subprocess.run([sys.executable, "-u", "main.py", "prepare", "--remove-archive"], cwd=work, check=True)
    return


@app.cell
def _(mo):
    preflight = mo.ui.run_button(label="3. Check all pretrained models (CPU, no training)")
    preflight
    return (preflight,)


@app.cell
def _(mo, preflight, subprocess, sys, work):
    mo.stop(not preflight.value)
    subprocess.run([sys.executable, "-u", "main.py", "preflight", "--model", "all", "--device", "cpu"],
                   cwd=work, check=True)
    return


@app.cell
def _(mo):
    login = mo.ui.run_button(label="4. Authorize your Google Drive backups")
    login
    return (login,)


@app.cell
def _(login, mo):
    mo.stop(not login.value)
    from gdrive_fsspec import GoogleDriveFileSystem
    drive = GoogleDriveFileSystem(use_listings_cache=False, skip_instance_cache=True,
                                 auth_kwargs={"use_local_webserver": False})
    drive.ls("")
    cached_drive = GoogleDriveFileSystem(use_listings_cache=False, skip_instance_cache=True, token="cache")
    cached_drive.ls("")
    drive_ready = True
    print("Drive authorization checked. Do not share credentials in chat.")
    return (drive_ready,)


@app.cell
def _(mo):
    model = mo.ui.dropdown(options=["beats", "pann_cnn10", "ced_tiny", "ced_mini", "ced_small",
                                   "efficient_mn10", "vit_s", "mobilevit", "mobilenet", "all"], value="beats", label="Model")
    drive_folder = mo.ui.text(value="MLNC/review3_pretrained_v2", label="Your Drive folder (reuse to resume)")
    start = mo.ui.run_button(label="5. Train / resume")
    mo.vstack([model, drive_folder, start])
    return drive_folder, model, start


@app.cell
def _(Path, drive_folder, drive_ready, json, mo, model, os, start, subprocess, sys, work):
    mo.stop(not start.value)
    assert drive_ready
    checks = json.loads((work / "logs/pretrained/preflight.json").read_text())
    assert len(checks) == 9 and all(v["verified"] for v in checks.values()), "Complete step 3 first"
    assert drive_folder.value.strip(), "Drive folder required"
    pid_file = work / "logs/pretrained/molab.pid"
    if pid_file.exists():
        try:
            os.kill(int(pid_file.read_text()), 0)
        except ProcessLookupError:
            pass
        else:
            state = Path(f"/proc/{int(pid_file.read_text())}/stat").read_text().split()[2]
            if state != "Z":
                raise RuntimeError("A training process is already active; check step 6")
    command = [sys.executable, "-u", "main.py", "train", "--model", model.value,
               "--config", "config/pretrained.yaml", "--device", "cuda",
               "--drive-dir", drive_folder.value.strip()]
    with (work / "logs/pretrained/molab_training.log").open("a") as output:
        worker = subprocess.Popen(command, cwd=work, stdout=output, stderr=subprocess.STDOUT,
                                  start_new_session=True)
    pid_file.write_text(str(worker.pid))
    print("Started PID", worker.pid, "; monitor below. A Molab session can still expire.")
    return


@app.cell
def _(mo):
    refresh = mo.ui.run_button(label="6. Refresh training log")
    refresh
    return (refresh,)


@app.cell
def _(mo, refresh, work):
    mo.stop(not refresh.value)
    log = work / "logs/pretrained/molab_training.log"
    print("\n".join(log.read_text(errors="replace").splitlines()[-60:]) if log.exists() else "Not started")
    print("Results:", work / "checkpoints/pretrained")
    return


if __name__ == "__main__":
    app.run()
