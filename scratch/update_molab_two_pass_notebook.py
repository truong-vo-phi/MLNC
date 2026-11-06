import json
from pathlib import Path
import marimo._code_mode as cm

updates = {
"MJUe": '''from pathlib import Path

for _p in (Path("/dataset/mix-dataset-v2/train/manifest.csv"), Path("/dataset/mix-dataset-v2/eval/manifest.csv"), Path("/dataset/mix-dataset-v2/labels.txt"), Path("/dataset/cache/train.npy"), Path("/dataset/cache/validation.npy")):
    if not _p.is_file():
        raise FileNotFoundError(f"Missing required dataset file: {_p}")
mo.md("Dataset ready: `/dataset/mix-dataset-v2` and FRCRN cache `/dataset/cache`.")''',
"overview": '''import torch
_gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
mo.md(f"""# Two-pass CE benchmark: 9 models

GPU: **{_gpu_name}**. Both passes use the same model implementations and independent pretrained initialization.

- Mixture: `/dataset/mix-dataset-v2` WAVs.
- FRCRN: `/dataset/cache` train and validation arrays, no external RMS normalization.
- 30,240 train and 3,064 validation samples; test is not used.
- The FRCRN cache has no sample IDs: its alignment with the manifest is assumed, not independently verified.
""")''',
"training_config": '''RUN_TRAINING = True
MODEL_NAMES = ("panns_cnn10", "ced_tiny", "ced_mini", "ced_small", "efficientat_mn10", "vit_small", "mobilevit_s", "mobilenet_v3_large", "beats12")
mo.md("""### Experiment settings

Run all will prepare prerequisites if needed, then train mixture and FRCRN independently, in that order, with all 9 models. Each uses 15 epochs, seed 2026, effective batch 64, pretrained backbone, unweighted CE, no Mixup or SpecAugment. Best checkpoint is selected by validation accuracy; ties keep the earlier epoch. Results and checkpoints are kept separately by input.
""")''',
"ce_v2_guide": '''mo.md("""### Inputs and outputs

Outputs: `/marimo/review-3/ce_v2/benchmark_results/two_pass_ce9/{mixture,frcrn}`. Preparation artifacts are under `ce9_clean_mixture_20261102`. Completed models are skipped and interrupted models resume from their last completed epoch. No test split or ensemble is used.
""")''',
"architecture_summary": '''mo.md("### Models: " + ", ".join(f"`{_m}`" for _m in MODEL_NAMES))''',
"train_progress": '''import os as _os, subprocess as _subprocess, sys as _sys
from pathlib import Path as _Path

benchmark_report = None
_root = _Path("/marimo/review-3/ce_v2")
_shared = _root / "benchmark_results/ce9_clean_mixture_20261102"

def _stream(_args, _env):
    _p = _subprocess.Popen(_args, cwd=_root, env=_env, stdout=_subprocess.PIPE, stderr=_subprocess.STDOUT, text=True, bufsize=1)
    _lines = []
    for _line in _p.stdout:
        print(_line, end="", flush=True)
        _lines.append(_line)
        mo.output.replace(mo.vstack([mo.md("### Two-pass training progress"), mo.plain_text("".join(_lines[-25:]))]))
    _code = _p.wait()
    if _code:
        raise RuntimeError(f"Command failed ({_code}): {_args}; see console output")

if RUN_TRAINING:
    if not (_shared / "model_smoke_checks.json").is_file() or not (_shared / "pretrained_provenance.json").is_file():
        _stream([_sys.executable, "-u", "prepare_ce9.py"], _os.environ.copy())
    for _mode in ("mixture", "frcrn"):
        print(f"\\nSTART PASS: {_mode}\\n", flush=True)
        _env = _os.environ.copy()
        _env["CE_INPUT_MODE"] = _mode
        _stream([_sys.executable, "-u", "train_two_pass.py"], _env)
    benchmark_report = {"completed": True}
else:
    print("Training disabled")
mo.md("Training completed. Results are in the two_pass_ce9 directory." if benchmark_report else "Training has not completed.")''',
"comparison_table": '''import json as _json
from pathlib import Path as _Path

_base = _Path("/marimo/review-3/ce_v2/benchmark_results/two_pass_ce9")
_rows = []
for _name in MODEL_NAMES:
    _row = {"model": _name}
    for _mode in ("mixture", "frcrn"):
        _path = _base / _mode / _name / "results.json"
        if _path.is_file():
            _result = _json.loads(_path.read_text())
            _metrics = _result["validation"]["all"]
            _row[_mode + " acc"] = round(_metrics["acc"] * 100, 2)
            _row[_mode + " macro-F1"] = round(_metrics["macro_f1"], 4)
        else:
            _row[_mode + " acc"] = None
            _row[_mode + " macro-F1"] = None
    _rows.append(_row)
mo.vstack([mo.md("### Validation comparison"), mo.ui.table(_rows)])''',
}

async with cm.get_context() as ctx:
    before = {key: ctx.cells[key].code for key in updates}
    backup = Path('/marimo/review-3/ce_v2/notebook_cells_before_two_pass.json')
    if not backup.exists():
        backup.write_text(json.dumps(before, indent=2, ensure_ascii=False), encoding='utf-8')
    for key, code in updates.items():
        ctx.edit_cell(ctx.cells[key].id, code=code)
print("Updated notebook cells without running training")
