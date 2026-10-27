import base64, json, sys, time
from pathlib import Path
sys.path.insert(0, "/marimo")
import numpy as np
import pandas as pd

cells = json.loads(base64.b64decode(Path("/marimo/output/_cells_b64.txt").read_text()))
code = {c["code"].split("\n")[0].split(")")[0] + ")": c["code"] for c in cells if not c["code"].startswith("mo.md")}
ns = {"__name__": "__main__"}
exec("import warnings; warnings.filterwarnings('ignore')\n", ns)
for key in ("# (0)", "# (1)", "# (2)"):
    exec(code[key], ns)
ns["fs"] = None
for key in ("# (3)", "# (4)", "# (5)", "# (5b)", "# (6)", "# (6b)", "# (7)"):
    exec(code[key], ns)
torch = ns["torch"]
assert ns["USE_LOGIT"] and ns["USE_AUG"] and ns["USE_NHAT_CACHE"] and ns["RUN_TRAINING"], "cau hinh sai: can USE_LOGIT, USE_AUG, USE_NHAT_CACHE, RUN_TRAINING"

stats = {"calls": 0, "mixed": 0}
original = ns["augment_pair"]
def counted(*args, **kwargs):
    out = original(*args, **kwargs)
    stats["calls"] += 1
    stats["mixed"] += int(out[3])
    assert bool(torch.isfinite(out[0]).all()) and bool(torch.isfinite(out[1]).all()), "augment sinh NaN"
    return out
ns["augment_pair"] = counted
exec(code["# (8)"], ns)
ns["augment_pair"] = counted
ns["TRAIN"]["epochs"] = 1
NAME = "e2e-final-test"

t0 = time.time()
ns["run_experiment"](NAME)
print("lan 1 xong sau %.0f s" % (time.time() - t0), flush=True)
run = ns["run"]
history = pd.read_csv(run.result_dir / f"history_{NAME}.csv")
assert float(history["train_kl"].iloc[0]) > 0, "KD logit khong hoat dong (train_kl = 0)"
print("KD logit hoat dong: train_kl epoch 1 = %.3f" % float(history["train_kl"].iloc[0]), flush=True)
fraction = stats["mixed"] / max(stats["calls"], 1)
assert 0.3 < fraction < 0.7, f"ti le batch tron bat thuong: {fraction}"
print("mixup: %d/%d batch (%.0f%%)" % (stats["mixed"], stats["calls"], 100 * fraction), flush=True)
checkpoint = torch.load(run.checkpoint_path(NAME), map_location="cpu", weights_only=False)
keys = list(checkpoint["model"])
assert any(k.startswith("att.") for k in keys) and any(k.startswith("cla.") for k in keys), "checkpoint thieu lop phan loai att/cla"
assert not any(k.startswith("separator.") for k in keys)
print("checkpoint co att/cla: %d tensor, epoch %d" % (len(keys), checkpoint["epoch"]), flush=True)
analysis = [run.result_dir / f"{prefix}_validation_{NAME}.{ext}" for prefix, ext in (("predictions", "npz"), ("confusion", "csv"), ("per_class", "csv"))]
assert all(p.is_file() for p in analysis), "thieu file phan tich du doan"
probs_first = np.load(analysis[0])["probs"].astype(np.float32)

t0 = time.time()
ns["run_experiment"](NAME)  # resume: epoch 1 da xong nen chi khoi phuc trang thai va danh gia lai
print("lan 2 (resume) xong sau %.0f s" % (time.time() - t0), flush=True)
probs_second = np.load(analysis[0])["probs"].astype(np.float32)
difference = float(np.abs(probs_first - probs_second).max())
assert difference < 5e-3, f"sau resume du doan khac lan 1 (max {difference:.4f}): lop phan loai khong duoc khoi phuc"
print("resume khoi phuc dung ca backbone va att/cla: sai lech xac suat lon nhat %.1e" % difference, flush=True)
print("E2E PASS", flush=True)
