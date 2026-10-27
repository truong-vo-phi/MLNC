import base64, json, sys, time
from pathlib import Path
sys.path.insert(0, "/marimo")
cells = json.loads(base64.b64decode(Path("/marimo/output/_cells_b64.txt").read_text()))
code = {c["code"].split("\n")[0].split(")")[0] + ")": c["code"] for c in cells if not c["code"].startswith("mo.md")}
ns = {"__name__": "__main__"}
exec("import warnings; warnings.filterwarnings('ignore')\n", ns)
for key in ("# (0)", "# (1)", "# (2)"):
    exec(code[key], ns)
ns["fs"] = None  # tien trinh nen khong dang nhap Drive duoc: ket qua luu tren sandbox va duoc keo ve may
for key in ("# (3)", "# (4)", "# (5)", "# (5b)", "# (6)", "# (6b)", "# (7)", "# (8)"):
    exec(code[key], ns)
print("cau hinh: RUN_TRAINING=%s USE_LOGIT=%s USE_AUG=%s USE_NHAT_CACHE=%s epochs=%s" % (ns["RUN_TRAINING"], ns["USE_LOGIT"], ns["USE_AUG"], ns["USE_NHAT_CACHE"], ns["TRAIN"]["epochs"]), flush=True)
assert ns["RUN_TRAINING"] and ns["USE_LOGIT"] and ns["USE_AUG"], "cau hinh khong dung voi lan chay du dinh"
exec(code["# (9)"], ns)
print("DONE", flush=True)
