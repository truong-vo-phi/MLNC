import base64, json, sys, time
from pathlib import Path
sys.path.insert(0, "/marimo")
cells = json.loads(base64.b64decode(Path("/marimo/output/_cells_b64.txt").read_text()))
code = {c["code"].split("\n")[0].split(")")[0] + ")": c["code"] for c in cells if not c["code"].startswith("mo.md")}
ns = {"__name__": "__main__"}
exec("import warnings; warnings.filterwarnings('ignore')\n", ns)
for key in ("# (0)", "# (1)", "# (2)"):
    exec(code[key], ns); print(key, "ok", flush=True)
ns["fs"] = None  # khong dang nhap Drive trong tien trinh nen
for key in ("# (3)", "# (4)", "# (5)", "# (5b)", "# (6)"):
    t0 = time.time(); exec(code[key], ns); print(key, "ok", round(time.time() - t0, 1), "s", flush=True)
print("use_cache:", ns["use_cache"](), flush=True)
t0 = time.time(); exec(code["# (6b)"], ns); print("(6b) xong", round(time.time() - t0), "s", flush=True)
print("DONE", flush=True)
