import subprocess
import base64

code = 'import base64; print(base64.b64encode(open("/marimo/output/cell_TqIu.png", "rb").read()).decode())'
b64 = subprocess.check_output(["python", "scratch/py_exec.py", "-c", code], text=True).strip()
with open("experiments/results/molab_notebook_cell_TqIu.png", "wb") as f:
    f.write(base64.b64decode(b64))
print("Saved local screenshot! Size:", len(b64))
