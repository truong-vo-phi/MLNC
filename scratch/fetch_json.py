import subprocess

code = 'print(open("/marimo/output/student_only_benchmark.json").read())'
res = subprocess.check_output(["python", "scratch/py_exec.py", "-c", code], text=True)
with open("experiments/results/student_only_benchmark.json", "w", encoding="utf-8") as f:
    f.write(res)
print("Saved! Size:", len(res))
