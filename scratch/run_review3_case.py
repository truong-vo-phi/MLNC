import os, sys, subprocess, pathlib

ROOT = pathlib.Path("/marimo/review-3")
cmd = [
    sys.executable,
    str(ROOT / "scripts/run_moe_ablations.py"),
    "--case", "with_kd",
]

print("=== BAT DAU LUOT CHAY 4: CED-SingleMoE-SplitFFN (review-3 config) ===")
print("Command:", " ".join(cmd))
print("Working dir:", ROOT)

proc = subprocess.Popen(
    cmd,
    cwd=str(ROOT),
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
)

for line in iter(proc.stdout.readline, ""):
    sys.stdout.write(line)
    sys.stdout.flush()

proc.wait()
print(f"\nLUOT CHAY 4 HOAN TAT voi exit code {proc.returncode}!")
