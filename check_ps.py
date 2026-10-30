import subprocess
res = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
print(res.stdout)
