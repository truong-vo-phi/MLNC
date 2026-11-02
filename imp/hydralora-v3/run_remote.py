import sys
from pathlib import Path

from deploy_to_sandbox import execute_remote

def run_script(script_path: str, timeout: int = 1800):
    code = f"""
with open('{script_path}', encoding='utf-8') as f:
    exec(f.read())
"""
    return execute_remote(code, timeout=timeout)

if __name__ == "__main__":
    script = sys.argv[1] if len(sys.argv) > 1 else "/marimo/imp/hydralora-v3/test_molab_step.py"
    sys.exit(run_script(script, timeout=1800))
