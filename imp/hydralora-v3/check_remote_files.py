from deploy_to_sandbox import execute_remote

code = """
import os
print("checkpoints:", os.listdir("/marimo/checkpoints"))
print("imp/hydralora-v3:", os.listdir("/marimo/imp/hydralora-v3"))
if os.path.exists("/marimo/imp/hydralora-v3/checkpoints"):
    print("imp/hydralora-v3/checkpoints:", os.listdir("/marimo/imp/hydralora-v3/checkpoints"))
"""
execute_remote(code)
