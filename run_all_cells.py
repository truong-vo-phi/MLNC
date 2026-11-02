import sys
sys.path.insert(0, r"D:\PROJECT\MLNC\imp\kd-moe-ced")
from deploy_to_sandbox import execute_remote

run_all_script = """
import marimo._code_mode as cm

async def run_all():
    async with cm.get_context() as ctx:
        order = list(ctx.cells.keys())
        for cid in order:
            if cid not in ["dSpg", "iLit"]:  # dSpg is download (already done), iLit is already done
                ctx.run_cell(cid)

await run_all()
"""

rc = execute_remote(run_all_script)
print(f"Run all executed with code {rc}")
