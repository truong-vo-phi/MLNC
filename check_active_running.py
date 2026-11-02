import sys
sys.path.insert(0, r"D:\PROJECT\MLNC\imp\kd-moe-ced")
from deploy_to_sandbox import execute_remote

check_status_code = """
import marimo._code_mode as cm

async def check():
    async with cm.get_context() as ctx:
        for cid, cell in ctx.cells.items():
            if cell.status not in ['idle']:
                print(f"Cell {cid}: status={cell.status}")
        print("Done status check.")

await check()
"""

execute_remote(check_status_code)
