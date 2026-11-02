import sys
sys.path.insert(0, r"D:\PROJECT\MLNC\imp\kd-moe-ced")
from deploy_to_sandbox import execute_remote

check_script = """
import marimo._code_mode as cm

async def final_check():
    async with cm.get_context() as ctx:
        print('Total cells:', len(ctx.cells))
        for cid, cell in ctx.cells.items():
            print(f'Cell {cid}: status={cell.status}, errors={cell.errors}')

await final_check()
"""

execute_remote(check_script)
