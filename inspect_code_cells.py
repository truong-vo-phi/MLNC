import sys
sys.path.insert(0, r"D:\PROJECT\MLNC\imp\kd-moe-ced")
from deploy_to_sandbox import execute_remote

inspect_code = """
import marimo._code_mode as cm

async def inspect_code_cells():
    async with cm.get_context() as ctx:
        for cid in ['lEQa', 'Xref', 'BYtC', 'Hstk']:
            print(f"=== CELL {cid} ===")
            print(ctx.cells[cid].code)

await inspect_code_cells()
"""

execute_remote(inspect_code)
