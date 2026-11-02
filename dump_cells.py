import sys
sys.path.insert(0, r"D:\PROJECT\MLNC\imp\kd-moe-ced")
from deploy_to_sandbox import execute_remote

dump_code = """
import marimo._code_mode as cm

async def inspect_all():
    async with cm.get_context() as ctx:
        for idx, (cid, cell) in enumerate(ctx.cells.items()):
            print(f"[{idx}] Cell {cid} (name={cell.name}):")
            lines = cell.code.strip().split('\\n')
            preview = lines[0] if lines else ""
            print(f"    Line count: {len(lines)} | First line: {preview[:70]}")

await inspect_all()
"""

execute_remote(dump_code)
