import marimo._code_mode as cm

async def print_cell(cid):
    async with cm.get_context() as ctx:
        c = ctx.cells[cid]
        print(f"=== CELL {cid} ===")
        print(c.code)

await print_cell("vblA")
