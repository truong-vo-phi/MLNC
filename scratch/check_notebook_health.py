import marimo._code_mode as cm

async with cm.get_context() as ctx:
    print(f"Total cells: {len(ctx.cells)}")
    for cid, cell in ctx.cells.items():
        first_line = cell.code.strip().split("\n")[0] if cell.code else ""
        print(f"Cell {cid:6s} | {cell.status:8s} | code: {first_line[:70]}")
