import marimo._code_mode as cm

async def list_cells():
    async with cm.get_context() as ctx:
        cells = ctx.cells
        for cell_id, cell in cells.items():
            print(f"Cell ID: {cell_id}, Name: {cell.name}")

await list_cells()
