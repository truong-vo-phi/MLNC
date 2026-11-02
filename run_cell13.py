import marimo._code_mode as cm

async with cm.get_context() as ctx:
    ctx.run_cell("nWHF")

print("Cell 13 (nWHF) executed successfully!")
