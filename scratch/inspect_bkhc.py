import marimo._code_mode as cm

async with cm.get_context() as ctx:
    cell = ctx.cells["bkHC"]
    print("Code in bkHC:")
    print(cell.code)
