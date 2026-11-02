import marimo._code_mode as cm

async with cm.get_context() as ctx:
    print(f"Cell iLit status: {ctx.cells['iLit'].status}")
    print(f"Cell ROlb status: {ctx.cells['ROlb'].status}")
