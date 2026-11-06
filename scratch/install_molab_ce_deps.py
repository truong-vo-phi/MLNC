import marimo._code_mode as cm

async with cm.get_context() as ctx:
    ctx.packages.add("loguru>=0.7.3")
print("Requested loguru installation")
