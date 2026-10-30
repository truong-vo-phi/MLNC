import marimo._code_mode as cm

async with cm.get_context() as ctx:
    print("Editing cells directly with top-level async with...")
    for cid in ["PKri", "SFPL", "ZHCJ", "qnkX", "Vxnm", "ulZA"]:
        assert cid in ctx.cells, f"{cid} not in cells"
    print("All cell IDs present!")
