import marimo._code_mode as cm

async with cm.get_context() as ctx:
    cell = ctx.cells["ce_v2_guide"]
    code = cell.code.replace("/path/to/mix-dataset-v2", "/dataset/mix-dataset-v2")
    code = code.replace("Extract the clean mixture-v2 dataset, then run", "Dataset is configured at `/dataset/mix-dataset-v2`. FRCRN cache is at `/dataset/cache`; the CE mixture baseline uses raw mixture WAVs. Run")
    ctx.edit_cell(cell.id, code=code)
    ctx.run_cell(cell.id)
print("Updated dataset guide")
