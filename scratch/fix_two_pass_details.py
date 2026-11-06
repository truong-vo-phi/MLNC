from pathlib import Path
import marimo._code_mode as cm

runner = Path('/marimo/review-3/ce_v2/train_two_pass.py')
source = runner.read_text(encoding='utf-8')
needle = 'def summarize():\n'
assert source.count(needle) == 1
source = source.replace(needle, 'def summarize():\n    if MODE == "frcrn":\n        return  # The source report describes raw mixture; do not mislabel FRCRN results.\n')
compile(source, str(runner), 'exec')
runner.write_text(source, encoding='utf-8')

async with cm.get_context() as ctx:
    cell = ctx.cells['comparison_table']
    code = cell.code.replace('_base = _Path(', '_completed = benchmark_report\n_base = _Path(', 1)
    ctx.edit_cell(cell.id, code=code)
print('Updated report semantics and comparison dependency')
