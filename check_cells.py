import marimo._code_mode as cm
ctx = cm.get_context()
for i, cell in enumerate(ctx.cells):
    q_count = cell.code.count("?")
    has_bad = any(w in cell.code for w in ["?ang", "th?nh", "c?a", "??", "h?p", "m?u", "hu?n", "ki?m"])
    non_ascii = len([c for c in cell.code if ord(c) > 127])
    print(f"Cell {i:2d} (id={cell.id}): bad_pattern={has_bad}, ? count={q_count}, non_ascii={non_ascii}")
