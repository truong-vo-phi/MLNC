"""Biên dịch các sơ đồ KD-MoE (cần XeLaTeX + pdftocairo trong PATH).

Chạy:  python build.py            (từ bất kỳ thư mục nào)
Sinh:  fig_branches.png, fig_student.png, fig_hybrid.png, fig_freqpool.png, fig_moe.png (+ .pdf)
fig_blocks.tex có 3 tikzpicture, mỗi cái thành một trang PDF rồi một PNG riêng.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
DPI = "200"
SINGLE = ("fig_branches", "fig_student")
PAGES = {"fig_blocks": ("fig_hybrid", "fig_freqpool", "fig_moe")}


def xelatex(name: str) -> None:
    subprocess.run(
        ["xelatex", "-interaction=nonstopmode", "-halt-on-error", f"{name}.tex"],
        cwd=HERE,
        check=True,
        stdout=subprocess.DEVNULL,
    )


def png(pdf: str, out: str, page: int | None = None) -> None:
    cmd = ["pdftocairo", "-png", "-r", DPI, "-singlefile"]
    if page is not None:
        cmd += ["-f", str(page), "-l", str(page)]
    subprocess.run([*cmd, f"{pdf}.pdf", out], cwd=HERE, check=True)


def main() -> None:
    for name in SINGLE:
        xelatex(name)
        png(name, name)
    for name, outputs in PAGES.items():
        xelatex(name)
        for page, out in enumerate(outputs, start=1):
            png(name, out, page)
    for pattern in ("*.aux", "*.log"):
        for path in HERE.glob(pattern):
            path.unlink()


if __name__ == "__main__":
    main()
