from __future__ import annotations

import argparse
import zipfile
from collections.abc import Sequence
from pathlib import Path


def zip_dir(src: Path, dst: Path, level: int = 1) -> list[str]:
    if not src.is_dir():
        raise FileNotFoundError(f"thư mục không tồn tại: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    names: list[str] = []
    with zipfile.ZipFile(dst, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=level) as archive:
        for path in sorted(p for p in src.rglob("*") if p.is_file()):
            name = path.relative_to(src).as_posix()
            archive.write(path, name)
            names.append(name)
            print(f"zipped {name} ({path.stat().st_size / 1e9:.2f} GB)", flush=True)
    return names


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Zip một thư mục (ZIP64, nén deflate).")
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--dst", type=Path, required=True)
    parser.add_argument("--level", type=int, default=1, choices=range(0, 10))
    args = parser.parse_args(argv)
    names = zip_dir(args.src, args.dst, args.level)
    size = args.dst.stat().st_size / 1e9
    print(f"done: {len(names)} tệp -> {args.dst} ({size:.2f} GB)", flush=True)


if __name__ == "__main__":
    main()
