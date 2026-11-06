import io, zipfile, base64, pathlib

root = pathlib.Path("imp/review-3")
archive = io.BytesIO()
with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
    for p in root.rglob("*"):
        if p.is_file() and not any(x in p.parts for x in ["__pycache__", "checkpoints", "logs", "benchmark_results", ".git", "pretrained", "tests"]):
            rel = p.relative_to(root).as_posix()
            z.write(p, rel)

payload = base64.b64encode(archive.getvalue()).decode("ascii")

deploy_code = f'''import base64, io, zipfile, pathlib

root = pathlib.Path("/marimo/review-3")
root.mkdir(parents=True, exist_ok=True)

archive = zipfile.ZipFile(io.BytesIO(base64.b64decode("{payload}")))
archive.extractall(root)

print(f"Extracted {{len(archive.infolist())}} files to {{root}}")
'''

pathlib.Path("scratch/deploy_review3_remote.py").write_text(deploy_code, encoding="ascii")
print("Packaged review-3 successfully. Wrote scratch/deploy_review3_remote.py")
