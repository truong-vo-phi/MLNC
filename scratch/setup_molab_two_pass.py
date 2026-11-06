from pathlib import Path
import shutil
import datetime

root = Path('/marimo/review-3/ce_v2')
train = root / 'train_ce9_clean.py'
models = root / 'noise_pipeline/ce9_models.py'
prepare = root / 'prepare_ce9.py'
dest = root / 'train_two_pass.py'
if dest.exists():
    raise FileExistsError(dest)
backup = root / ('backup_two_pass_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
backup.mkdir()
for p in (models, prepare):
    shutil.copy2(p, backup / p.name)

model_text = models.read_text()
old = '"vit_small", "mobilevit_s", "mobilenet_v3_large")'
assert model_text.count(old) == 1
model_text = model_text.replace(old, '"vit_small", "mobilevit_s", "mobilenet_v3_large", "beats12")')

prep_text = prepare.read_text()
old = '    from noise_pipeline.ce9_models import HF_REVISIONS, MODEL_NAMES, build_model\n'
assert prep_text.count(old) == 1
prep_text = prep_text.replace(old, old + '''    from huggingface_hub import hf_hub_download as _download_beats
    beats = ROOT / "checkpoints/pretrained/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt"
    beats.parent.mkdir(parents=True, exist_ok=True)
    if not beats.is_file():
        import shutil
        downloaded = _download_beats(repo_id="WeiChihChen/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2", filename=beats.name, revision="172e4071a2912392d9151c70750b9a12f55fc54e")
        shutil.copyfile(downloaded, beats)
    if sha(beats) != "e5815275a04b6885e7b8af63d120b29bffae2cd2225cf4915e1ec6d819d3022c":
        raise ValueError("BEATs pretrained checkpoint SHA256 mismatch")
''')
prep_text = prep_text.replace('"adapter_sha256": sha(ROOT/"noise_pipeline/ce9_models.py")}', '"beats_sha256": sha(beats), "beats_source": "WeiChihChen/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2@172e4071a2912392d9151c70750b9a12f55fc54e", "adapter_sha256": sha(ROOT/"noise_pipeline/ce9_models.py")}')

text = train.read_text()
old = 'OUT = ROOT / "benchmark_results/ce9_clean_mixture_20261102"'
assert text.count(old) == 1
text = text.replace(old, '''SHARED = ROOT / "benchmark_results/ce9_clean_mixture_20261102"
MODE = os.environ.get("CE_INPUT_MODE", "")
if MODE not in ("mixture", "frcrn"):
    raise ValueError("Set CE_INPUT_MODE to mixture or frcrn")
OUT = ROOT / "benchmark_results/two_pass_ce9" / MODE''')
old = 'def prepare_raw_cache():\n'
assert text.count(old) == 1
text = text.replace(old, '''def prepare_raw_cache():
    if MODE == "frcrn":
        import numpy as np
        root, labels, meta = read_metadata()
        directory = OUT / "raw_mixture_cache"
        directory.mkdir(parents=True, exist_ok=True)
        paths = {"train": Path("/dataset/cache/train.npy"), "eval": Path("/dataset/cache/validation.npy")}
        provenance = {"input": "FRCRN cache; no external RMS normalization", "alignment": "assumed manifest order; cache has no independent sample IDs", "manifest_sha256": {s: sha(root/s/"manifest.csv") for s in meta}, "labels_sha256": sha(root/"labels.txt"), "cache_sha256": {s: sha(p) for s,p in paths.items()}}
        previous = directory / "provenance.json"
        if previous.exists() and json.loads(previous.read_text()) != provenance:
            raise ValueError("FRCRN cache provenance mismatch")
        for split, path in paths.items():
            arr = np.load(path, mmap_mode="r")
            if arr.shape != (len(meta[split]["ids"]), 64000) or arr.dtype != np.float32:
                raise ValueError(f"Invalid FRCRN cache: {path} {arr.shape} {arr.dtype}")
            link = directory / f"{split}_mixture.npy"
            if link.is_symlink() and link.resolve() != path.resolve():
                raise ValueError(f"Unexpected cache symlink: {link}")
            if link.exists() and not link.is_symlink():
                raise ValueError(f"Refusing to overwrite {link}")
            if not link.exists():
                link.symlink_to(path)
        write_json(previous, provenance)
        print("FRCRN cache prepared; row alignment with manifest is assumed", flush=True)
        return
''')
text = text.replace('build_model(name, OUT/"pretrained")', 'build_model(name, SHARED/"pretrained")')
text = text.replace('"input": "raw mixture WAV cache; 16kHz mono 4s"', '"input": ("raw mixture WAV cache" if MODE == "mixture" else "FRCRN cache without external RMS normalization") + "; 16kHz mono 4s", "input_mode": MODE')
text = text.replace('checks = json.loads((OUT/"model_smoke_checks.json").read_text())', 'checks = json.loads((SHARED/"model_smoke_checks.json").read_text())')
text = text.replace('assert (OUT/"pretrained_provenance.json").is_file()', 'assert (SHARED/"pretrained_provenance.json").is_file()')
text = text.replace('checks_path = OUT / "model_smoke_checks.json"', 'checks_path = SHARED / "model_smoke_checks.json"')
text = text.replace('(OUT / "pretrained_provenance.json").is_file()', '(SHARED / "pretrained_provenance.json").is_file()')
text = text.replace('"raw_mixture_cache/provenance.json")}', '"raw_mixture_cache/provenance.json"), "input_mode": MODE}')
compile(model_text, str(models), 'exec')
compile(prep_text, str(prepare), 'exec')
compile(text, str(dest), 'exec')
models.write_text(model_text)
prepare.write_text(prep_text)
dest.write_text(text)
print('Created two-pass runner; backup:', backup)
