"""Download pinned public pretrained assets and smoke-test all nine adapters."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "benchmark_results/ce9_clean_mixture_20261102"
ASSETS = OUT / "pretrained"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url, filename):
    import requests
    target = ASSETS / filename
    if target.exists():
        return str(target)
    last = None
    for attempt in range(4):
        try:
            with requests.get(url, stream=True, timeout=(30, 90)) as response:
                response.raise_for_status()
                temp = target.with_suffix(target.suffix+".partial")
                with temp.open("wb") as stream:
                    for chunk in response.iter_content(1024*1024):
                        stream.write(chunk)
                size = int(response.headers.get("Content-Length", temp.stat().st_size))
                if temp.stat().st_size != size or size < 1000000:
                    raise ValueError(f"Unexpected weight size: {filename}")
                temp.replace(target)
            return str(target)
        except Exception as error:
            last = error
            time.sleep(min(2**attempt, 8))
    raise RuntimeError(f"Download failed: {url}") from last


def main():
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ASSETS.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import hf_hub_download
    from noise_pipeline.ce9_models import HF_REVISIONS, MODEL_NAMES, build_model, beats_checkpoint
    tasks = []
    with ThreadPoolExecutor(max_workers=5) as pool:
        for name, revision in HF_REVISIONS.items():
            for file in ("config.json", "model.safetensors"):
                tasks.append(pool.submit(hf_hub_download, "mispeech/"+name.replace("_", "-"), file,
                                         revision=revision, local_dir=ASSETS/name))
        tasks.append(pool.submit(download, "https://zenodo.org/records/3987831/files/Cnn10_mAP%3D0.380.pth?download=1", "Cnn10_mAP=0.380.pth"))
        tasks.append(pool.submit(download, "https://github.com/fschmid56/EfficientAT/releases/download/v0.0.1/mn10_as_mAP_471.pt", "mn10_as_mAP_471.pt"))
        for task in as_completed(tasks):
            print("Asset ready:", task.result(), flush=True)
    import torch
    torch.set_num_threads(4)
    torch.manual_seed(2026)
    device = torch.device("cuda")
    checks = {}
    for name in MODEL_NAMES:
        print("SMOKE", name, flush=True)
        model = build_model(name, ASSETS).to(device)
        model.train()
        batch = {"beats12": 8, "ced_small": 16, "vit_small": 16, "mobilevit_s": 16}.get(name, 32)
        x = torch.randn(batch, 64000, device=device)*.05
        y = torch.arange(batch, device=device) % 36
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits.float(), y)
        assert logits.shape == (batch, 36) and torch.isfinite(loss)
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        model.eval()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            single = model(x[:1])
        assert single.shape == (1, 36) and torch.isfinite(single).all()
        checks[name] = {"status": "passed", "training_microbatch": batch, "loss": loss.item(), "parameters": sum(p.numel() for p in model.parameters()),
                        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad), "receipt": model.receipt}
        (OUT/"model_smoke_checks.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        del model, logits, single, loss, gradients
        torch.cuda.empty_cache()
    sources = {}
    for name in ("ced_ce9", "efficientat_ce9", "panns_ce9", "beats"):
        path = ROOT/"third_party"/name
        revision = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True)
        sources[name] = revision.stdout.strip() or "existing local source; see file hashes"
    receipt = {"source_revisions": sources, "hf_revisions": HF_REVISIONS,
               "asset_hashes": {str(p.relative_to(ASSETS)): sha(p) for p in ASSETS.rglob('*') if p.is_file() and '.cache' not in p.parts},
               "beats_checkpoint_sha256": sha(beats_checkpoint()),
               "adapter_sha256": sha(ROOT/"noise_pipeline/ce9_models.py")}
    (OUT/"pretrained_provenance.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print("All nine model adapters passed real forward/backward and batch-one evaluation.", flush=True)


if __name__ == "__main__":
    main()
