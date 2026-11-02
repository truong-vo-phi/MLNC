"""Spot-check the downloaded cache against FRCRN on the paired raw mixtures."""
from pathlib import Path
import csv
import json
import sys

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "benchmark_results/team_kd_moe_research_20261102"
REPO = ROOT.parents[1]  # MLNC checkout; FRCRN sources match the verified snapshot.
sys.path.insert(0, str(REPO))
from src.model.frcrn import FRCRNSeparator


def compare(a, b):
    a, b = a.astype("float64"), b.astype("float64")
    gain = float(np.dot(a,b)/max(np.dot(b,b),1e-30))
    return dict(cosine=float(np.dot(a,b)/max(np.linalg.norm(a)*np.linalg.norm(b),1e-30)),
                relative_error_after_gain=float(np.linalg.norm(a-gain*b)/max(np.linalg.norm(a),1e-30)),
                fitted_gain=gain)


def main():
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT.mkdir(parents=True, exist_ok=True)
    paths = json.loads((OUT / "TEAM_DATA_PATHS.json").read_text())
    model = FRCRNSeparator.from_yaml(REPO / "config/frcrn.yaml")
    report = model.load_pretrained(Path(paths["frcrn_checkpoint"]))
    assert report.is_clean
    model.to(device).freeze()
    data = Path(paths["frcrn_cache_root"])
    raw_root = Path(paths["raw_mixture_root"])
    checks = []
    for split, filename in (("train","train.npy"),("eval","validation.npy")):
        with (raw_root / split / "manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        cache = np.load(data / filename, mmap_mode="r")
        indices = []
        for snr in (-5,0,5,10,15,20):
            candidates = [i for i,row in enumerate(rows) if int(float(row["target_snr_db"])) == snr]
            indices.append(candidates[len(candidates)//2])
        indices = sorted(set([0,len(rows)-1,*indices]))
        for offset in range(0,len(indices),2):
            chosen = indices[offset:offset+2]
            waves = [sf.read(raw_root / split / "mixture" / (rows[i]["sample_id"]+".wav"),
                             dtype="float32")[0] for i in chosen]
            raw = torch.from_numpy(np.stack(waves)).to(device)
            with torch.no_grad(), torch.autocast(device.type, enabled=False):
                noise = model(raw).cpu().numpy()
            for local, index in enumerate(chosen):
                saved = np.asarray(cache[index])
                check = dict(split=split, index=index, sample_id=rows[index]["sample_id"],
                             snr_db=rows[index]["target_snr_db"], noise=compare(saved,noise[local]),
                             speech=compare(saved,waves[local]-noise[local]))
                checks.append(check)
                print(f"cache check {split} {check['sample_id']} noise_cos={check['noise']['cosine']:.6f} speech_cos={check['speech']['cosine']:.6f}",flush=True)
    result = dict(checkpoint="best_e037_20261020T203958Z.pt", frcrn_config=model.config.to_dict(),
                  weights_loaded_cleanly=True, sample_checks=checks,
                  noise_matches_all_spot_checks=all(x["noise"]["cosine"] > 0.999 for x in checks),
                  note=f"{len(checks)} spot checks only; cache has no ID sidecar, so this is not an exhaustive ordering proof")
    (OUT / "frcrn_cache_spot_checks.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k != "sample_checks"},indent=2),flush=True)


if __name__ == "__main__":
    main()
