"""Run Section 13 CED-SingleMoE-SplitFFN ablations on the Molab waveform cache."""
from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.ced_moe_split import CEDSingleMoESplitFFN, MoETrainer


class CacheManifestDataset(Dataset):
    """Pair a row-ordered waveform .npy cache with its dataset manifest."""
    def __init__(self, array_path: Path, manifest_path: Path, num_classes: int):
        self.array = np.load(array_path, mmap_mode="r")
        with manifest_path.open(encoding="utf-8-sig", newline="") as f:
            self.rows = list(csv.DictReader(f))
        if len(self.array) != len(self.rows):
            raise ValueError(f"Cache/manifest row mismatch: {array_path}={len(self.array)}, {manifest_path}={len(self.rows)}")
        self.num_classes = num_classes

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        target = torch.tensor(json.loads(row["multi_hot_36"]), dtype=torch.float32)
        if target.numel() != self.num_classes:
            raise ValueError(f"Expected {self.num_classes} labels, got {target.numel()} at row {i}")
        return {
            "waveform": torch.from_numpy(np.array(self.array[i], copy=True)),
            "target": target,
            "snr_db": torch.tensor(float(row.get("target_snr_db", "0")), dtype=torch.float32),
            "sample_id": row.get("sample_id", str(i)),
        }


def setup_logger(path: Path) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("moe_ablations")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


def load_teacher(path: Path, device: torch.device) -> torch.nn.Module:
    if not path.is_file():
        raise FileNotFoundError(f"BEATs teacher checkpoint missing: {path}; run prepare_moe_ablations.py first")
    import sys as _sys
    third_party = ROOT / "ce_v2" / "third_party" / "beats"
    if str(third_party) not in _sys.path:
        _sys.path.insert(0, str(third_party))
    from BEATs import BEATs, BEATsConfig

    config = {
        "input_patch_size": 16, "embed_dim": 512, "encoder_embed_dim": 768,
        "encoder_layers": 12, "encoder_ffn_embed_dim": 3072,
        "encoder_attention_heads": 12, "deep_norm": True, "conv_pos": 128,
        "conv_pos_groups": 16, "relative_position_embedding": True,
        "num_buckets": 320, "max_distance": 800, "gru_rel_pos": True,
        "dropout": 0.0, "attention_dropout": 0.0, "activation_dropout": 0.0,
        "dropout_input": 0.0, "finetuned_model": False,
    }

    class Teacher(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Module()
            self.encoder.model = BEATs(BEATsConfig(config))
            self.head = torch.nn.Linear(768, 36)

        def forward(self, wave):
            hidden, _ = self.encoder.model.extract_features(wave.float())
            return self.head(hidden.mean(dim=1))

    teacher = Teacher()
    blob = torch.load(path, map_location="cpu", weights_only=False)
    state = blob.get("state_dict", blob)
    cleaned = {key: value for key, value in state.items() if key in teacher.state_dict()}
    own = teacher.state_dict()
    matched = {k: v for k, v in cleaned.items() if k in own and own[k].shape == v.shape}
    if len(matched) < 200:
        raise RuntimeError(f"Only {len(matched)} BEATs tensors match the teacher model; refusing random teacher.")
    teacher.load_state_dict(matched, strict=False)
    teacher.to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    print(f"BEATs teacher loaded: {len(matched)} tensors from {path}", flush=True)
    return teacher


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/ablations.yaml")
    parser.add_argument("--case", nargs="+", help="Run one or more selected case IDs.")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--prepare-only", action="store_true", help="Validate files and print cases without training.")
    args = parser.parse_args()

    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    defaults = cfg["defaults"]
    data = Path(defaults["dataset_root"])
    train_path = Path(defaults.get("train_array", data / "train.npy"))
    val_path = Path(defaults.get("validation_array", data / "validation.npy"))
    manifest_root = Path(defaults["manifest_root"])
    train_manifest = manifest_root / "train" / "manifest.csv"
    val_manifest = manifest_root / "eval" / "manifest.csv"
    for required in (train_path, val_path, train_manifest, val_manifest):
        if not required.is_file():
            raise FileNotFoundError(f"Required ablation input missing: {required}")

    train_ds = CacheManifestDataset(train_path, train_manifest, defaults["num_classes"])
    val_ds = CacheManifestDataset(val_path, val_manifest, defaults["num_classes"])
    if (len(train_ds), len(val_ds)) != (30240, 3064):
        raise ValueError(f"Unexpected split counts: train={len(train_ds)}, validation={len(val_ds)}")
    print(f"DATA READY train={len(train_ds)} validation={len(val_ds)} waveform_samples={train_ds.array.shape[1]}", flush=True)

    cases = cfg["cases"]
    selected = [c for c in cases if not args.case or c["id"] in args.case]
    unknown = sorted(set(args.case or ()) - {c["id"] for c in cases})
    if unknown:
        raise ValueError(f"Unknown case IDs: {unknown}")
    for c in selected:
        print(f"CASE READY {c['id']}: {c['question']}", flush=True)
    if args.prepare_only:
        return

    weights = ROOT / "pretrained" / "model.safetensors"
    if not weights.is_file():
        raise FileNotFoundError(f"Official CED-Tiny weights are required before training: {weights}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_root = ROOT / "logs" / "ablations"
    for case in selected:
        seed = int(defaults["seed"])
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        case_dir = out_root / case["id"]
        logger = setup_logger(case_dir / "run.log")
        logger.info("START CASE %s on %s", case["id"], device)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size or defaults["batch_size"], shuffle=True, num_workers=4, pin_memory=torch.cuda.is_available())
        val_loader = DataLoader(val_ds, batch_size=args.batch_size or defaults["batch_size"], shuffle=False, num_workers=4, pin_memory=torch.cuda.is_available())
        model = CEDSingleMoESplitFFN(
            num_classes=defaults["num_classes"],
            split_blocks=tuple(case["split_blocks"]),
            router_mode=case["router"],
            use_shared_expert=case["shared_expert"],
            single_head=case["heads"] == "single",
            constant_alpha=float(case.get("alpha", 0.5)),
            load_pretrained=True,
            pretrained_path=str(weights),
        )
        if model.pretrained_loaded_tensors == 0:
            raise RuntimeError(f"No CED-Tiny tensors loaded from {weights}; refusing random-init comparison.")
        teacher_cfg = case.get("teacher")
        teacher_path = Path(teacher_cfg) if teacher_cfg else ROOT / "pretrained/beats_teacher_fold0_weights.pt"
        if not teacher_path.is_absolute():
            teacher_path = ROOT / teacher_path
        teacher = load_teacher(teacher_path, device) if case.get("kd", False) else None
        trainer = MoETrainer(
            student_model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            teacher_model=teacher,
            device=device,
            epochs=args.epochs or defaults["epochs"],
            lambda_router=float(case["router_loss_weight"]),
            lambda_kd=float(case.get("lambda_kd", 0.5 if case.get("kd", False) else 0.0)),
            checkpoint_dir=case_dir / "checkpoints",
            logger=logger,
        )
        history = trainer.train()
        best_record = next((item for item in history.get("history", []) if item.get("epoch") == trainer.best_epoch), None)
        if best_record is not None:
            metrics = best_record.get("val_metrics", {})
            breakdown = best_record.get("breakdown", {})
        else:
            metrics, breakdown = trainer.evaluate()
        result = {"case": case, "best_epoch": trainer.best_epoch, "best_validation_accuracy": trainer.best_metric, "metrics": metrics, "snr_breakdown": breakdown, "history": history}
        (case_dir / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        logger.info("COMPLETE %s best_epoch=%d val_top1=%.4f", case["id"], trainer.best_epoch, trainer.best_metric)


if __name__ == "__main__":
    main()
