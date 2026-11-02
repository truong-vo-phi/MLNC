"""Reproducible full-data Teacher caching and 15-epoch team CED-MoE KD run."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "benchmark_results/team_kd_moe_full_20261102"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    log = (OUT / "run.log").open("a", encoding="utf-8", buffering=1)

    class Tee:
        def write(self, value):
            sys.__stdout__.write(value)
            log.write(value)

        def flush(self):
            sys.__stdout__.flush()
            log.flush()

    sys.stdout = Tee()
    paths = json.loads((ROOT / "benchmark_results/team_kd_moe_research_20261102/TEAM_DATA_PATHS.json").read_text())
    runtime = OUT / "runtime"
    if not runtime.exists():
        shutil.copytree(ROOT / "runtime/kd_moe_ced", runtime,
                        ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", "output"))
        trainer_path = runtime / "src/kd_trainer.py"
        text = trainer_path.read_text(encoding="utf-8")
        text = text.replace("            loss.backward()", "            if not torch.isfinite(loss):\n                raise FloatingPointError('Non-finite training loss')\n            loss.backward()")
        text = text.replace("torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)", "torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0, error_if_nonfinite=True)")
        text = text.replace("            total_loss += float(loss.item())", '            if batch_idx % 50 == 0:\n                print(f"  epoch={epoch+1} step={batch_idx+1}/{num_batches} loss={loss.item():.5f}", flush=True)\n            total_loss += float(loss.item())')
        trainer_path.write_text(text, encoding="utf-8")
    sys.path.insert(0, str(runtime))
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    import numpy as np
    import torch
    import torch.nn.functional as F
    from huggingface_hub import hf_hub_download
    from src.dataset import Mixture36Dataset, NoiseCacheDataset, KDNoiseCacheDataset, build_dataloader
    from src.kd_trainer import KDTrainer
    from src.models.beat import load_fold_teacher
    from src.models.split_moe import CEDSingleMoESplitFFN
    from src.models.vit_grep import load_ced_tiny
    from src.pipeline import ChampionPipeline, tta_forward

    torch.set_num_threads(4)
    random.seed(2026)
    np.random.seed(2026)
    torch.manual_seed(2026)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Full scheduled experiment requires CUDA")
    config = {"seed": 2026, "epochs": 15, "batch_size": 64, "teacher_batch_size": 8,
              "lr_backbone": 4e-5, "lr_experts": 1.2e-4, "lr_router": 2e-4,
              "weight_decay": .05, "warmup_epochs": 2, "clip_grad_norm": 1.,
              "mixup_alpha": .4, "mixup_prob": .8, "label_smoothing": .05,
              "lambda_kd": .65, "lambda_route": .20, "kd_temperature": 2.,
              "amp": "bfloat16", "selection": "highest eval top1 without TTA, same as team CLI",
              "teacher_input": "FRCRN cache without additional external RMS normalization, matching new CED online helper",
              "teacher_tta": "shift 0 only", "source_commit": "697064dfd638f0b934f1a9ee84cac2cf0f32c063",
              "test_used": False, "paths": paths, "torch": torch.__version__}
    write_json(OUT / "config.json", config)
    status = {"stage": "verify_inputs", "status": "running"}

    def update(**values):
        status.update(values)
        write_json(OUT / "status.json", status)

    update()
    bases = {s: Mixture36Dataset(paths["raw_mixture_root"], s) for s in ("train", "eval")}
    assert len(bases["train"]) == 30240 and len(bases["eval"]) == 3064
    ids = {s: [base[i]["sample_id"] for i in range(len(base))] for s, base in bases.items()}
    assert not set(ids["train"]) & set(ids["eval"])
    for split, base in bases.items():
        assert len(set(ids[split])) == len(base)
        for i in range(len(base)):
            y = base[i]["target"]
            assert len(y) == 36 and int(y.sum()) == 1 and bool(((y == 0) | (y == 1)).all())
    cache_dir = OUT / "teacher_cache"
    cache_dir.mkdir(exist_ok=True)
    provenance_path = cache_dir / "provenance.json"
    if provenance_path.exists():
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        for key in ("train_manifest", "eval_manifest", "teacher_fold0", "teacher_ema", "train_cache", "eval_cache"):
            if provenance["inputs"][key]["sha256"] != digest(paths[key]):
                raise ValueError(f"Input changed since caching: {key}")
    else:
        print("Hashing dataset manifests, FRCRN caches and Teacher checkpoints", flush=True)
        provenance = {"teacher_input": config["teacher_input"], "teacher_tta": "none", "amp": "bfloat16",
                      "inputs": {key: {"path": paths[key], "sha256": digest(paths[key])}
                                 for key in ("train_manifest", "eval_manifest", "teacher_fold0", "teacher_ema", "train_cache", "eval_cache")},
                      "sample_ids": ids, "class_names": bases["train"].class_names}
        write_json(provenance_path, provenance)
    if provenance["sample_ids"] != ids:
        raise ValueError("Manifest order mismatch")

    for teacher_key, short in (("teacher_fold0", "t0"), ("teacher_ema", "tema")):
        teacher = None
        for split in ("train", "eval"):
            prefix = "train" if split == "train" else "val"
            filename = cache_dir / f"{prefix}_{short}_logits.npy"
            cursor_file = filename.with_suffix(".progress.json")
            expected_shape = (len(bases[split]), 36)
            progress = json.loads(cursor_file.read_text()) if cursor_file.exists() else {"completed": 0}
            completed = int(progress["completed"])
            if completed and not filename.exists():
                raise ValueError("Progress exists without logits file")
            if filename.exists():
                array = np.lib.format.open_memmap(filename, mode="r+")
                if array.shape != expected_shape or array.dtype != np.float32:
                    raise ValueError(f"Invalid cached logits: {filename}")
            else:
                array = np.lib.format.open_memmap(filename, mode="w+", dtype="float32", shape=expected_shape)
            if completed < len(bases[split]):
                if teacher is None:
                    print("Loading", teacher_key, flush=True)
                    teacher, metadata = load_fold_teacher(paths[teacher_key], runtime / "config/beat.yaml", device)
                    write_json(cache_dir / f"{short}_metadata.json", metadata)
                dataset = NoiseCacheDataset(bases[split], paths["train_cache" if split == "train" else "eval_cache"])
                subset = torch.utils.data.Subset(dataset, range(completed, len(dataset)))
                loader = build_dataloader(subset, batch_size=8, num_workers=0)
                start = time.monotonic()
                with torch.inference_mode():
                    for batch_idx, batch in enumerate(loader):
                        wave = batch["waveform"].to(device)
                        logits, _ = tta_forward(teacher, wave, device)
                        values = logits.cpu().numpy()
                        if not np.isfinite(values).all():
                            raise FloatingPointError("Non-finite Teacher logits")
                        end = completed + len(values)
                        array[completed:end] = values
                        array.flush()
                        completed = end
                        write_json(cursor_file, {"completed": completed, "total": len(dataset)})
                        if batch_idx % 100 == 0 or completed == len(dataset):
                            print(f"{short} {split}: {completed}/{len(dataset)} ({time.monotonic()-start:.1f}s)", flush=True)
                            update(stage="teacher_logits", teacher=short, split=split, completed=completed, total=len(dataset))
            if not np.isfinite(array).all():
                raise FloatingPointError("Invalid completed cache")
            write_json(filename.with_suffix(".verified.json"), {"shape": list(array.shape), "sha256": digest(filename), "complete": True})
            del array
        del teacher
        torch.cuda.empty_cache()
    update(stage="train", teacher_logits_complete=True)
    print("All four full Teacher logits caches verified. Initializing Student.", flush=True)

    # Reseed after Teacher construction so resume/initialization are independent of caching.
    random.seed(2026)
    np.random.seed(2026)
    torch.manual_seed(2026)
    pretrained_path = hf_hub_download("mispeech/ced-tiny", "model.safetensors")
    import src.models.split_moe as module
    original = module.load_ced_tiny

    def checked_load(model):
        receipt = load_ced_tiny(model, path=pretrained_path)
        if receipt["loaded_tensors"] == 0:
            raise RuntimeError("Failed pretrained initialization")
        write_json(OUT / "ced_initialization.json", {**receipt, "path": pretrained_path, "sha256": digest(pretrained_path)})
        return receipt

    module.load_ced_tiny = checked_load
    model = CEDSingleMoESplitFFN(num_classes=36, load_pretrained=True)
    module.load_ced_tiny = original
    datasets = {s: KDNoiseCacheDataset(bases[s], paths["train_cache" if s == "train" else "eval_cache"],
                                     cache_dir / f"{'train' if s == 'train' else 'val'}_t0_logits.npy",
                                     cache_dir / f"{'train' if s == 'train' else 'val'}_tema_logits.npy") for s in bases}
    train_loader = build_dataloader(datasets["train"], batch_size=64, shuffle=True)
    val_loader = build_dataloader(datasets["eval"], batch_size=64)
    checkpoints = OUT / "checkpoints"
    trainer = KDTrainer(model, train_loader, val_loader, device, epochs=15, out_dir=checkpoints)
    history_file = OUT / "history.json"
    history = []
    start_epoch = 0
    last = checkpoints / "last.pt"
    if last.exists():
        state = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        trainer.optimizer.load_state_dict(state["optimizer_state_dict"])
        trainer.scheduler.load_state_dict(state["scheduler_state_dict"])
        trainer.best_top1_acc = state["best_top1_acc"]
        trainer.best_results = state["best_results"]
        start_epoch = state["epoch"]
        history = state["history"]
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    for epoch in range(start_epoch, 15):
        update(stage="train", epoch=epoch+1, epochs=15)
        started = time.monotonic()
        train_result = trainer.train_epoch(epoch)
        val_result = trainer.evaluate()
        if not all(math.isfinite(v) for v in train_result.values()):
            raise FloatingPointError("Invalid epoch loss")
        top1 = val_result["overall"]["top1"]
        is_best = top1 > trainer.best_top1_acc or not (checkpoints / "best.pt").exists()
        if is_best:
            trainer.best_top1_acc = top1
            trainer.best_results = val_result["overall"]
        history.append({"epoch": epoch+1, "train": train_result, "val": val_result,
                        "seconds": time.monotonic()-started, "is_best": is_best})
        payload = {"model_state_dict": model.state_dict(), "optimizer_state_dict": trainer.optimizer.state_dict(),
                   "scheduler_state_dict": trainer.scheduler.state_dict(), "epoch": epoch+1,
                   "best_top1_acc": trainer.best_top1_acc, "best_results": trainer.best_results,
                   "history": history, "config": config, "python_rng": random.getstate(),
                   "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
                   "cuda_rng": torch.cuda.get_rng_state_all()}
        temporary = checkpoints / "last.tmp.pt"
        torch.save(payload, temporary)
        temporary.replace(last)
        if is_best:
            shutil.copyfile(last, checkpoints / "best.pt")
        write_json(history_file, history)
        update(epoch=epoch+1, best_val_accuracy=trainer.best_top1_acc, latest_val=val_result["overall"])
        print(f"Epoch {epoch+1}/15 loss={train_result['loss']:.5f} val_acc={top1:.4%} val_mAP={val_result['overall']['map']:.4%} best={is_best} seconds={history[-1]['seconds']:.1f}", flush=True)

    update(stage="final_validation")
    state = torch.load(checkpoints / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    pipe = ChampionPipeline(model, device=device)
    collected = {"student": [], "student_tta": [], "ensemble": []}
    targets, snrs, alphas = [], [], []
    with torch.inference_mode():
        for batch in val_loader:
            wave = batch["waveform"].to(device)
            logits, _ = tta_forward(model, wave, device)
            collected["student"].append(F.softmax(logits, -1).cpu().numpy())
            p_student, alpha = pipe.predict_student_tta(wave)
            pt0 = F.softmax(batch["t0_logits"] / 1.2, -1).numpy()
            ptema = F.softmax(batch["tema_logits"] / 1.2, -1).numpy()
            ensemble = .25*p_student + ((1-alpha)*.45+alpha*.30)*pt0 + ((1-alpha)*.30+alpha*.45)*ptema
            collected["student_tta"].append(p_student)
            collected["ensemble"].append(ensemble)
            targets.extend(batch["target_idx"].tolist())
            snrs.extend(batch["snr_db"].tolist())
            alphas.append(alpha)
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, average_precision_score, roc_auc_score
    y = np.array(targets)
    snrs = np.array(snrs)

    def metrics(probs, labels):
        prediction = probs.argmax(1)
        p, r, f, _ = precision_recall_fscore_support(labels, prediction, labels=np.arange(36), average="macro", zero_division=0)
        _, _, micro, _ = precision_recall_fscore_support(labels, prediction, average="micro", zero_division=0)
        onehot = np.eye(36)[labels]
        ap = [average_precision_score(onehot[:, i], probs[:, i]) for i in range(36) if onehot[:, i].sum()]
        auc = [roc_auc_score(onehot[:, i], probs[:, i]) for i in range(36) if 0 < onehot[:, i].sum() < len(labels)]
        return {"n": len(labels), "accuracy": accuracy_score(labels, prediction), "precision_macro": p,
                "recall_macro": r, "macro_f1": f, "micro_f1": micro, "mAP": float(np.mean(ap)),
                "balanced_accuracy": r, "macro_auc": float(np.mean(auc)) if auc else None}

    final = {"selection_epoch": state["epoch"], "checkpoint_sha256": digest(checkpoints / "best.pt"), "split": "eval", "modes": {}}
    for mode, values in collected.items():
        probs = np.concatenate(values)
        final["modes"][mode] = {"overall": metrics(probs, y), "per_snr": {str(s): metrics(probs[snrs == s], y[snrs == s]) for s in (-5, 0, 5, 10, 15, 20)}}
        np.savez(OUT / f"predictions_{mode}.npz", probabilities=probs, targets=y, snrs=snrs, alphas=np.concatenate(alphas))
        p, r, f, support = precision_recall_fscore_support(y, probs.argmax(1), labels=np.arange(36), zero_division=0)
        with (OUT / f"per_class_{mode}.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.writer(stream)
            writer.writerow(["class", "support", "precision", "recall", "f1", "AP"])
            for i, name in enumerate(bases["eval"].class_names):
                writer.writerow([name, int(support[i]), p[i], r[i], f[i], average_precision_score(y == i, probs[:, i])])
    write_json(OUT / "final_metrics.json", final)
    lines = ["# Kết quả MOE-KD-CED theo lịch nhóm", "", "Dataset HF v2: 30.240 train, 3.064 eval; không dùng test.",
             f"Train 15 epoch; chọn best.pt bằng eval accuracy không TTA. Epoch được chọn: {state['epoch']}.", "",
             "Teacher logits single-pass trên FRCRN cache theo nhánh online CED gốc, không thêm RMS bên ngoài Teacher.",
             "Student có RMS normalization nội bộ. Loss/optimizer/scheduler giữ mặc định train CLI của nhóm, batch 64.", "",
             "| Chế độ | Accuracy | Precision | Recall | Macro-F1 | Micro-F1 | mAP | Balanced Acc | Macro-AUC |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    keys = ("accuracy", "precision_macro", "recall_macro", "macro_f1", "micro_f1", "mAP", "balanced_accuracy", "macro_auc")
    for mode, res in final["modes"].items():
        lines.append("| " + mode + " | " + " | ".join(f"{100*res['overall'][k]:.2f}%" for k in keys) + " |")
    lines.extend(["", "Student TTA: trung bình logits ở shift 0/±800, temperature 1,4. Ensemble thêm hai Teacher temperature 1,2; tổng trọng số Teacher 75%.",
                  "Các số có TTA/ensemble dùng cùng checkpoint chọn từ Student không TTA; không chọn lại checkpoint bằng test.", "",
                  "## Theo SNR", "", "| SNR | Số mẫu | Student Acc | Student TTA Acc | Ensemble Acc |", "|---|---:|---:|---:|---:|"])
    for s in (-5, 0, 5, 10, 15, 20):
        vals = [final["modes"][m]["per_snr"][str(s)] for m in collected]
        lines.append(f"| {s} dB | {vals[0]['n']} | " + " | ".join(f"{v['accuracy']*100:.2f}%" for v in vals) + " |")
    lines.extend(["", "Chi tiết 8 metric theo từng SNR trong final_metrics.json; từng lớp trong per_class_*.csv. Config, hash nguồn/cache, ID và history được lưu cùng thư mục."])
    (OUT / "BAO_CAO_KET_QUA_VI.md").write_text("\n".join(lines), encoding="utf-8")
    update(stage="complete", status="complete", best_epoch=state["epoch"])
    print(json.dumps(final, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise
