"""Sequential full-network CE-only benchmark on the team's clean mixture v2."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "benchmark_results/ce9_clean_mixture_20261102"
SNRS = (-5, 0, 5, 10, 15, 20)
METRICS = ("acc", "precision", "recall", "macro_f1", "micro_f1", "mAP")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def seed_all():
    import numpy as np
    import torch
    random.seed(2026)
    np.random.seed(2026)
    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)


def read_metadata():
    import numpy as np
    paths = json.loads((ROOT/"benchmark_results/team_kd_moe_research_20261102/TEAM_DATA_PATHS.json").read_text())
    root = Path(paths["raw_mixture_root"])
    labels = (root/"labels.txt").read_text(encoding="utf-8").splitlines()
    assert len(labels) == 36 and len(set(labels)) == 36
    data = {}
    for split, n in (("train", 30240), ("eval", 3064)):
        with (root/split/"manifest.csv").open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == n
        ids, targets, snrs = [], [], []
        for row in rows:
            y = np.array(json.loads(row["multi_hot_36"]))
            assert y.shape == (36,) and y.sum() == 1 and np.isin(y, (0, 1)).all()
            index = int(y.argmax())
            if "label_names" in row:
                assert row["label_names"] == labels[index], (row["sample_id"], row["label_names"], labels[index])
            snr = int(float(row["target_snr_db"]))
            assert snr in SNRS
            ids.append(row["sample_id"])
            targets.append(index)
            snrs.append(snr)
        assert len(set(ids)) == n
        data[split] = {"ids": ids, "targets": targets, "snrs": snrs}
    assert not set(data["train"]["ids"]) & set(data["eval"]["ids"])
    return root, labels, data


def prepare_raw_cache():
    import numpy as np
    import soundfile as sf
    root, labels, meta = read_metadata()
    directory = OUT/"raw_mixture_cache"
    directory.mkdir(parents=True, exist_ok=True)
    provenance = {"input": "raw mixture WAV; no masking/FRCRN; no external RMS normalization", "sample_rate": 16000,
                  "samples_per_clip": 64000, "class_names": labels, "metadata": meta,
                  "manifest_sha256": {s: sha(root/s/"manifest.csv") for s in meta}, "labels_sha256": sha(root/"labels.txt")}
    old = directory/"provenance.json"
    if old.exists():
        assert json.loads(old.read_text(encoding="utf-8")) == provenance, "Raw-cache provenance mismatch"
    else:
        write_json(old, provenance)
    for split in meta:
        filename = directory/f"{split}_mixture.npy"
        progress_file = directory/f"{split}_progress.json"
        done = json.loads(progress_file.read_text())["completed"] if progress_file.exists() else 0
        n = len(meta[split]["ids"])
        if done:
            assert filename.exists()
        mode = "r+" if filename.exists() else "w+"
        array = np.lib.format.open_memmap(filename, mode=mode, dtype="float32", shape=(n, 64000))
        assert array.shape == (n, 64000)

        def load(index):
            path = root/split/"mixture"/(meta[split]["ids"][index]+".wav")
            wave, sr = sf.read(path, dtype="float32")
            if sr != 16000 or wave.shape != (64000,) or not np.isfinite(wave).all():
                raise ValueError(f"Invalid raw mixture WAV: {path}")
            return index, wave

        with ThreadPoolExecutor(max_workers=8) as pool:
            for begin in range(done, n, 512):
                end = min(begin+512, n)
                for i, wave in pool.map(load, range(begin, end)):
                    array[i] = wave
                array.flush()
                write_json(progress_file, {"completed": end, "total": n})
                if begin % 2048 == 0 or end == n:
                    print(f"Raw mixture cache {split}: {end}/{n}", flush=True)
        # Direct sample-wise checks make FRCRN/raw cache confusion detectable.
        for i in np.linspace(0, n-1, 12, dtype=int):
            _, wave = load(int(i))
            assert np.array_equal(array[i], wave)
        write_json(directory/f"{split}_verified.json", {"complete": True, "shape": list(array.shape),
                                                       "sha256": sha(filename), "exact_wav_spot_checks": 12})
        del array


def score(probs, targets):
    import numpy as np
    from sklearn.metrics import precision_recall_fscore_support, average_precision_score
    prediction = probs.argmax(1)
    p, r, f, _ = precision_recall_fscore_support(targets, prediction, labels=np.arange(36), average="macro", zero_division=0)
    _, _, micro, _ = precision_recall_fscore_support(targets, prediction, average="micro", zero_division=0)
    aps = [average_precision_score(targets == i, probs[:, i]) for i in range(36) if (targets == i).any()]
    return {"n": len(targets), "acc": float(np.mean(prediction == targets)), "precision": float(p), "recall": float(r),
            "macro_f1": float(f), "micro_f1": float(micro), "mAP": float(np.mean(aps))}


def summarize():
    from noise_pipeline.ce9_models import MODEL_NAMES
    records = []
    configs = {}
    smoke_file = OUT/"model_smoke_checks.json"
    smoke = json.loads(smoke_file.read_text(encoding="utf-8")) if smoke_file.exists() else {}
    lines = ["# Benchmark 9 mô hình CE-only trên mixture v2 đã clean", "",
             "30.240 train, 3.064 validation. Mixture WAV gốc, không dùng FRCRN/KD/MoE. Không mở tập test.",
             "Các model fine-tune toàn bộ backbone với head 36 lớp mới. AudioSet và ImageNet là nguồn pretrained khác nhau; đây là so sánh các cấu hình chuyển giao, không cô lập ảnh hưởng riêng kiến trúc.", "",
             "Mỗi mô hình chạy 15 epoch; chọn best.pt theo val_acc. Không TTA/ensemble. Precision/recall là macro; mAP từ xác suất softmax, không từ nhãn argmax.", "",
             "| Model | Parameters (M) | Trainable (M) | Trạng thái | Best epoch | acc | precision | recall | macro_f1 | micro_f1 | mAP |",
             "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name in MODEL_NAMES:
        config_file = OUT/name/"config.json"
        config = json.loads(config_file.read_text(encoding="utf-8")) if config_file.exists() else {}
        if config:
            configs[name] = config
        parameters = config.get("parameters", smoke.get(name, {}).get("parameters"))
        trainable = config.get("trainable", smoke.get(name, {}).get("trainable"))
        size_cells = " | ".join(f"{v/1e6:.3f}" if v is not None else "—" for v in (parameters, trainable))
        result_file = OUT/name/"results.json"
        if result_file.exists():
            result = json.loads(result_file.read_text(encoding="utf-8"))
            overall = result["validation"]["all"]
            lines.append(f"| {name} | {size_cells} | complete | {result['best_epoch']} | " + " | ".join(f"{100*overall[k]:.2f}%" for k in METRICS) + " |")
            for split, values in result["validation"].items():
                records.append({"model": name, "parameters": parameters, "trainable": trainable,
                                "snr": split, "best_epoch": result["best_epoch"], **values})
        else:
            status_file = OUT/name/"status.json"
            status = json.loads(status_file.read_text()).get("status", "pending") if status_file.exists() else "pending"
            lines.append(f"| {name} | {size_cells} | {status} | — | — | — | — | — | — | — |")
    lines.extend(["", "Parameters/Trainable tính bằng triệu tham số (M = 1.000.000), gồm head 36 lớp. Tổng tham số PANNs bao gồm các tham số frontend STFT/Mel cố định; cột Trainable chỉ đếm tham số được tối ưu. Không tính optimizer state hoặc chỉ suy ra số tham số từ kích thước file checkpoint.",
                  "Config tổng hợp: CONFIG_9_MODELS_VI.md; bản đầy đủ có thể xử lý bằng code: CONFIG_9_MODELS.json. Các config.json gốc của từng lần train được giữ nguyên.",
                  "Bảng này cập nhật khi từng mô hình hoàn tất. Metric theo 6 SNR và all nằm trong metrics_all_models.csv và từng model/metrics_per_snr.csv.",
                  "Mỗi model có train.log, history.csv/json, epoch_metrics.jsonl, predictions.npz, per_class.csv và checkpoints/best.pt, last.pt.",
                  "train_acc/train_mAP là tổng hợp dự đoán online trong epoch (train mode, trọng số thay đổi theo bước), không phải đánh giá lại model cuối epoch trên train. val_* dùng model eval cố định cuối epoch."])
    (OUT/"BAO_CAO_CE9_VI.md").write_text("\n".join(lines), encoding="utf-8")
    with (OUT/"metrics_all_models.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "parameters", "trainable", "snr", "best_epoch", "n", *METRICS])
        writer.writeheader()
        writer.writerows(records)
    write_json(OUT/"CONFIG_9_MODELS.json", {"description": "Exact saved training configs; no hyperparameters reconstructed from current defaults",
                                         "models": configs})
    config_lines = ["# Config đã chạy cho 9 model CE-only", "",
                    "Nguồn: config.json đã lưu của từng model. Các số sau là cấu hình thực tế của lượt train, không thay đổi trọng số hoặc chạy lại thí nghiệm.", "",
                    "## Dữ liệu và cách đánh giá", "",
                    "Mixture gốc từ HF mix-dataset-v2 đã clean; 30.240 train, 3.064 validation (eval), 36 lớp, mono 16 kHz, 4 giây. Không dùng FRCRN hoặc tập test. Sáu SNR: −5, 0, 5, 10, 15, 20 dB và all.",
                    "Các metric: acc, macro precision, macro recall, macro_f1, micro_f1, mAP từ softmax probabilities.", "",
                    "## Cấu hình riêng từng model", "",
                    "| Model | Parameters | Trainable | Micro-batch | Accumulation | Effective batch | LR backbone | LR head | Pretrained |",
                    "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for name, c in configs.items():
        config_lines.append(f"| {name} | {c['parameters']:,} | {c['trainable']:,} | {c['batch_size']} | {c['gradient_accumulation']} | {c['effective_batch_size']} | {c['backbone_lr']:.0e} | {c['head_lr']:.0e} | {c['initialization']['domain']} |")
    if configs:
        shared_keys = ("epochs", "seed", "loss", "label_smoothing", "mixup", "specaugment", "teacher", "frcrn",
                       "optimizer", "weight_decay", "warmup_epochs", "scheduler", "clip_grad_norm", "amp", "selection")
        common = {key: next(iter(configs.values()))[key] for key in shared_keys
                  if all(c[key] == next(iter(configs.values()))[key] for c in configs.values())}
        config_lines.extend(["", "## Cấu hình chung (đã đối chiếu giữa các model)", "", "```json",
                             json.dumps(common, ensure_ascii=False, indent=2), "```", "",
                             "## Frontend và khởi tạo", "",
                             "Mỗi model dùng head 36 lớp mới. PANNs resample 32 kHz và dùng frontend gốc 64 Mel; CED dùng frontend gốc 16 kHz/64 Mel và mean pooling, không thêm CLS/GREP/MoE; EfficientAT resample 32 kHz và dùng frontend gốc 128 Mel; BEATs dùng Kaldi Fbank gốc và mở đủ 12 block.",
                             "ViT-S/MobileViT/MobileNet dùng ImageNet pretrained trên ảnh phổ 128 Mel: FFT 1024/window 400/hop 160, z-score mỗi mẫu, clip ±3 → [0,1], resize theo pretrained config, lặp RGB và chuẩn hóa mean/std của model. CED/PANNs/EfficientAT/BEATs dùng AudioSet pretrained; vì vậy bảng này không cô lập riêng ảnh hưởng kiến trúc.", "",
                             "## Cách đọc log", "",
                             "train_loss/train_acc/train_mAP đo online trong train mode qua các minibatch; trọng số thay đổi trong epoch. val_loss/val_acc/val_mAP đo trong eval mode cố định cuối epoch. Không dùng TTA hoặc ensemble. Best checkpoint được chọn theo val_acc cao nhất; hòa thì giữ epoch sớm hơn.", "",
                             "CONFIG_9_MODELS.json giữ nguyên đầy đủ nội dung config.json của cả 9 model, gồm receipt pretrained, số tham số và hash adapter/cache. Revision nguồn và hash pretrained nằm trong pretrained_provenance.json; phiên bản thư viện trong run_provenance.json."])
    (OUT/"CONFIG_9_MODELS_VI.md").write_text("\n".join(config_lines), encoding="utf-8")


def train_one(name):
    import numpy as np
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from noise_pipeline.ce9_models import build_model
    torch.set_num_threads(4)
    seed_all()
    root, labels, meta = read_metadata()
    run = OUT/name
    run.mkdir(parents=True, exist_ok=True)
    if (run/"results.json").exists():
        print(name, "already complete", flush=True)
        return
    log = (run/"train.log").open("a", encoding="utf-8", buffering=1)

    def emit(value):
        print(value, flush=True)
        log.write(value+"\n")

    class CachedMixture(Dataset):
        def __init__(self, split):
            self.array = np.load(OUT/"raw_mixture_cache"/f"{split}_mixture.npy", mmap_mode="r")
            self.targets = np.array(meta[split]["targets"], dtype=np.int64)
            self.snrs = np.array(meta[split]["snrs"], dtype=np.int16)
            assert self.array.shape == (len(self.targets), 64000)

        def __len__(self):
            return len(self.targets)

        def __getitem__(self, i):
            return torch.from_numpy(self.array[i].copy()), int(self.targets[i]), int(self.snrs[i])

    batch = {"beats12": 8, "ced_small": 16, "vit_small": 16, "mobilevit_s": 16}.get(name, 32)
    accumulation = 64//batch
    backbone_lr, head_lr = (1e-5, 1e-4) if name == "beats12" else (3e-5, 3e-4)
    config = {"model": name, "epochs": 15, "seed": 2026, "batch_size": batch,
              "gradient_accumulation": accumulation, "effective_batch_size": 64, "loss": "unweighted CrossEntropy on raw logits",
              "label_smoothing": 0., "mixup": False, "specaugment": False, "teacher": False, "frcrn": False,
              "backbone_lr": backbone_lr, "head_lr": head_lr, "optimizer": "AdamW", "weight_decay": .01,
              "warmup_epochs": 2, "scheduler": "cosine", "clip_grad_norm": 1., "amp": "bfloat16",
              "selection": "max validation acc, earliest epoch on ties", "input": "raw mixture WAV cache; 16kHz mono 4s",
              "train_samples": 30240, "val_samples": 3064, "test_evaluated": False,
              "precision_recall_average": "macro", "mAP": "mean class AP from softmax probabilities",
              "train_metric_semantics": "online train-mode predictions across changing minibatch weights",
              "adapter_sha256": sha(ROOT/"noise_pipeline/ce9_models.py"),
              "raw_cache_provenance_sha256": sha(OUT/"raw_mixture_cache/provenance.json")}
    device = torch.device("cuda")
    model = build_model(name, OUT/"pretrained").to(device)
    config["initialization"] = model.receipt
    config["parameters"] = sum(p.numel() for p in model.parameters())
    config["trainable"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    write_json(run/"config.json", config)
    head_fragments = ("head.", "outputlayer.1.", "fc_audioset.", "classifier.5.", "classifier.weight", "classifier.bias")
    backbone, heads = [], []
    for key, parameter in model.named_parameters():
        if parameter.requires_grad:
            (heads if any(part in key for part in head_fragments) else backbone).append(parameter)
    assert heads and backbone
    optimizer = torch.optim.AdamW([{"params": backbone, "lr": backbone_lr}, {"params": heads, "lr": head_lr}], weight_decay=.01)
    train_data, val_data = CachedMixture("train"), CachedMixture("eval")
    train_loader = DataLoader(train_data, batch_size=batch, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=batch, shuffle=False, num_workers=0, pin_memory=True)
    steps = math.ceil(len(train_loader)/accumulation)
    warmup = 2*steps

    def lr_scale(step):
        if step < warmup:
            return (step+1)/warmup
        return .5*(1+math.cos(math.pi*(step-warmup)/max(1, 15*steps-warmup)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    checkpoints = run/"checkpoints"
    checkpoints.mkdir(exist_ok=True)
    last = checkpoints/"last.pt"
    history, first_epoch, best_acc, best_epoch = [], 0, -1., 0
    if last.exists():
        state = torch.load(last, map_location="cpu", weights_only=False)
        assert state["config"] == config, "Cannot resume with changed config"
        model.load_state_dict(state["model_state_dict"], strict=True)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        first_epoch, history, best_acc, best_epoch = state["epoch"], state["history"], state["best_acc"], state["best_epoch"]
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        del state

    def evaluate():
        model.eval()
        ps, ys, ss, total = [], [], [], 0.
        with torch.inference_mode():
            for wave, y, snr in val_loader:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(wave.to(device, non_blocking=True)).float()
                    loss = nn.functional.cross_entropy(logits, y.to(device))
                if not torch.isfinite(logits).all() or not torch.isfinite(loss):
                    raise FloatingPointError("Invalid validation outputs")
                total += loss.item()*len(y)
                ps.append(logits.softmax(1).cpu().numpy())
                ys.append(y.numpy())
                ss.append(snr.numpy())
        return total/len(val_data), np.concatenate(ps), np.concatenate(ys), np.concatenate(ss)

    for epoch in range(first_epoch, 15):
        write_json(run/"status.json", {"status": "training", "epoch": epoch+1, "epochs": 15, "best_acc": best_acc})
        model.train()
        started = time.monotonic()
        probabilities, target_list = [], []
        total_loss, samples = 0., 0
        optimizer.zero_grad(set_to_none=True)
        for index, (wave, y, _) in enumerate(train_loader):
            window_start = (index//accumulation)*accumulation*batch
            denominator = min(64, len(train_data)-window_start)
            wave, y = wave.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(wave).float()
                loss = nn.functional.cross_entropy(logits, y)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite train loss")
            (loss*len(y)/denominator).backward()
            total_loss += loss.item()*len(y)
            samples += len(y)
            probabilities.append(logits.detach().softmax(1).cpu().numpy())
            target_list.append(y.cpu().numpy())
            if (index+1)%accumulation == 0 or index+1 == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if index%100 == 0 or index+1 == len(train_loader):
                emit(f"[{name}] epoch {epoch+1}/15 step {index+1}/{len(train_loader)} samples={samples}/30240 train_loss={total_loss/samples:.6f}")
                write_json(run/"status.json", {"status": "training", "epoch": epoch+1, "step": index+1,
                                               "steps": len(train_loader), "samples": samples, "best_acc": best_acc})
        train_score = score(np.concatenate(probabilities), np.concatenate(target_list))
        val_loss, val_probs, y_val, snrs = evaluate()
        val_metrics = {"all": score(val_probs, y_val)}
        val_metrics.update({str(s): score(val_probs[snrs == s], y_val[snrs == s]) for s in SNRS})
        row = {"epoch": epoch+1, "train_loss": total_loss/samples, "train_acc": train_score["acc"], "train_mAP": train_score["mAP"],
               "val_loss": val_loss, "val_acc": val_metrics["all"]["acc"], "val_mAP": val_metrics["all"]["mAP"],
               "seconds": time.monotonic()-started, "validation": val_metrics}
        is_best = row["val_acc"] > best_acc
        if is_best:
            best_acc, best_epoch = row["val_acc"], epoch+1
        history.append(row)
        state = {"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                 "scheduler_state_dict": scheduler.state_dict(), "epoch": epoch+1, "best_acc": best_acc, "best_epoch": best_epoch,
                 "history": history, "config": config, "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                 "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()}
        tmp = checkpoints/"last.tmp.pt"
        torch.save(state, tmp)
        tmp.replace(last)
        if is_best:
            shutil.copyfile(last, checkpoints/"best.pt")
        del state
        write_json(run/"history.json", history)
        fields = ["epoch", "train_loss", "train_acc", "train_mAP", "val_loss", "val_acc", "val_mAP", "seconds"]
        with (run/"history.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows({k: h[k] for k in fields} for h in history)
        # Rewrite from committed history so a resumed interrupted epoch cannot duplicate log rows.
        with (run/"epoch_metrics.jsonl").open("w", encoding="utf-8") as f:
            for h in history:
                f.write(json.dumps(h)+"\n")
        emit(f"[{name}] EPOCH {epoch+1}/15 train_loss={row['train_loss']:.6f} train_acc={row['train_acc']:.6f} train_mAP={row['train_mAP']:.6f} val_loss={val_loss:.6f} val_acc={row['val_acc']:.6f} val_mAP={row['val_mAP']:.6f} best={is_best} time={row['seconds']:.1f}s")
        for snr, metrics in val_metrics.items():
            emit(f"[{name}] VAL snr={snr} n={metrics['n']} " + " ".join(f"{k}={metrics[k]:.6f}" for k in METRICS))
    checkpoint = torch.load(checkpoints/"best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    val_loss, probs, y, snrs = evaluate()
    validation = {"all": score(probs, y), **{str(s): score(probs[snrs == s], y[snrs == s]) for s in SNRS}}
    expected = history[best_epoch-1]["validation"]["all"]
    assert abs(validation["all"]["acc"]-expected["acc"]) < 1e-12
    np.savez(run/"predictions.npz", probabilities=probs, targets=y, snrs=snrs, sample_ids=np.array(meta["eval"]["ids"]))
    from sklearn.metrics import precision_recall_fscore_support, average_precision_score
    pr, rc, f1, support = precision_recall_fscore_support(y, probs.argmax(1), labels=np.arange(36), zero_division=0)
    with (run/"per_class.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "support", "precision", "recall", "f1", "AP"])
        for i, label in enumerate(labels):
            writer.writerow([label, int(support[i]), pr[i], rc[i], f1[i], average_precision_score(y == i, probs[:, i])])
    with (run/"metrics_per_snr.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["snr", "n", *METRICS])
        writer.writeheader()
        writer.writerows({"snr": snr, **v} for snr, v in validation.items())
    result = {"model": name, "best_epoch": best_epoch, "epochs": 15, "split": "eval", "test_evaluated": False,
              "checkpoint_sha256": sha(checkpoints/"best.pt"), "val_loss": val_loss, "validation": validation}
    write_json(run/"results.json", result)
    write_json(run/"status.json", {"status": "complete", "best_epoch": best_epoch, "validation": validation["all"]})
    emit(f"[{name}] COMPLETE best_epoch={best_epoch} {json.dumps(validation['all'])}")


def queue():
    from noise_pipeline.ce9_models import MODEL_NAMES
    checks = json.loads((OUT/"model_smoke_checks.json").read_text())
    assert all(checks.get(n, {}).get("status") == "passed" for n in MODEL_NAMES), "All adapters must pass smoke first"
    assert (OUT/"pretrained_provenance.json").is_file()
    prepare_raw_cache()
    summarize()
    failures = []
    for i, name in enumerate(MODEL_NAMES):
        if (OUT/name/"results.json").exists():
            continue
        write_json(OUT/"queue_status.json", {"status": "running", "current": name, "index": i+1, "total": 9, "failed": failures})
        print(f"START {i+1}/9: {name}", flush=True)
        result = subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()), "--model", name], cwd=ROOT)
        if result.returncode:
            failures.append(name)
        summarize()
    completed = [n for n in MODEL_NAMES if (OUT/n/"results.json").exists()]
    write_json(OUT/"queue_status.json", {"status": "complete" if len(completed) == 9 else "completed_with_failures", "completed": completed, "failed": failures})
    summarize()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--prepare-data", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    if args.prepare_data:
        prepare_raw_cache()
    elif args.summarize:
        summarize()
    elif args.model:
        train_one(args.model)
    else:
        queue()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        error = traceback.format_exc()
        print(error, file=sys.stderr, flush=True)
        if "--model" in sys.argv:
            name = sys.argv[sys.argv.index("--model")+1]
            write_json(OUT/name/"status.json", {"status": "failed", "error": error})
            (OUT/name/"error.log").write_text(error, encoding="utf-8")
        raise
