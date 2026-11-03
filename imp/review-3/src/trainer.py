"""Existing Review-3 Trainer with effective group schedules and complete CE logs."""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
from pathlib import Path
import random
import re
import time
from typing import Any

import numpy as np
import torch
from torch import nn
from src.metrics import compute_macro_acc, compute_metrics, compute_snr_breakdown
from src.utils import count_parameters


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader=None, config=None,
                 device=None, logger=None, checkpoint_dir="checkpoints", model_name="model"):
        self.config = config or {}
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader, self.val_loader = train_loader, val_loader
        self.test_loader = test_loader
        self.logger = logger or logging.getLogger("trainer")
        self.checkpoint_dir = Path(checkpoint_dir) / model_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        t = self.config.get("training", {})
        self.epochs = int(t.get("epochs", 50))
        self.lr = float(t.get("lr", 3e-4))
        self.weight_decay = float(t.get("weight_decay", .05))
        self.grad_clip = float(t.get("grad_clip", 1.))
        self.label_smoothing = float(t.get("label_smoothing", .1))
        self.patience = int(t.get("patience", 8))
        self.accumulation = int(t.get("accumulation", 1))
        self.effective_batch = int(t.get("batch_size", train_loader.batch_size))
        self.metric_for_best = self.config.get("logging", {}).get("metric_for_best", "macro_acc")
        self.criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        self.use_amp = self.device.type == "cuda" and t.get("amp", "bf16") in ("bf16", "bfloat16")
        self.optimizer = torch.optim.AdamW(self._parameter_groups(), betas=tuple(t.get("betas", (.9, .999))))
        steps = math.ceil(len(train_loader) / self.accumulation)
        total_steps = self.epochs * steps
        warmup = max(int(total_steps * float(t.get("warmup_ratio", .05))), 1)
        functions = []
        for group in self.optimizer.param_groups:
            floor = float(t.get("min_lr", 1e-6)) / group["reference_lr"]
            def schedule(step, floor=floor):
                if step < warmup:
                    return (step + 1) / warmup
                progress = min(max((step-warmup)/max(total_steps-warmup, 1), 0.), 1.)
                return floor + (1-floor) * .5 * (1+math.cos(math.pi*progress))
            functions.append(schedule)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, functions)
        self.ema_decay = float(t.get("ema_decay", .999))
        self.ema = {name: value.detach().clone() for name, value in self.model.named_parameters()}
        self.best_metric, self.best_epoch, self.patience_counter = -1., -1, 0
        self.history = []
        self.first_epoch = 0
        self.last_predictions = None
        self.train_mAP = 0.
        self.drive_fs = None
        self.drive_dir = None
        drive_root = self.config.get("logging", {}).get("drive_dir")
        if drive_root:
            from src.drive_checkpoint import DriveCheckpointStore
            self.drive_fs = DriveCheckpointStore()
            self.drive_dir = str(drive_root).strip("/") + "/" + model_name
            self.drive_fs.makedirs(self.drive_dir, exist_ok=True)
            self._restore_drive()
        last = self.checkpoint_dir / "last.pt"
        if last.is_file():
            state = torch.load(last, map_location="cpu", weights_only=False)
            if state["config"] != self.config:
                raise ValueError("Existing checkpoint belongs to a different effective config; refusing overwrite")
            self.model.load_state_dict(state["model_state_dict"], strict=True)
            self.optimizer.load_state_dict(state["optimizer_state_dict"])
            self.scheduler.load_state_dict(state["scheduler_state_dict"])
            self.ema = {k: v.to(self.device) for k, v in state["ema"].items()}
            self.first_epoch = state["epoch"]
            self.history = state["history"]
            self.best_metric, self.best_epoch = state["best_metric"], state["best_epoch"]
            self.patience_counter = state["patience_counter"]
            random.setstate(state["python_rng"])
            np.random.set_state(state["numpy_rng"])
            torch.set_rng_state(state["torch_rng"])
            if self.device.type == "cuda":
                torch.cuda.set_rng_state_all(state["cuda_rng"])
            if state.get("sampler_rng") is not None:
                self.train_loader.sampler.generator.set_state(state["sampler_rng"])
            del state

    def _restore_drive(self):
        if (self.checkpoint_dir/"last.pt").is_file():
            return
        marker=self.drive_dir+"/resume_manifest.json"
        if not self.drive_fs.exists(marker):
            return
        with self.drive_fs.open(marker,"rb") as stream:
            manifest=json.loads(stream.read())
        staged=[]
        for field,local in (("best","best.pt"),("last","last.pt")):
            name=manifest[field]["file"]
            if not re.fullmatch(r"(?:last|resume_best)_e\d+\.pt",name):
                raise ValueError("Invalid checkpoint backup filename")
            target=self.checkpoint_dir/local
            temp=target.with_suffix(".restore.pt")
            self.drive_fs.get_file(self.drive_dir+"/"+name,str(temp))
            if hashlib.sha256(temp.read_bytes()).hexdigest()!=manifest[field]["sha256"]:
                raise ValueError("Checkpoint backup SHA256 mismatch")
            staged.append((temp,target))
        for temp,target in staged:
            temp.replace(target)
        self.logger.info("Restored matching last/best checkpoint pair from Drive at epoch %d",manifest["epoch"])

    def _backup_drive(self,epoch,improved=False,final=False):
        if self.drive_fs is None:
            return
        try:
            self.drive_fs.invalidate_cache(self.drive_dir)
            self.drive_fs.ls(self.drive_dir)
            def upload(local,remote):
                self.drive_fs.put_file(str(self.checkpoint_dir/local),self.drive_dir+"/"+remote)
            if improved or final:
                upload("best.pt","best.pt")
            for name in ("history.json","history.csv","effective_config.json","optimizer_groups.json"):
                if (self.checkpoint_dir/name).exists():upload(name,name)
            interval=int(self.config.get("logging",{}).get("backup_every_epochs",5))
            if epoch==1 or epoch%interval==0 or final:
                marker=self.drive_dir+"/resume_manifest.json"
                previous={}
                if self.drive_fs.exists(marker):
                    with self.drive_fs.open(marker,"rb") as stream:previous=json.loads(stream.read())
                manifest={"epoch":epoch,"best_epoch":self.best_epoch}
                for field,local,name in (("last","last.pt",f"last_e{epoch:03d}.pt"),("best","best.pt",f"resume_best_e{epoch:03d}.pt")):
                    upload(local,name)
                    manifest[field]={"file":name,"sha256":hashlib.sha256((self.checkpoint_dir/local).read_bytes()).hexdigest()}
                local_marker=self.checkpoint_dir/"resume_manifest.json"
                write_json(local_marker,manifest)
                self.drive_fs.put_file(str(local_marker),marker)
                # Retain the previous valid generation until the new manifest is committed.
                for field in ("last","best"):
                    old=previous.get(field,{}).get("file","")
                    if old and old!=manifest[field]["file"] and re.fullmatch(r"(?:last|resume_best)_e\d+\.pt",old):
                        self.drive_fs.rm(self.drive_dir+"/"+old)
            if final:
                for name in ("results.json","metrics_per_snr.csv","validation_predictions.npz"):
                    if (self.checkpoint_dir/name).exists():upload(name,name)
            write_json(self.checkpoint_dir/"backup_status.json",{"ok":True,"epoch":epoch,"drive_dir":self.drive_dir})
        except Exception as error:
            self.logger.warning("Drive backup failed at epoch %d (%s); local checkpoint is retained",epoch,type(error).__name__)
            write_json(self.checkpoint_dir/"backup_status.json",{"ok":False,"epoch":epoch,"error_type":type(error).__name__})

    def _parameter_groups(self):
        training = self.config.get("training", {})
        head_lr = float(training.get("head_lr", self.lr))
        layer_decay = float(training.get("layer_decay", 1.))
        no_decay = ("bias", "ln", "norm", "pos", "a_att", "a_gru")
        groups = {}
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            head = name.startswith("head.") or any(k in name for k in (".head.",".classifier.",".fc_audioset.",".outputlayer.1."))
            reference = head_lr if head else self.lr
            scale = 1.
            if self.model_name.startswith("ced_") and layer_decay < 1 and not head:
                match = re.search(r"(?:^|\.)blocks\.(\d+)\.", name)
                if match:
                    scale = layer_decay ** (11-int(match[1]))
                elif any(k in name for k in ("patch_embed", "cls_token", "rel_pos_bias", "freq_pos_embed", "time_pos_embed")):
                    scale = layer_decay ** 12
            decay = 0. if any(k in name for k in no_decay) else self.weight_decay
            key = (reference, scale, decay)
            if key not in groups:
                groups[key] = {"params": [], "lr": reference*scale, "reference_lr": reference,
                               "layer_scale": scale, "weight_decay": decay, "parameter_names": []}
            groups[key]["params"].append(p)
            groups[key]["parameter_names"].append(name)
        return list(groups.values())

    @torch.no_grad()
    def _update_ema(self):
        for name, parameter in self.model.named_parameters():
            self.ema[name].mul_(self.ema_decay).add_(parameter.detach(), alpha=1-self.ema_decay)

    def train_epoch(self, epoch):
        self.model.train()
        total_loss, total_samples, correct = 0., 0, 0
        probabilities, targets = [], []
        self.optimizer.zero_grad(set_to_none=True)
        for index, batch in enumerate(self.train_loader):
            wave = batch["waveform"].to(self.device, non_blocking=True)
            y = batch["target_idx"].to(self.device, non_blocking=True)
            window_start = (index//self.accumulation)*self.effective_batch
            denominator = min(self.effective_batch, len(self.train_loader.dataset)-window_start)
            with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.use_amp):
                logits = self.model(wave).float()
                loss = self.criterion(logits, y)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite train loss")
            (loss*len(y)/denominator).backward()
            if (index+1)%self.accumulation == 0 or index+1 == len(self.train_loader):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip, error_if_nonfinite=True)
                self.optimizer.step()
                self.scheduler.step()
                self._update_ema()
                self.optimizer.zero_grad(set_to_none=True)
            total_loss += float(loss.item())*len(y)
            total_samples += len(y)
            correct += int((logits.argmax(1) == y).sum())
            probabilities.append(logits.detach().softmax(1).cpu().numpy())
            targets.append(y.cpu().numpy())
            if index % 100 == 0 or index+1 == len(self.train_loader):
                self.logger.info("TRAIN epoch=%d/%d step=%d/%d n=%d train_loss=%.6f train_acc=%.6f",
                                 epoch, self.epochs, index+1, len(self.train_loader), total_samples,
                                 total_loss/total_samples, correct/total_samples)
                write_json(self.checkpoint_dir/"status.json", {"status": "training", "epoch": epoch,
                           "step": index+1, "steps": len(self.train_loader), "samples": total_samples})
        self.train_mAP = compute_metrics(np.concatenate(probabilities), np.concatenate(targets))["mAP"]
        return total_loss/max(total_samples, 1), correct/max(total_samples, 1)

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        total_loss, total = 0., 0
        probabilities, targets, snrs, ids = [], [], [], []
        for batch in loader:
            wave = batch["waveform"].to(self.device, non_blocking=True)
            y = batch["target_idx"].to(self.device, non_blocking=True)
            with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.use_amp):
                logits = self.model(wave).float()
                loss = self.criterion(logits, y)
            if not torch.isfinite(logits).all() or not torch.isfinite(loss):
                raise FloatingPointError("Non-finite validation output")
            probabilities.append(logits.softmax(1).cpu().numpy())
            targets.append(y.cpu().numpy())
            snrs.append(batch["snr_db"].numpy())
            ids.extend(batch["sample_id"])
            total_loss += loss.item()*len(y)
            total += len(y)
        probs, y, snrs = np.concatenate(probabilities), np.concatenate(targets), np.concatenate(snrs)
        metrics = compute_metrics(probs, y)
        metrics["macro_acc"] = compute_macro_acc(probs, y, snrs)
        breakdown = compute_snr_breakdown(probs, y, snrs)
        breakdown["all"] = {**metrics, "count": len(y)}
        self.last_predictions = {"probabilities": probs, "targets": y, "snrs": snrs, "sample_ids": np.array(ids)}
        return total_loss/max(total, 1), metrics, breakdown

    def _save_history(self):
        write_json(self.checkpoint_dir/"history.json", {"history": self.history, "best_metric": self.best_metric,
                                                     "best_epoch": self.best_epoch, "test_evaluated": False})
        fields = ("epoch", "train_loss", "train_acc", "train_mAP", "val_loss", "val_acc", "val_mAP", "val_macro_acc", "epoch_time_s")
        with (self.checkpoint_dir/"history.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows({k: row[k] for k in fields} for row in self.history)

    def fit(self):
        total_p, train_p = count_parameters(self.model)
        self.logger.info("START %s parameters=%d trainable=%d", self.model_name, total_p, train_p)
        self.logger.info("EFFECTIVE_CONFIG %s", json.dumps(self.config, ensure_ascii=False))
        write_json(self.checkpoint_dir/"effective_config.json", self.config)
        write_json(self.checkpoint_dir/"optimizer_groups.json", [{k: v for k, v in g.items() if k != "params"}
                                                               for g in self.optimizer.param_groups])
        for epoch in range(self.first_epoch+1, self.epochs+1):
            if self.patience_counter >= self.patience:
                break
            started = time.monotonic()
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, metrics, breakdown = self.evaluate(self.val_loader)
            current = metrics[self.metric_for_best]  # No silent fallback to Macro-F1.
            better = current > self.best_metric
            if better:
                self.best_metric, self.best_epoch, self.patience_counter = current, epoch, 0
            else:
                self.patience_counter += 1
            row = {"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "train_mAP": self.train_mAP,
                   "val_loss": val_loss, "val_acc": metrics["acc"], "val_mAP": metrics["mAP"],
                   "val_macro_acc": metrics["macro_acc"], "val_metrics": metrics, "snr_breakdown": breakdown,
                   "epoch_time_s": time.monotonic()-started}
            self.history.append(row)
            self.logger.info("EPOCH %d/%d train_loss=%.6f train_acc=%.6f train_mAP=%.6f val_loss=%.6f val_acc=%.6f val_mAP=%.6f val_macro_acc=%.6f best=%s seconds=%.1f",
                             epoch, self.epochs, train_loss, train_acc, self.train_mAP, val_loss, metrics["acc"], metrics["mAP"], current, better, row["epoch_time_s"])
            for tag, result in breakdown.items():
                self.logger.info("VAL SNR=%s n=%d %s", tag, result["count"],
                                 " ".join(f"{k}={result[k]:.6f}" for k in ("acc", "precision", "recall", "macro_f1", "micro_f1", "mAP")))
            sampler = self.train_loader.sampler
            state = {"epoch": epoch, "model_state_dict": self.model.state_dict(), "optimizer_state_dict": self.optimizer.state_dict(),
                     "scheduler_state_dict": self.scheduler.state_dict(), "ema": self.ema, "best_metric": self.best_metric,
                     "best_epoch": self.best_epoch, "patience_counter": self.patience_counter, "history": self.history,
                     "config": self.config, "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                     "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if self.device.type == "cuda" else [],
                     "sampler_rng": sampler.generator.get_state() if getattr(sampler, "generator", None) is not None else None}
            temp = self.checkpoint_dir/"last.tmp.pt"
            torch.save(state, temp)
            temp.replace(self.checkpoint_dir/"last.pt")
            if better:
                best = {"epoch": epoch, "model_state_dict": self.model.state_dict(), "best_metric": self.best_metric,
                        "val_metrics": metrics, "snr_breakdown": breakdown, "config": self.config}
                temp = self.checkpoint_dir/"best.tmp.pt"
                torch.save(best, temp)
                temp.replace(self.checkpoint_dir/"best.pt")
            del state
            self._save_history()
            self._backup_drive(epoch,improved=better)
        best = torch.load(self.checkpoint_dir/"best.pt", map_location="cpu", weights_only=False)
        self.model.load_state_dict(best["model_state_dict"], strict=True)
        val_loss, metrics, breakdown = self.evaluate(self.val_loader)
        if abs(metrics["macro_acc"]-self.best_metric) > 1e-12:
            raise AssertionError("Reloaded checkpoint does not reproduce selected validation macro_acc")
        np.savez(self.checkpoint_dir/"validation_predictions.npz", **self.last_predictions)
        with (self.checkpoint_dir/"metrics_per_snr.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            fields = ("snr", "count", "acc", "precision", "recall", "macro_f1", "micro_f1", "mAP")
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows({"snr": tag, **{k: v[k] for k in fields if k != "snr"}} for tag, v in breakdown.items())
        result = {"model_name": self.model_name, "best_epoch": self.best_epoch, "best_metric": self.best_metric,
                  "metric_for_best": self.metric_for_best, "epochs_ran": len(self.history), "history": self.history,
                  "validation": metrics, "snr_breakdown": breakdown, "val_loss": val_loss,
                  "parameters": total_p, "trainable": train_p, "test_evaluated": False, "test": {},
                  "initialization": self.config.get("initialization", {"mode":"review3_scratch"}),
                  "pretrained_receipt": self.config.get("pretrained_receipt"),
                  "checkpoint_sha256": hashlib.sha256((self.checkpoint_dir/"best.pt").read_bytes()).hexdigest()}
        write_json(self.checkpoint_dir/"results.json", result)
        write_json(self.checkpoint_dir/"status.json", {"status": "complete", "best_epoch": self.best_epoch,
                   "epochs_ran": len(self.history), "validation": metrics})
        self._backup_drive(self.history[-1]["epoch"],final=True)
        return result
