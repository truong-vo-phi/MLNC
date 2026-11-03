"""
Main Entry-Point CLI for Review-3 Project:
Acoustic Noise Classification across Multiple SNR Levels using 9 Diverse Architectures.

Usage:
  python main.py list-models
  python main.py verify
  python main.py train --model pann_cnn10 [--epochs 30] [--smoke]
  python main.py train --model all [--smoke]
  python main.py eval --model pann_cnn10 [--checkpoint checkpoints/pann_cnn10/best_model.pt]
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import subprocess
import json
import sys
from datetime import datetime
from pathlib import Path

# Safe stdout/stderr encoding on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import torch
import yaml

PACKAGE_ROOT = Path(__file__).resolve().parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from src.dataset import get_dataloader, get_dataset
from src.metrics import compute_metrics, compute_snr_breakdown
from src.models import build_model, list_available_models
from src.trainer import Trainer
from src.utils import count_parameters, set_seed, setup_logger


def _file_sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def cmd_prepare(args):
    """CPU/network-only preparation; download only raw train/eval and genuine weights."""
    import importlib.util
    import shutil
    import zipfile
    from huggingface_hub import hf_hub_download
    sibling=PACKAGE_ROOT.parent/"mix_dataset_v2"
    assets=(PACKAGE_ROOT/"pretrained").resolve()
    assets.mkdir(exist_ok=True)
    sources=json.loads((PACKAGE_ROOT/"config/pretrained_sources.json").read_text())
    if not args.skip_data:
        data=Path(args.data_dir).resolve()
        data.mkdir(parents=True,exist_ok=True)
        if not (data/"mix-dataset-v2/eval/manifest.csv").is_file():
            revision=json.loads((sibling/"config/hf_dataset_revision.json").read_text())["revision"]
            tree=json.loads((sibling/"config/hf_dataset_files.json").read_text())
            info=next(x for x in tree if x["path"]=="mix-dataset-v2.zip")
            if shutil.disk_usage(data).free < info["size"]+9*2**30:
                raise RuntimeError("Need about 18 GiB free for ZIP + extraction; checkpoints need additional space")
            spec=importlib.util.spec_from_file_location("_review3_dataset_download",sibling/"download_team_mix_v2.py")
            download=importlib.util.module_from_spec(spec);spec.loader.exec_module(download)
            archive=download.download_ranges("phitruong0606/mix-dataset-v2",revision,"mix-dataset-v2.zip",data,info["size"])
            if _file_sha256(archive)!=info["lfs"]["oid"]:raise ValueError("Dataset SHA256 mismatch")
            destination=data.resolve()
            with zipfile.ZipFile(archive) as z:
                chosen=[entry for entry in z.infolist() if entry.filename.startswith(("mix-dataset-v2/train/","mix-dataset-v2/eval/")) or entry.filename=="mix-dataset-v2/labels.txt"]
                for i,entry in enumerate(chosen,1):
                    if not (destination/entry.filename).resolve().is_relative_to(destination):raise ValueError("Unsafe ZIP path")
                    z.extract(entry,destination)
                    if i%5000==0:print(f"Extracted {i}/{len(chosen)} train/eval entries",flush=True)
            if args.remove_archive:archive.unlink()
        print("Raw train/eval dataset ready:",data/"mix-dataset-v2",flush=True)
    spec=importlib.util.spec_from_file_location("_review3_asset_download",sibling/"prepare_ce9.py")
    prepare=importlib.util.module_from_spec(spec);spec.loader.exec_module(prepare)
    prepare.ASSETS=assets
    # The Zenodo stream can stall on cloud hosts. This pinned mirror is byte-identical.
    pann=sources["panns_cnn10_mirror"]
    pann_target=assets/"Cnn10_mAP=0.380.pth"
    if not pann_target.is_file():
        downloaded=hf_hub_download(pann["repo"],pann["filename"],revision=pann["revision"],local_dir=assets)
        if _file_sha256(downloaded)!=pann["sha256"]:raise ValueError("PANN mirror SHA256 mismatch")
        shutil.copyfile(downloaded,pann_target)
    if _file_sha256(pann_target)!=pann["sha256"]:raise ValueError("PANN pretrained SHA256 mismatch")
    print("Verified pretrained: Cnn10_mAP=0.380.pth",flush=True)
    checkpoints=[("https://github.com/fschmid56/EfficientAT/releases/download/v0.0.1/mn10_as_mAP_471.pt","mn10_as_mAP_471.pt")]
    for url,filename in checkpoints:
        target=Path(prepare.download(url,filename))
        if _file_sha256(target)!=sources["sha256"][filename]:raise ValueError(f"SHA256 mismatch: {filename}")
        print("Verified pretrained:",filename,flush=True)
    revisions={"ced_tiny":"ace276d29dd0bb3f3517b0fa8cf300738c409019","ced_mini":"26c3ebcae85d4330f4fc26763f029539a3afcda0","ced_small":"06bb40c5ec089e96867ebc5246be02441f4a71e4"}
    for name,revision in revisions.items():
        for filename in ("config.json","model.safetensors"):
            target=hf_hub_download("mispeech/"+name.replace("_","-"),filename,revision=revision,local_dir=assets/name)
            key=name+"/"+filename
            if key in sources["sha256"] and _file_sha256(target)!=sources["sha256"][key]:raise ValueError(f"SHA256 mismatch: {key}")
        print("Verified pretrained:",name,flush=True)
    beats=sources["beats"]
    target=hf_hub_download(beats["repo"],beats["filename"],revision=beats["revision"],local_dir=assets)
    if _file_sha256(target)!=beats["sha256"]:raise ValueError("BEATs pretrained SHA256 mismatch")
    print("Verified official BEATs contents by SHA256. No training was started.",flush=True)


def cmd_preflight(args):
    import gc
    import numpy as np
    from scripts.run_all_models import CANONICAL_MODELS,resolve_model_config,configure_existing_model
    config=yaml.safe_load((PACKAGE_ROOT/args.config).read_text(encoding="utf-8"))
    config["_models_config"]=str(PACKAGE_ROOT/"config/models.yaml")
    if args.assets:
        config["initialization"]["assets_dir"]=str(Path(args.assets).resolve())
    if args.beats_checkpoint:
        config["initialization"]["beats_checkpoint"]=str(Path(args.beats_checkpoint).resolve())
    if config.get("initialization",{}).get("mode")!="official_pretrained":raise ValueError("Preflight requires official_pretrained mode")
    selected=CANONICAL_MODELS if args.model=="all" else [args.model]
    torch.set_num_threads(4)
    receipt={}
    for name in selected:
        cfg=resolve_model_config(name,config)
        set_seed(cfg["training"]["seed"])
        model=configure_existing_model(name,cfg).to(args.device).eval()
        with torch.inference_mode():
            logits=model(torch.randn(1,64000,device=args.device)*.05)
        if logits.shape!=(1,36) or not torch.isfinite(logits).all():raise ValueError("Invalid pretrained forward")
        if not model.receipt.get("verified_pretrained_loader"):raise ValueError("Unverified initialization")
        total,trainable=count_parameters(model)
        receipt[name]={"verified":True,"parameters":total,"trainable":trainable,"receipt":model.receipt}
        print(f"PRETRAINED VERIFIED {name}: {total:,} parameters; forward finite; native frontend",flush=True)
        del model,logits
        gc.collect()
        if args.device.startswith("cuda"):torch.cuda.empty_cache()
    from src.trainer import write_json
    write_json(PACKAGE_ROOT/"logs/pretrained/preflight.json",receipt)
    print("Preflight finished. No optimizer step or training epoch was run.",flush=True)


def cmd_list_models(args: argparse.Namespace) -> None:
    models_yaml = PACKAGE_ROOT / "config" / "models.yaml"
    with open(models_yaml, encoding="utf-8") as f:
        cfg = yaml.safe_load(f).get("models", {})

    print("=" * 96)
    print("REVIEW-3: 9 ACOUSTIC NOISE CLASSIFICATION MODELS OVERVIEW")
    print("=" * 96)
    print(f"{'#':<3} | {'Model Key':<15} | {'Display Name':<16} | {'Family':<18} | {'Params (M)':<11} | {'Description'}")
    print("-" * 96)

    for idx, key in enumerate(list_available_models(), 1):
        info = cfg.get(key, {})
        disp_name = info.get("display_name", key)
        family = info.get("family", "Unknown")
        desc = info.get("description", "")
        # Compute exact parameters
        try:
            m = build_model(key, num_classes=36)
            total_p, _ = m.count_parameters()
            params_str = f"{total_p / 1e6:6.2f}M"
        except Exception:
            params_str = "N/A"

        print(f"{idx:<3} | {key:<15} | {disp_name:<16} | {family:<18} | {params_str:<11} | {desc}")

    print("=" * 96)


def cmd_verify(args: argparse.Namespace) -> None:
    from scripts.verify_models import verify_all_models
    success = verify_all_models()
    sys.exit(0 if success else 1)


def cmd_train(args: argparse.Namespace) -> None:
    from scripts.run_all_models import CANONICAL_MODELS, train_single_model
    from src.trainer import write_json
    path = PACKAGE_ROOT / args.config
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.lr:
        config["training"]["explicit_cli_lr"] = args.lr
    if args.data_root:
        config["dataset"]["data_root"] = str(Path(args.data_root).resolve())
    if args.assets:
        config.setdefault("initialization",{})["assets_dir"]=str(Path(args.assets).resolve())
    if args.beats_checkpoint:
        config.setdefault("initialization",{})["beats_checkpoint"]=str(Path(args.beats_checkpoint).resolve())
    if args.drive_dir:
        config.setdefault("logging",{})["drive_dir"]=args.drive_dir
    if args.waveform_cache:
        config["dataset"]["waveform_cache"] = str(Path(args.waveform_cache).resolve())
    if args.workers is not None:
        config["dataset"]["num_workers"] = args.workers
    registry_path = Path(args.models_config).resolve() if args.models_config else PACKAGE_ROOT/"config/models.yaml"
    config["_models_config"] = str(registry_path)
    config["source_configs"] = {"config_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                "models_sha256": hashlib.sha256(registry_path.read_bytes()).hexdigest()}
    device_name = args.device or config["training"].get("device", "auto")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device_name == "auto" else torch.device(device_name)
    if args.model == "all":
        directory = PACKAGE_ROOT/config["logging"].get("log_dir","logs")
        directory.mkdir(parents=True,exist_ok=True)
        suffix = "smoke" if args.smoke else "group_ce"
        snapshot = directory/f"{suffix}_config_snapshot.yaml"
        registry_snapshot = directory/f"{suffix}_models_snapshot.yaml"
        snapshot.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        registry_snapshot.write_text(registry_path.read_text(encoding="utf-8"), encoding="utf-8")
        failures, completed = [], []
        records = []
        for i, name in enumerate(CANONICAL_MODELS, 1):
            write_json(directory/f"{suffix}_queue.json", {"status": "running", "current": name, "index": i,
                       "total": len(CANONICAL_MODELS), "completed": completed, "failed": failures})
            command = [sys.executable, "-u", str(Path(__file__).resolve()), "train", "--model", name,
                       "--config", str(snapshot), "--models-config", str(registry_snapshot), "--device", str(device)]
            if args.smoke:
                command.append("--smoke")
            print(f"REVIEW-3 {i}/9: {name}; initialization={config.get('initialization',{}).get('mode','review3_scratch')}; raw mixture, CE-only", flush=True)
            child = subprocess.run(command, cwd=PACKAGE_ROOT)
            if child.returncode:
                failures.append(name)
            else:
                completed.append(name)
                checkpoint_dir=PACKAGE_ROOT/config["logging"]["checkpoint_dir"]
                if args.smoke:
                    checkpoint_dir=checkpoint_dir/"smoke"
                result=json.loads((checkpoint_dir/name/"results.json").read_text(encoding="utf-8"))
                records.append(result)
            write_json(directory/f"{suffix}_summary.json", records)
            lines=["# Review-3 CE-only theo cấu hình nhóm", "", "Implementation and pretrained receipt are recorded per checkpoint; raw mixture v2, validation only.", "",
                   "| Model | Params (M) | Trainable (M) | Epochs | Best | acc | precision | recall | macro_f1 | micro_f1 | mAP | macro_acc selection |",
                   "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
            by_name={v["model_name"]:v for v in records}
            for key in CANONICAL_MODELS:
                if key in by_name:
                    r=by_name[key]
                    v=r["validation"]
                    lines.append(f"| {key} | {r['parameters']/1e6:.3f} | {r['trainable']/1e6:.3f} | {r['epochs_ran']} | {r['best_epoch']} | "+" | ".join(f"{v[k]*100:.2f}%" for k in ("acc","precision","recall","macro_f1","micro_f1","mAP","macro_acc"))+" |")
                else:
                    lines.append(f"| {key} | {'failed' if key in failures else 'pending'} | — | — | — | — | — | — | — | — | — | — |")
            (directory/f"{suffix}_summary.md").write_text("\n".join(lines), encoding="utf-8")
        write_json(directory/f"{suffix}_queue.json", {"status": "complete" if not failures else "completed_with_failures",
                   "completed": completed, "failed": failures})
        if failures:
            raise SystemExit(1)
        return
    if args.model not in CANONICAL_MODELS:
        raise ValueError(f"This CE-only command selects exactly these nine models: {CANONICAL_MODELS}")
    result = train_single_model(args.model, config=config, device=device, smoke_test=args.smoke)
    print(f"COMPLETE {args.model}: best_epoch={result['best_epoch']} macro_acc={result['best_metric']:.6f}", flush=True)


def cmd_eval(args: argparse.Namespace) -> None:
    config_path = PACKAGE_ROOT / args.config
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = args.model

    ckpt_path = Path(args.checkpoint) if args.checkpoint else PACKAGE_ROOT / "checkpoints" / model_name / "best_model.pt"
    if not ckpt_path.is_file():
        print(f"Error: Checkpoint file not found at: {ckpt_path}")
        sys.exit(1)

    from scripts.evaluate_models import evaluate_model
    data_root = args.data_root or config.get("dataset", {}).get("data_root")
    num_classes = config.get("dataset", {}).get("num_classes", 36)
    noise_cache = getattr(args, "noise_cache", None) or config.get("noise_cache", {}).get("cache_file") or config.get("noise_cache", {}).get("cache_dir")
    apply_rms_norm = not getattr(args, "no_rms_norm", False)
    target_rms = getattr(args, "target_rms", 0.07925)

    evaluate_model(
        model_name=model_name,
        checkpoint_path=ckpt_path,
        data_root=data_root,
        noise_cache_path=noise_cache,
        split=args.split,
        device=device,
        num_classes=num_classes,
        apply_rms_norm=apply_rms_norm,
        target_rms=target_rms,
        max_samples=args.max_samples,
    )


def cmd_eval_masked(args: argparse.Namespace) -> None:
    """Tác vụ 2: Đánh giá mô hình sau khi qua mask FRCRN từ file .npy."""
    from scripts.evaluate_masked_models import evaluate_single_masked_model
    config_path = PACKAGE_ROOT / args.config
    cfg = {}
    if config_path.is_file():
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

    data_root = args.data_root or cfg.get("dataset", {}).get("data_root")
    num_classes = cfg.get("dataset", {}).get("num_classes", 36)
    noise_cache = args.noise_cache or cfg.get("noise_cache", {}).get("cache_file") or cfg.get("noise_cache", {}).get("cache_dir")
    apply_rms_norm = not args.no_rms_norm
    target_rms = args.target_rms

    device_str = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = PACKAGE_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"eval_masked_{args.model}_{timestamp}.log"
    logger = setup_logger(f"eval_masked_{timestamp}", log_file=log_file)

    logger.info("=" * 86)
    logger.info("TAC VU 2: DANH GIA MO HINH TREN NOISE FRCRN MASK (.npy)")
    logger.info("Timestamp: %s | Device: %s | Split: %s", timestamp, device, args.split)
    logger.info("Noise Cache File: %s", noise_cache)
    logger.info("RMS Normalization: %s (target_rms = %.5f)", apply_rms_norm, target_rms)
    logger.info("=" * 86)

    models_to_eval = list_available_models() if args.model.lower() == "all" else [args.model]
    results = []

    for name in models_to_eval:
        ckpt = args.checkpoint if (args.checkpoint and len(models_to_eval) == 1) else None
        res = evaluate_single_masked_model(
            model_name=name,
            checkpoint_path=ckpt,
            noise_cache_path=noise_cache,
            data_root=data_root,
            split=args.split,
            batch_size=args.batch_size,
            device=device,
            num_classes=num_classes,
            apply_rms_norm=apply_rms_norm,
            target_rms=target_rms,
            max_samples=args.max_samples,
            allow_untrained=args.allow_untrained,
            logger=logger,
        )
        if res is not None:
            results.append(res)

    if len(results) > 1:
        logger.info("=" * 86)
        logger.info("BANG TONG HOP SO SANH CAC MO HINH SAU KHI QUA MASK FRCRN")
        logger.info("=" * 86)
        logger.info("  | Model Name      | Params(M) | Accuracy | Precision |  Recall  | Macro-F1 | Micro-F1 |   mAP    |")
        logger.info("  |-----------------|-----------|----------|-----------|----------|----------|----------|----------|")
        for r in sorted(results, key=lambda x: x["metrics"]["macro_f1"], reverse=True):
            m = r["metrics"]
            logger.info(
                "  | %-15s | %9.2f | %7.2f%% | %9.4f | %8.4f | %8.4f | %8.4f | %8.4f |",
                r["model"],
                r["params_m"],
                m["acc"] * 100,
                m["precision"],
                m["recall"],
                m["macro_f1"],
                m["micro_f1"],
                m["mAP"],
            )
        logger.info("=" * 86)

    summary_path = log_dir / "eval_masked_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info("Ket qua danh gia chi tiet da ghi vao: %s", log_file)
    logger.info("File tong hop json: %s", summary_path)
    print(f"\n[OK] Hoan thanh danh gia {len(results)} mo hinh tren tap mask FRCRN (.npy).")
    print(f"     Chi tiet log: {log_file}")


def cmd_kd(args: argparse.Namespace) -> None:
    """Task 3: BEATs -> CED-Tiny Knowledge Distillation Benchmark."""
    from scripts.run_kd_benchmark import run_kd_benchmark
    run_kd_benchmark(args)


def cmd_moe(args: argparse.Namespace) -> None:
    """Task 4: Proposed Single-Backbone MoE (CED-SingleMoE-SplitFFN ~6.62M params) Benchmark."""
    from scripts.run_task4_moe import run_task4_benchmark
    run_task4_benchmark(
        epochs=args.epochs or 15,
        batch_size=args.batch_size or 32,
        smoke=args.smoke,
        no_kd=args.no_kd,
        device_name=None if args.device == "auto" else args.device,
    )


def cmd_kd(args: argparse.Namespace) -> None:
    """Task 3: BEATs -> CED-Tiny Knowledge Distillation Benchmark."""
    from scripts.run_kd_benchmark import run_kd_benchmark
    run_kd_benchmark(args)


def cmd_moe(args: argparse.Namespace) -> None:
    """Task 4: Proposed Single-Backbone MoE (CED-SingleMoE-SplitFFN ~6.62M params) Benchmark."""
    from scripts.run_task4_moe import run_task4_benchmark
    run_task4_benchmark(
        epochs=args.epochs or 15,
        batch_size=args.batch_size or 32,
        smoke=args.smoke,
        no_kd=args.no_kd,
        device_name=None if args.device == "auto" else args.device,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Review-3: 9-Model Acoustic Noise Classification System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    prepare=subparsers.add_parser("prepare",help="Download verified raw data and pretrained assets; no training")
    prepare.add_argument("--data-dir",default="data")
    prepare.add_argument("--skip-data",action="store_true")
    prepare.add_argument("--remove-archive",action="store_true")
    preflight=subparsers.add_parser("preflight",help="Verify genuine pretrained loads before using GPU time")
    preflight.add_argument("--model",default="all")
    preflight.add_argument("--config",default="config/pretrained.yaml")
    preflight.add_argument("--device",default="cpu")
    preflight.add_argument("--assets",default=None)
    preflight.add_argument("--beats-checkpoint",default=None)

    # list-models
    subparsers.add_parser("list-models", help="List all 9 models with parameters and families")

    # verify
    subparsers.add_parser("verify", help="Run sanity forward-pass verification on all 9 models")

    # train
    parser_train = subparsers.add_parser("train", help="Train a specific model or all models")
    parser_train.add_argument("--model", type=str, required=True, help="Model name (e.g. pann_cnn10, ced_tiny) or 'all'")
    parser_train.add_argument("--epochs", type=int, default=None, help="Number of epochs to train")
    parser_train.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser_train.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser_train.add_argument("--data-root", type=str, default=None, help="Root path of mixture dataset")
    parser_train.add_argument("--waveform-cache", type=str, default=None, help="Verified raw-mixture cache directory; never FRCRN cache")
    parser_train.add_argument("--models-config", type=str, default=None, help="Snapshot of model profiles")
    parser_train.add_argument("--assets",type=str,default=None)
    parser_train.add_argument("--beats-checkpoint",type=str,default=None)
    parser_train.add_argument("--drive-dir",type=str,default=None,help="Own Google Drive destination; cached login required")
    parser_train.add_argument("--workers", type=int, default=None)
    parser_train.add_argument("--smoke", action="store_true", help="Quick 1-epoch smoke test run")
    parser_train.add_argument("--device", type=str, default="auto", help="Compute device ('cuda', 'cpu', 'auto')")
    parser_train.add_argument("--config", type=str, default="config/pretrained.yaml", help="Path to config.yaml")

    # eval
    parser_eval = subparsers.add_parser("eval", help="Evaluate a trained checkpoint across SNR levels")
    parser_eval.add_argument("--model", type=str, required=True, help="Model name")
    parser_eval.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser_eval.add_argument("--noise-cache", type=str, default=None, help="Path to FRCRN-masked .npy cache or directory")
    parser_eval.add_argument("--no-rms-norm", action="store_true", help="Disable RMS normalization")
    parser_eval.add_argument("--target-rms", type=float, default=0.07925, help="Target RMS value (default: 0.07925)")
    parser_eval.add_argument("--split", type=str, default="validation", help="Dataset split (validation or test)")
    parser_eval.add_argument("--max-samples", type=int, default=None, help="Maximum samples to evaluate")
    parser_eval.add_argument("--data-root", type=str, default=None, help="Root path of mixture dataset")
    parser_eval.add_argument("--config", type=str, default="config/config.yaml", help="Path to config.yaml")

    # eval-masked (Task 2: Evaluate models on FRCRN-masked noise from .npy)
    parser_eval_masked = subparsers.add_parser("eval-masked", help="Task 2: Evaluate models on FRCRN-masked noise (.npy)")
    parser_eval_masked.add_argument("--model", type=str, default="all", help="Model name (e.g. pann_cnn10, ced_tiny) or 'all'")
    parser_eval_masked.add_argument("--noise-cache", type=str, default=None, help="Path to FRCRN-masked .npy file or folder")
    parser_eval_masked.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser_eval_masked.add_argument("--split", type=str, default="validation", help="Dataset split (validation, test, train)")
    parser_eval_masked.add_argument("--max-samples", type=int, default=None, help="Maximum samples to evaluate")
    parser_eval_masked.add_argument("--batch-size", type=int, default=32, help="Evaluation batch size")
    parser_eval_masked.add_argument("--no-rms-norm", action="store_true", help="Disable RMS normalization")
    parser_eval_masked.add_argument("--target-rms", type=float, default=0.07925, help="Target RMS (default: 0.07925)")
    parser_eval_masked.add_argument("--allow-untrained", action="store_true", help="Evaluate untrained model baseline if checkpoint missing")
    parser_eval_masked.add_argument("--device", type=str, default="auto", help="Compute device ('cuda', 'cpu', 'auto')")
    parser_eval_masked.add_argument("--data-root", type=str, default=None, help="Root path of mixture dataset for manifest")
    parser_eval_masked.add_argument("--config", type=str, default="config/config.yaml", help="Path to config.yaml")

    # kd (Task 3: BEATs -> CED-Tiny Knowledge Distillation)
    parser_kd = subparsers.add_parser("kd", help="Task 3: BEATs -> CED-Tiny KD (5 methods + BeatBiasPostLN comparison)")
    parser_kd.add_argument("--method", type=str, default="all",
                           choices=["all", "logit", "crd", "mgd", "logit+crd", "logit+mgd"],
                           help="Specific KD method or 'all' to run complete benchmark")
    parser_kd.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser_kd.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser_kd.add_argument("--smoke", action="store_true", help="Quick 1-epoch smoke test")
    parser_kd.add_argument("--noise-cache", type=str, default=None, help="Path to FRCRN-masked .npy cache or directory")
    parser_kd.add_argument("--data-root", type=str, default=None, help="Root path of mixture dataset")
    parser_kd.add_argument("--device", type=str, default="auto", help="Compute device ('cuda', 'cpu', 'auto')")
    parser_kd.add_argument("--config", type=str, default="config/config.yaml", help="Path to config.yaml")

    # moe (Task 4: Proposed CED-SingleMoE-SplitFFN ~6.62M params Benchmark)
    parser_moe = subparsers.add_parser("moe", help="Task 4: Benchmark proposed CED-SingleMoE-SplitFFN (~6.62M params)")
    parser_moe.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser_moe.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser_moe.add_argument("--smoke", action="store_true", help="Quick 1-epoch smoke test")
    parser_moe.add_argument("--no-kd", action="store_true", help="Disable BEATs Knowledge Distillation")
    parser_moe.add_argument("--device", type=str, default="auto", help="Compute device ('cuda', 'cpu', 'auto')")

    args = parser.parse_args()

    if args.command == "prepare":
        cmd_prepare(args)
    elif args.command == "preflight":
        cmd_preflight(args)
    elif args.command == "list-models":
        cmd_list_models(args)
    elif args.command == "verify":
        cmd_verify(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "eval":
        cmd_eval(args)
    elif args.command == "eval-masked":
        cmd_eval_masked(args)
    elif args.command == "kd":
        cmd_kd(args)
    elif args.command == "moe":
        cmd_moe(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
