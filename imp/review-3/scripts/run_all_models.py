"""
Batch Training Runner:
Trains each of the 9 models sequentially, saves checkpoints and .log files,
and compiles a unified comparative benchmark table across all models and SNR levels.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import types
import hashlib
import inspect
import numpy as np
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from src.dataset import get_dataloader, get_dataset, MixtureDataset
from src.models import build_model, list_available_models
from src.trainer import Trainer
from src.utils import count_parameters, set_seed, setup_logger


CANONICAL_MODELS = ("pann_cnn10", "ced_tiny", "ced_mini", "ced_small", "efficient_mn10", "vit_s", "mobilevit", "mobilenet", "beats")


def resolve_model_config(model_name, config):
    effective = copy.deepcopy(config)
    with Path(effective.get("_models_config", PACKAGE_ROOT/"config/models.yaml")).open(encoding="utf-8") as stream:
        registry = yaml.safe_load(stream)
    profile_key = {"pann_cnn10": "panns_cnn10", "efficient_mn10": "efficientat_mn10"}.get(model_name, model_name)
    if model_name.startswith("ced_"):
        profile_key = "ced_meanpool"
    profile = registry.get("kd_profiles", {}).get(profile_key, {})
    training, front = effective["training"], effective["frontend"]
    if profile:
        for key in ("lr", "weight_decay", "warmup_ratio", "layer_decay", "drop_path"):
            if key in profile:
                training[key] = profile[key]
        training["head_lr"] = training["lr"]
        front.update(profile.get("frontend", {}))
    else:
        rates = registry["ce_only_benchmark"]["missing_profile_lr"][model_name]
        training["lr"], training["head_lr"] = rates["backbone"], rates["head"]
    if training.get("explicit_cli_lr") is not None:
        training["lr"] = training["head_lr"] = float(training["explicit_cli_lr"])
    masks = profile.get("specaugment", {})
    for key in ("freq_masks", "freq_width", "time_masks", "time_width"):
        if key in masks:
            front["specaug_"+key] = masks[key]
    effective["ce_only"] = {"input": "raw mixture v2", "teacher": False, "kd": False, "moe": False,
                            "stage1_epochs_applied": 0, "pretrained": effective.get("initialization",{}).get("mode")=="official_pretrained",
                            "implementation": "verified original pretrained loaders" if effective.get("initialization",{}).get("mode")=="official_pretrained" else "existing Review-3 constructors",
                            "profile": profile_key if profile else "common + user-selected previous baseline LRs",
                            "original_run_json_executed": False,
                            "architecture_policy": ("native verified pretrained backbone and pooling; replace only task head; KD architecture fields not applied"
                                if effective.get("initialization",{}).get("mode")=="official_pretrained" else
                                "retain Review-3 backbone, pooling and initialization; KD-profile architecture fields not applied"),
                            "selection": "mean Balanced Accuracy over six SNRs, present classes in each SNR",
                            "report_weights": "best_raw"}
    training["micro_batch_size"] = min(int(training["batch_size"]), {"beats": 8, "ced_small": 16, "vit_s": 16, "mobilevit": 16}.get(model_name, 32))
    if training["batch_size"] % training["micro_batch_size"]:
        raise ValueError("Effective batch must be divisible by micro-batch")
    training["accumulation"] = training["batch_size"]//training["micro_batch_size"]
    effective["logging"]["metric_for_best"] = "macro_acc"
    return effective


def configure_existing_model(model_name, config):
    if config.get("initialization",{}).get("mode")=="official_pretrained":
        return configure_pretrained_model(model_name,config)
    from src.frontend import LogMel, SpecAugment
    from timm.layers import DropPath
    front = config["frontend"]
    model = build_model(model_name, num_classes=config["dataset"]["num_classes"],
                        sample_rate=config["dataset"]["sample_rate"], n_mels=front["n_mels"], num_frames=front["num_frames"])
    names = ("sample_rate", "n_fft", "win_length", "hop_length", "n_mels", "f_min", "f_max", "num_frames", "log_eps", "norm_eps")
    model.frontend = LogMel(**{k: front[k] for k in names})
    model.spec_augment = SpecAugment(**{k: front["specaug_"+k] for k in ("freq_masks", "freq_width", "time_masks", "time_width")})
    drop_path = float(config["training"].get("drop_path", 0.))
    if model_name.startswith("ced_") and drop_path:
        for i, block in enumerate(model.blocks):
            probability = drop_path*i/max(len(model.blocks)-1, 1)
            # Apply the requested residual regularizer; retain existing block forward and weights.
            block.proj_drop = torch.nn.Sequential(block.proj_drop, DropPath(probability))
            block.mlp.add_module("configured_drop_path", DropPath(probability))
    return model


class _NativeSpecTransform(torch.nn.Module):
    """Frame count and existing SpecAugment only; no learnable adapter or backbone."""
    def __init__(self, config):
        super().__init__()
        from src.frontend import SpecAugment
        front=config["frontend"]
        self.frames=front["num_frames"]
        self.mask=SpecAugment(**{k:front["specaug_"+k] for k in ("freq_masks","freq_width","time_masks","time_width")})

    def forward(self,x):
        x=x.unsqueeze(1)
        if x.shape[-1]<self.frames:
            x=torch.nn.functional.pad(x,(0,self.frames-x.shape[-1]))
        else:
            x=x[...,:self.frames]
        return self.mask(x).squeeze(1)


def configure_pretrained_model(model_name,config):
    from timm.layers import DropPath
    path=PACKAGE_ROOT.parent/"mix_dataset_v2/noise_pipeline/ce9_models.py"
    if not path.is_file():
        raise FileNotFoundError("The verified pretrained library from this MLNC branch is required")
    name="_review3_verified_pretrained"
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,path)
        module=importlib.util.module_from_spec(spec)
        sys.modules[name]=module
        spec.loader.exec_module(module)
    module=sys.modules[name]
    mapping={"pann_cnn10":"panns_cnn10","efficient_mn10":"efficientat_mn10","vit_s":"vit_small","mobilevit":"mobilevit_s","mobilenet":"mobilenet_v3_large","beats":"beats12"}
    initialization=config["initialization"]
    assets=Path(initialization["assets_dir"])
    if not assets.is_absolute(): assets=PACKAGE_ROOT/assets
    beats=Path(initialization["beats_checkpoint"])
    if not beats.is_absolute(): beats=PACKAGE_ROOT/beats
    if model_name=="beats" and not beats.is_file(): raise FileNotFoundError(beats)
    paths_file=path.parents[1]/"benchmark_results/team_kd_moe_research_20261102/TEAM_DATA_PATHS.json"
    paths_file.parent.mkdir(parents=True,exist_ok=True)
    local=json.loads(paths_file.read_text(encoding="utf-8")) if paths_file.is_file() else {}
    local["beats_pretrained"]=str(beats.resolve())
    paths_file.write_text(json.dumps(local,indent=2),encoding="utf-8")
    model=module.build_model(mapping.get(model_name,model_name),assets,classes=config["dataset"]["num_classes"])
    model.group_spec=_NativeSpecTransform(config)
    if model_name.startswith("ced_"):
        model.model.front_end.register_forward_hook(lambda m,i,o:model.group_spec(o))
        for i,block in enumerate(model.model.blocks):
            probability=float(config["training"].get("drop_path",0))*i/max(len(model.model.blocks)-1,1)
            block.drop_path1=DropPath(probability) if probability else torch.nn.Identity()
            block.drop_path2=DropPath(probability) if probability else torch.nn.Identity()
    elif model_name=="pann_cnn10":
        model.model.bn0.register_forward_hook(lambda m,i,o:model.group_spec(o.squeeze(-1)).unsqueeze(-1))
    elif model_name=="efficient_mn10":
        model.frontend.register_forward_hook(lambda m,i,o:model.group_spec(o))
    elif model_name=="beats":
        original=model.encoder.preprocess
        def preprocess(source,**kwargs):
            return model.group_spec(original(source,**kwargs).transpose(1,2)).transpose(1,2)
        model.encoder.preprocess=preprocess
    else:
        def forward(native,wave):
            with torch.autocast(wave.device.type,enabled=False):
                x=native.db(native.mel(wave.float()))
                x=(x-x.mean((1,2),keepdim=True))/(x.std((1,2),keepdim=True)+1e-5)
                x=(native.group_spec(x).clamp(-3,3)+3)/6
                x=torch.nn.functional.interpolate(x[:,None],native.size,mode="bilinear",align_corners=False).expand(-1,3,-1,-1)
                x=(x-native.mean)/native.std
            return native.model(x)
        model.forward=types.MethodType(forward,model)
    model.receipt["verified_pretrained_loader"]=True
    model.receipt["adapter_sha256"]=hashlib.sha256(path.read_bytes()).hexdigest()
    model.receipt["frontend_policy"]="native pretrained frontend; only group frame cropping and SpecAugment added"
    return model


def _stratified_indices(dataset):
    chosen = {}
    for i,row in enumerate(dataset.rows):
        key=(int(np.argmax(json.loads(row["multi_hot_36"]))),int(float(row["target_snr_db"])))
        chosen.setdefault(key,i)
    return sorted(chosen.values())


def train_single_model(model_name, config, epochs=None, batch_size=None, device=None, smoke_test=False):
    from torch.utils.data import DataLoader, RandomSampler, Subset
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = resolve_model_config(model_name, config)
    t, d = config["training"], config["dataset"]
    if epochs:
        t["epochs"] = epochs
    if batch_size:
        t["batch_size"] = batch_size
        t["micro_batch_size"] = min(t["micro_batch_size"], batch_size)
        t["accumulation"] = batch_size//t["micro_batch_size"]
    if smoke_test:
        t["epochs"], t["micro_batch_size"], t["accumulation"] = 1,4,16
        d["num_workers"] = 0
    torch.set_num_threads(4)
    set_seed(t["seed"])
    log_dir = PACKAGE_ROOT/config["logging"].get("log_dir", "logs")
    log_file = log_dir/(model_name+("_smoke" if smoke_test else "_group_ce")+".log")
    logger = setup_logger(name="train_"+model_name,log_file=log_file)
    logger.info("Initialization: %s",config.get("initialization",{}).get("mode","review3_scratch"))
    kwargs = dict(root=d["data_root"], sample_rate=d["sample_rate"],duration=d["duration"],num_classes=d["num_classes"],waveform_cache=d.get("waveform_cache"))
    train_ds=MixtureDataset(split="train",**kwargs)
    val_ds=MixtureDataset(split="eval",**kwargs)
    if len(train_ds)!=30240 or len(val_ds)!=3064:
        raise ValueError(f"Expected clean v2 30240/3064; got {len(train_ds)}/{len(val_ds)}")
    train_ids=[r["sample_id"] for r in train_ds.rows]
    val_ids=[r["sample_id"] for r in val_ds.rows]
    if set(train_ids)&set(val_ids):
        raise ValueError("Train/validation IDs overlap")
    config["data_audit"]={"train_samples":len(train_ds),"validation_samples":len(val_ds),"test_opened":False,
                          "train_manifest_sha256":hashlib.sha256((Path(d["data_root"])/"train/manifest.csv").read_bytes()).hexdigest(),
                          "eval_manifest_sha256":hashlib.sha256((Path(d["data_root"])/"eval/manifest.csv").read_bytes()).hexdigest(),
                          "labels_sha256":hashlib.sha256((Path(d["data_root"])/"labels.txt").read_bytes()).hexdigest(),
                          "train_ids_sha256":hashlib.sha256("\n".join(train_ids).encode()).hexdigest(),
                          "eval_ids_sha256":hashlib.sha256("\n".join(val_ids).encode()).hexdigest()}
    config["dataset"]["class_names"]=train_ds.class_names
    if smoke_test:
        train_ds=Subset(train_ds,_stratified_indices(train_ds))
        val_ds=Subset(val_ds,_stratified_indices(val_ds))
    order=torch.Generator().manual_seed(t["seed"])
    workers=int(d.get("num_workers",4))
    loader_kwargs=dict(num_workers=workers,pin_memory=device.type=="cuda",persistent_workers=workers>0)
    train_loader=DataLoader(train_ds,batch_size=t["micro_batch_size"],sampler=RandomSampler(train_ds,generator=order),**loader_kwargs)
    val_loader=DataLoader(val_ds,batch_size=min(8,t["eval_batch_size"]) if smoke_test else t["eval_batch_size"],shuffle=False,**loader_kwargs)
    model=configure_existing_model(model_name,config)
    source=Path(inspect.getfile(type(model)))
    config["model_source"]={"class":type(model).__module__+"."+type(model).__name__,"file":os.path.relpath(source,PACKAGE_ROOT),
                            "sha256":hashlib.sha256(source.read_bytes()).hexdigest(),"external_pretrained_loaded":bool(config["ce_only"]["pretrained"])}
    if config["ce_only"]["pretrained"]:
        config["pretrained_receipt"]=model.receipt
    directory=PACKAGE_ROOT/config["logging"].get("checkpoint_dir","checkpoints")
    if smoke_test:
        directory=directory/"smoke"
    trainer=Trainer(model,train_loader,val_loader,test_loader=None,config=config,device=device,logger=logger,checkpoint_dir=directory,model_name=model_name)
    result=trainer.fit()
    result["log_file"]=str(log_file)
    result["params_m"]=count_parameters(model)[0]/1e6
    return result


def main():
    parser = argparse.ArgumentParser(description="Train all Review-3 models")
    parser.add_argument("--models", nargs="+", default=["all"], help="Model names or 'all'")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--smoke", action="store_true", help="Run quick 1-epoch smoke test on all models")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file")
    args = parser.parse_args()

    cfg_path = PACKAGE_ROOT / args.config
    with open(cfg_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_models = list_available_models()
    selected_models = all_models if "all" in args.models else args.models

    print("=" * 86)
    print(f"REVIEW-3: RUNNING BATCH TRAINING FOR {len(selected_models)} MODELS (Smoke={args.smoke})")
    print(f"Device: {device} | Models: {', '.join(selected_models)}")
    print("=" * 86)

    summary_results = []
    for model_name in selected_models:
        print(f"\n>>> Starting model: {model_name}...")
        start_t = time.time()
        try:
            res = train_single_model(
                model_name,
                config=config,
                epochs=args.epochs,
                batch_size=args.batch_size,
                device=device,
                smoke_test=args.smoke,
            )
            elapsed = time.time() - start_t
            res["elapsed_s"] = elapsed
            summary_results.append(res)
            print(f">>> Finished {model_name} in {elapsed:.1f}s | Best Metric: {res['best_metric']:.4f}")
        except Exception as e:
            print(f">>> FAILED {model_name}: {e}")
            import traceback
            traceback.print_exc()

    # Save summary report
    summary_path = PACKAGE_ROOT / "logs" / "benchmark_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_results, f, indent=2)
    print(f"\nBatch training completed! Benchmark summary saved to {summary_path}")


if __name__ == "__main__":
    main()

