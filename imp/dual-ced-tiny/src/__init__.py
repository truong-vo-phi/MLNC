"""
Dual-CED-Tiny: dual CED experts with GREP, Alpha Post-LN, and learned routing.
"""

from __future__ import annotations

from src.dataset import FrcrnCacheDataset, MixtureDataset, SyntheticMixtureDataset, get_dataloader, get_dataset
from src.frontend import LogMelFrontend, SpecAugment
from src.inference import overlap_crop_starts, overlap_logits
from src.kd_trainer import KDTrainer
from src.metrics import compute_metrics, compute_snr_breakdown
from src.models import CEDTiny, DualCEDTiny, build_model, list_available_models, load_ced_pretrained
from src.router import EnergySNRRouter, SelfContainedSpectralTokenRouter
from src.trainer import Trainer
from src.utils import count_parameters, load_checkpoint, save_checkpoint, set_seed, setup_logger

__all__ = [
    "DualCEDTiny",
    "CEDTiny",
    "load_ced_pretrained",
    "EnergySNRRouter",
    "SelfContainedSpectralTokenRouter",
    "LogMelFrontend",
    "SpecAugment",
    "FrcrnCacheDataset",
    "MixtureDataset",
    "SyntheticMixtureDataset",
    "get_dataset",
    "get_dataloader",
    "overlap_crop_starts",
    "overlap_logits",
    "compute_metrics",
    "compute_snr_breakdown",
    "Trainer",
    "KDTrainer",
    "build_model",
    "list_available_models",
    "setup_logger",
    "set_seed",
    "count_parameters",
    "save_checkpoint",
    "load_checkpoint",
]
