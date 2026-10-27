from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
MASKING_LABELS = {False: "no masking", True: "FRCRN masking"}
TEST_METRICS = ("top1_accuracy", "macro_f1", "mAP")
FRCRN_STAGES = (("pretrained", BLUE), ("finetuned", ORANGE))
FRCRN_TEST_METRICS = ("noise_si_snr", "loss")


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    ax.title.set_color(TEXT_PRIMARY)
    ax.xaxis.label.set_color(TEXT_SECONDARY)
    ax.yaxis.label.set_color(TEXT_SECONDARY)


def _figure(rows: int, cols: int, width: float, height: float) -> tuple[plt.Figure, Any]:
    fig, axes = plt.subplots(rows, cols, figsize=(width, height), squeeze=False, facecolor=SURFACE)
    return fig, axes


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return path


def _series(history: Sequence[dict[str, float]], key: str) -> tuple[list[int], list[float]]:
    return [record["epoch"] for record in history], [record[key] for record in history]


def _line(ax: plt.Axes, history, key: str, color: str, label: str) -> None:
    if not history or key not in history[0]:
        return
    epochs, values = _series(history, key)
    ax.plot(
        epochs,
        values,
        color=color,
        linewidth=2,
        marker="o",
        markersize=5,
        markeredgecolor=SURFACE,
        markeredgewidth=1.5,
        label=label,
    )


def _scenario_name(group: str, masking: bool) -> str:
    return f"{group}_{'mask' if masking else 'nomask'}"


def plot_map_curves(results: dict[str, dict[str, Any]], snr_groups: Sequence[str], path: Path) -> Path:
    fig, axes = _figure(2, len(snr_groups), 3.6 * len(snr_groups), 6.4)
    for row, masking in enumerate((False, True)):
        for col, group in enumerate(snr_groups):
            ax = axes[row][col]
            _style(ax)
            result = results.get(_scenario_name(group, masking))
            if result is not None:
                _line(ax, result["history"], "train_mAP", BLUE, "train")
                _line(ax, result["history"], "val_mAP", ORANGE, "validation")
            ax.set_title(f"{group} | {MASKING_LABELS[masking]}", fontsize=10, loc="left")
            ax.set_ylim(0, 1.02)
            if row == 1:
                ax.set_xlabel("epoch")
            if col == 0:
                ax.set_ylabel("mAP")
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", ncol=2, frameon=False, labelcolor=TEXT_SECONDARY)
    fig.suptitle("mAP per epoch: train vs validation", x=0.01, ha="left", color=TEXT_PRIMARY, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, path)


def plot_loss_curves(
    results: dict[str, dict[str, Any]], snr_groups: Sequence[str], path: Path, title: str = "BEATs training loss"
) -> Path:
    fig, axes = _figure(1, len(snr_groups), 3.6 * len(snr_groups), 3.4)
    for col, group in enumerate(snr_groups):
        ax = axes[0][col]
        _style(ax)
        for masking, color in ((False, BLUE), (True, ORANGE)):
            result = results.get(_scenario_name(group, masking))
            if result is not None:
                _line(ax, result["history"], "train_loss", color, MASKING_LABELS[masking])
        ax.set_title(group, fontsize=10, loc="left")
        ax.set_xlabel("epoch")
        if col == 0:
            ax.set_ylabel("train loss")
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", ncol=2, frameon=False, labelcolor=TEXT_SECONDARY)
    fig.suptitle(title, x=0.01, ha="left", color=TEXT_PRIMARY, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return _save(fig, path)


def _plot_grouped_test(
    results: dict[str, dict[str, Any]],
    snr_groups: Sequence[str],
    path: Path,
    metrics: Sequence[str],
    series: Sequence[tuple[Callable[[str], str], str, str]],
    column_width: float,
    ylim: tuple[float, float] | None = None,
) -> Path:
    fig, axes = _figure(1, len(metrics), column_width * len(metrics), 3.6)
    width = 0.36
    positions = range(len(snr_groups))
    for col, metric in enumerate(metrics):
        ax = axes[0][col]
        _style(ax)
        for offset, (scenario, color, label) in zip((-width / 2, width / 2), series, strict=True):
            values = [
                results[name]["test"][metric] if (name := scenario(group)) in results else 0.0 for group in snr_groups
            ]
            bars = ax.bar([p + offset for p in positions], values, width * 0.92, color=color, label=label)
            ax.bar_label(bars, fmt="%.2f", fontsize=8, color=TEXT_SECONDARY, padding=2)
        ax.set_xticks(list(positions), snr_groups)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.set_title(f"test {metric}", fontsize=10, loc="left")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=2, frameon=False, labelcolor=TEXT_SECONDARY)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return _save(fig, path)


def plot_test_metrics(results: dict[str, dict[str, Any]], snr_groups: Sequence[str], path: Path) -> Path:
    series = [
        (lambda group, masking=masking: _scenario_name(group, masking), color, MASKING_LABELS[masking])
        for masking, color in ((False, BLUE), (True, ORANGE))
    ]
    return _plot_grouped_test(results, snr_groups, path, TEST_METRICS, series, 4.2, ylim=(0, 1.05))


def plot_confusion_matrix(matrix: np.ndarray, class_names: Sequence[str], path: Path, title: str) -> Path:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] != len(class_names):
        raise ValueError("matrix must be square with one row per class name")
    row_totals = matrix.sum(axis=1, keepdims=True)
    rates = np.divide(matrix, row_totals, out=np.zeros(matrix.shape, dtype=np.float64), where=row_totals > 0)
    size = max(6.0, 0.32 * len(class_names) + 2.5)
    fig, axes = _figure(1, 1, size, size)
    ax = axes[0][0]
    image = ax.imshow(rates, cmap=LinearSegmentedColormap.from_list("confusion", [SURFACE, BLUE]), vmin=0.0, vmax=1.0)
    ticks = list(range(len(class_names)))
    ax.set_xticks(ticks, class_names, rotation=90)
    ax.set_yticks(ticks, class_names)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=7)
    ax.set_xlabel("predicted", color=TEXT_SECONDARY)
    ax.set_ylabel("true", color=TEXT_SECONDARY)
    ax.set_title(f"{title} | n={int(matrix.sum())}", fontsize=10, loc="left", color=TEXT_PRIMARY)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
    colorbar.set_label("share of true class", color=TEXT_SECONDARY)
    colorbar.ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
    fig.tight_layout()
    return _save(fig, path)


def plot_frcrn_curve(result: dict[str, Any], path: Path) -> Path:
    fig, axes = _figure(1, 1, 5.2, 3.6)
    ax = axes[0][0]
    _style(ax)
    _line(ax, result["history"], "val_noise_si_snr", ORANGE, "fine-tuned (validation)")
    ax.axhline(result["baseline_score"], color=BLUE, linewidth=2, label="pretrained baseline")
    ax.set_title("FRCRN noise estimate SI-SNR", fontsize=10, loc="left")
    ax.set_xlabel("epoch")
    ax.set_ylabel("SI-SNR (dB)")
    ax.legend(frameon=False, labelcolor=TEXT_SECONDARY, fontsize=9)
    fig.tight_layout()
    return _save(fig, path)


def plot_frcrn_test(results: dict[str, dict[str, Any]], snr_groups: Sequence[str], path: Path) -> Path:
    series = [(lambda group, stage=stage: f"{group}_{stage}", color, stage) for stage, color in FRCRN_STAGES]
    return _plot_grouped_test(results, snr_groups, path, FRCRN_TEST_METRICS, series, 4.6)
