import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Publication-quality typography and styling
plt.rcParams.update({
    "font.size": 10,
    "font.family": "serif",
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "figure.titlesize": 12,
    "lines.linewidth": 1.8,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
})

# Load raw training history
r1 = json.load(open("experiments/results/run1_with_rms_summary.json"))
r3 = json.load(open("experiments/results/run3_split_10_15_summary.json"))

# Epoch ranges
r1_low_epochs = np.arange(1, len(r1["low"]["history"]) + 1)
r1_high_epochs = np.arange(1, len(r1["high"]["history"]) + 1)
r3_low_epochs = np.arange(1, len(r3["low"]["history"]) + 1)
r3_high_epochs = np.arange(1, len(r3["high"]["history"]) + 1)

# Accuracy arrays (%)
r1_low_acc = np.array([x["val_top1_accuracy"] * 100 for x in r1["low"]["history"]])
r1_high_acc = np.array([x["val_top1_accuracy"] * 100 for x in r1["high"]["history"]])
r3_low_acc = np.array([x["val_top1_accuracy"] * 100 for x in r3["low"]["history"]])
r3_high_acc = np.array([x["val_top1_accuracy"] * 100 for x in r3["high"]["history"]])

# mAP arrays (%)
r1_low_map = np.array([x["val_mAP"] * 100 for x in r1["low"]["history"]])
r1_high_map = np.array([x["val_mAP"] * 100 for x in r1["high"]["history"]])
r3_low_map = np.array([x["val_mAP"] * 100 for x in r3["low"]["history"]])
r3_high_map = np.array([x["val_mAP"] * 100 for x in r3["high"]["history"]])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3), dpi=300)

# ----------------- Subplot 1: Top-1 Accuracy -----------------
ax1.plot(r3_low_epochs, r3_low_acc, color="#1f77b4", marker="o", markersize=4, 
         label=r"Run 3 $S_{\mathrm{low}}$ ($\leq 10$ dB Domain)", zorder=4)
ax1.plot(r1_low_epochs, r1_low_acc, color="#2ca02c", marker="s", markersize=4, linestyle="--", 
         label=r"Run 1 $S_{\mathrm{low}}$ (All SNRs Domain)", zorder=3)
ax1.plot(r3_high_epochs, r3_high_acc, color="#d62728", marker="^", markersize=4, linestyle="-.", 
         label=r"Run 3 $S_{\mathrm{high}}$ ($\geq 15$ dB Domain)", zorder=3)
ax1.plot(r1_high_epochs, r1_high_acc, color="#ff7f0e", marker="d", markersize=4, linestyle=":", 
         label=r"Run 1 $S_{\mathrm{high}}$ (20 dB Domain)", zorder=2)

# Mark peak points
ax1.scatter([9], [r3_low_acc[8]], color="#1f77b4", s=65, zorder=5, edgecolor="black")
ax1.annotate(f"Best: {r3_low_acc[8]:.2f}%", xy=(9, r3_low_acc[8]), xytext=(9.2, 62.6),
             fontsize=8, fontweight="bold", color="#0f4c81")

ax1.scatter([1], [r3_high_acc[0]], color="#d62728", s=65, zorder=5, edgecolor="black")
ax1.annotate(f"Init: {r3_high_acc[0]:.2f}%", xy=(1, r3_high_acc[0]), xytext=(1.4, 51.8),
             fontsize=8, fontweight="bold", color="#a01010")

ax1.set_xlabel("Training Epoch")
ax1.set_ylabel("Validation Top-1 Accuracy (%)")
ax1.set_title("(a) Domain Validation Top-1 Accuracy Dynamics")
ax1.set_xticks(range(1, 16, 2))
ax1.set_xlim(0.5, 15.5)
ax1.set_ylim(44.0, 64.5)
ax1.grid(True)
ax1.legend(loc="lower right", framealpha=0.92)

# ----------------- Subplot 2: Macro mAP -----------------
ax2.plot(r3_low_epochs, r3_low_map, color="#1f77b4", marker="o", markersize=4, 
         label=r"Run 3 $S_{\mathrm{low}}$ ($\leq 10$ dB Domain)", zorder=4)
ax2.plot(r1_low_epochs, r1_low_map, color="#2ca02c", marker="s", markersize=4, linestyle="--", 
         label=r"Run 1 $S_{\mathrm{low}}$ (All SNRs Domain)", zorder=3)
ax2.plot(r3_high_epochs, r3_high_map, color="#d62728", marker="^", markersize=4, linestyle="-.", 
         label=r"Run 3 $S_{\mathrm{high}}$ ($\geq 15$ dB Domain)", zorder=3)
ax2.plot(r1_high_epochs, r1_high_map, color="#ff7f0e", marker="d", markersize=4, linestyle=":", 
         label=r"Run 1 $S_{\mathrm{high}}$ (20 dB Domain)", zorder=2)

ax2.scatter([9], [r3_low_map[8]], color="#1f77b4", s=65, zorder=5, edgecolor="black")
ax2.annotate(f"Best: {r3_low_map[8]:.2f}%", xy=(9, r3_low_map[8]), xytext=(9.2, 68.6),
             fontsize=8, fontweight="bold", color="#0f4c81")

ax2.scatter([1], [r3_high_map[0]], color="#d62728", s=65, zorder=5, edgecolor="black")
ax2.annotate(f"Init: {r3_high_map[0]:.2f}%", xy=(1, r3_high_map[0]), xytext=(1.4, 55.4),
             fontsize=8, fontweight="bold", color="#a01010")

ax2.set_xlabel("Training Epoch")
ax2.set_ylabel("Validation Macro-mAP (%)")
ax2.set_title("(b) Domain Validation Macro-mAP Dynamics")
ax2.set_xticks(range(1, 16, 2))
ax2.set_xlim(0.5, 15.5)
ax2.set_ylim(47.0, 70.5)
ax2.grid(True)
ax2.legend(loc="lower right", framealpha=0.92)

plt.tight_layout()

# Save outputs
out_pdf = Path("latex/Springer_Dual_CED_tiny/fig_training_curves.pdf")
out_png = Path("latex/Springer_Dual_CED_tiny/fig_training_curves.png")
plt.savefig(out_pdf, bbox_inches="tight")
plt.savefig(out_png, bbox_inches="tight")
print(f"Generated clean training curves: {out_pdf} and {out_png}")
