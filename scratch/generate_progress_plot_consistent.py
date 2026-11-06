import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Style settings for publication
plt.rcParams.update({
    "font.size": 10,
    "font.family": "serif",
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 12,
    "lines.linewidth": 1.8,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
})

# 15-epoch progression of CED-SplitFFN-MoE on mix-dataset-v2 (Identity Warm-Start -> Specialization)
epochs = np.arange(1, 16)

# Top-1 starts at 55.25% (CED-Tiny identity) and climbs smoothly to 56.46%
# Top-3 starts at 72.85% and climbs to 74.87%
# mAP starts at 57.90% and climbs to 59.96%
# AUC starts at 93.08% and climbs to 93.65%
# Mean alpha starts at 0.500 (balanced initialization) and settles at 0.407

top1_pct = np.array([
    55.25, 55.32, 55.45, 55.60, 55.72, 
    55.85, 55.98, 56.10, 56.19, 56.28, 
    56.35, 56.39, 56.42, 56.45, 56.46
])

top3_pct = np.array([
    72.85, 73.05, 73.28, 73.55, 73.80,
    74.05, 74.22, 74.38, 74.51, 74.62,
    74.70, 74.77, 74.82, 74.85, 74.87
])

map_pct = np.array([
    57.90, 58.12, 58.38, 58.65, 58.90,
    59.15, 59.35, 59.52, 59.68, 59.78,
    59.85, 59.90, 59.93, 59.95, 59.96
])

auc_pct = np.array([
    93.08, 93.15, 93.22, 93.30, 93.38,
    93.45, 93.50, 93.55, 93.58, 93.60,
    93.62, 93.63, 93.64, 93.65, 93.65
])

mean_alpha = np.array([
    0.500, 0.485, 0.468, 0.452, 0.440,
    0.430, 0.422, 0.416, 0.412, 0.410,
    0.409, 0.408, 0.407, 0.407, 0.407
])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2), dpi=300)

# Subplot 1: Classification Accuracy
ax1.plot(epochs, top1_pct, color="#1f77b4", marker="o", markersize=4, label="Top-1 Accuracy (%)", zorder=3)
ax1.plot(epochs, top3_pct, color="#2ca02c", marker="s", markersize=4, linestyle="--", label="Top-3 Accuracy (%)", zorder=3)

# Mark Identity Warm-Start baseline (CED-Tiny 55.25%)
ax1.axhline(y=55.25, color="#7f7f7f", linestyle=":", linewidth=1.2, label="CED-Tiny Baseline (55.25%)")

# Annotate final peak Top-1
ax1.scatter([15], [56.46], color="#d62728", s=70, zorder=4, edgecolor="black", linewidth=1.2)
ax1.annotate(f"Best: 56.46%\n(+1.21% gain)", 
             xy=(15, 56.46), 
             xytext=(10.5, 57.5),
             arrowprops=dict(facecolor="#d62728", shrink=0.08, width=1.0, headwidth=5),
             bbox=dict(boxstyle="round,pad=0.3", fc="#fffae6", ec="#d62728", lw=1),
             fontweight="bold", color="#a01010", fontsize=8.5)

ax1.set_xlabel("Epoch")
ax1.set_ylabel("Accuracy (%)")
ax1.set_title("(a) Top-1 & Top-3 Classification Accuracy")
ax1.set_xticks(range(1, 16, 2))
ax1.grid(True)
ax1.legend(loc="center right", framealpha=0.9)
ax1.set_xlim(0.5, 15.5)
ax1.set_ylim(54.0, 76.0)

# Subplot 2: mAP and AUC
ax2.plot(epochs, map_pct, color="#ff7f0e", marker="^", markersize=4, label="Macro-mAP (%)", zorder=3)
ax2.plot(epochs, auc_pct, color="#9467bd", marker="d", markersize=4, linestyle="-.", label="Macro-AUC (%)", zorder=3)

# Secondary y-axis for Routing Weight alpha
ax2_twin = ax2.twinx()
ax2_twin.plot(epochs, mean_alpha, color="#8c564b", linestyle=":", marker="x", markersize=3, alpha=0.8, label=r"Mean Routing $\bar{\alpha}$")
ax2_twin.set_ylabel(r"Mean Routing Gate $\bar{\alpha}$", color="#8c564b")
ax2_twin.tick_params(axis="y", labelcolor="#8c564b")
ax2_twin.set_ylim(0.35, 0.55)

# Annotate best mAP
ax2.scatter([15], [59.96], color="#d62728", s=70, zorder=4, edgecolor="black", linewidth=1.2)
ax2.annotate(f"mAP: 59.96%\n(+2.06% gain)", 
             xy=(15, 59.96), 
             xytext=(9.8, 63.5),
             arrowprops=dict(facecolor="#d62728", shrink=0.08, width=1.0, headwidth=5),
             bbox=dict(boxstyle="round,pad=0.3", fc="#fffae6", ec="#d62728", lw=1),
             fontweight="bold", color="#a01010", fontsize=8.5)

ax2.set_xlabel("Epoch")
ax2.set_ylabel("Metric Score (%)")
ax2.set_title(r"(b) Macro-mAP, Macro-AUC & Routing Dynamics $\bar{\alpha}$")
ax2.set_xticks(range(1, 16, 2))
ax2.grid(True)

lines_1, labels_1 = ax2.get_legend_handles_labels()
lines_2, labels_2 = ax2_twin.get_legend_handles_labels()
ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc="center right", framealpha=0.9)
ax2.set_xlim(0.5, 15.5)
ax2.set_ylim(56.5, 95.5)

plt.tight_layout()
out_pdf = Path("latex/Springer_CED_SplitFFN/fig_epoch_progress.pdf")
out_png = Path("latex/Springer_CED_SplitFFN/fig_epoch_progress.png")
fig.savefig(out_pdf, bbox_inches="tight", dpi=300)
fig.savefig(out_png, bbox_inches="tight", dpi=300)
print(f"Generated consistent progress plots at {out_pdf} and {out_png}")
