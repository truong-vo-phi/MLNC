import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Load logs
log_file = Path("output/molab_sync/logs/summary_50epochs.json")
with open(log_file, "r", encoding="utf-8") as f:
    data = json.load(f)

history = data.get("history", [])
epochs = [h["epoch"] for h in history]
top1 = [h["top1"] if h["top1"] <= 1.0 else h["top1"] / 100.0 for h in history]
top1_pct = [t * 100.0 for t in top1]

metrics = [h.get("metrics", {}) for h in history]
top3_pct = [m.get("top3", 0.0) * 100.0 for m in metrics]
map_pct = [m.get("map", 0.0) * 100.0 for m in metrics]
auc_pct = [m.get("auc", 0.0) * 100.0 for m in metrics]
mean_alpha = [m.get("mean_alpha", 0.5) for m in metrics]

# Find best epoch based on Top-1
best_idx = int(np.argmax(top1_pct))
best_epoch = epochs[best_idx]
best_top1 = top1_pct[best_idx]
best_map = map_pct[best_idx]

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

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2), dpi=300)

# Subplot 1: Accuracy Progression
ax1.plot(epochs, top1_pct, color="#1f77b4", marker="o", markersize=4, label="Top-1 Accuracy (%)", zorder=3)
ax1.plot(epochs, top3_pct, color="#2ca02c", marker="s", markersize=4, linestyle="--", label="Top-3 Accuracy (%)", zorder=3)

# Annotate best Top-1
ax1.scatter([best_epoch], [best_top1], color="#d62728", s=80, zorder=4, edgecolor="black", linewidth=1.2)
ax1.annotate(f"Best: {best_top1:.2f}%\n(Epoch {best_epoch})", 
             xy=(best_epoch, best_top1), 
             xytext=(best_epoch - 6, best_top1 - 1.8),
             arrowprops=dict(facecolor="#d62728", shrink=0.08, width=1.0, headwidth=5),
             bbox=dict(boxstyle="round,pad=0.3", fc="#fffae6", ec="#d62728", lw=1),
             fontweight="bold", color="#a01010", fontsize=8.5)

# Annotate Early Stopping
ax1.axvline(x=28, color="#7f7f7f", linestyle=":", linewidth=1.5, label="Early Stop (Ep 28)")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Accuracy (%)")
ax1.set_title("(a) Top-1 & Top-3 Classification Accuracy")
ax1.grid(True)
ax1.legend(loc="lower right", framealpha=0.9)
ax1.set_xlim(0, 30)
ax1.set_ylim(38, 65)

# Subplot 2: mAP and AUC
ax2.plot(epochs, map_pct, color="#ff7f0e", marker="^", markersize=4, label="Macro-mAP (%)", zorder=3)
ax2.plot(epochs, auc_pct, color="#9467bd", marker="d", markersize=4, linestyle="-.", label="Macro-AUC (%)", zorder=3)

# Secondary y-axis for Routing Weight alpha
ax2_twin = ax2.twinx()
ax2_twin.plot(epochs, mean_alpha, color="#8c564b", linestyle=":", marker="x", markersize=3, alpha=0.7, label=r"Mean Routing $\bar{\alpha}$")
ax2_twin.set_ylabel(r"Routing Weight $\bar{\alpha}$", color="#8c564b")
ax2_twin.tick_params(axis="y", labelcolor="#8c564b")
ax2_twin.set_ylim(0.15, 0.40)

# Annotate best mAP
ax2.scatter([best_epoch], [best_map], color="#d62728", s=80, zorder=4, edgecolor="black", linewidth=1.2)
ax2.annotate(f"{best_map:.2f}%\n(Ep {best_epoch})", 
             xy=(best_epoch, best_map), 
             xytext=(best_epoch - 5, best_map - 0.7),
             arrowprops=dict(facecolor="#d62728", shrink=0.08, width=1.0, headwidth=5),
             bbox=dict(boxstyle="round,pad=0.3", fc="#fffae6", ec="#d62728", lw=1),
             fontweight="bold", color="#a01010", fontsize=8.5)

ax2.axvline(x=28, color="#7f7f7f", linestyle=":", linewidth=1.5)
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Metric Score (%)")
ax2.set_title("(b) Macro-mAP and Macro-AUC")
ax2.grid(True)

# Combine legends for ax2 and ax2_twin
lines_1, labels_1 = ax2.get_legend_handles_labels()
lines_2, labels_2 = ax2_twin.get_legend_handles_labels()
ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc="lower right", framealpha=0.9)
ax2.set_xlim(0, 30)
ax2.set_ylim(39, 90)

plt.tight_layout()

# Save vector PDF and high-res PNG into Springer LaTeX folder
out_dir = Path("latex/Springer_CED_SplitFFN")
fig_pdf = out_dir / "fig_epoch_progress.pdf"
fig_png = out_dir / "fig_epoch_progress.png"

plt.savefig(fig_pdf, format="pdf", bbox_inches="tight")
plt.savefig(fig_png, format="png", dpi=300, bbox_inches="tight")
plt.close()

print(f"Figures successfully generated:\n- {fig_pdf}\n- {fig_png}")
