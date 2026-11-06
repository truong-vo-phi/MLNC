import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT_DIR = r"D:\PROJECT\MLNC\latex\Springer_Dual_CED_tiny"
os.makedirs(OUT_DIR, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 9, "mathtext.fontset": "cm", "figure.dpi": 300})


def box(ax, center, width, height, title, body="", dashed=False, fontsize=8.5):
    x, y = center
    rect = FancyBboxPatch(
        (x - width / 2, y - height / 2), width, height,
        boxstyle="round,pad=0.22,rounding_size=0.8",
        facecolor="white", edgecolor="black", linewidth=1.0,
        linestyle="--" if dashed else "-",
    )
    ax.add_patch(rect)
    label = title if not body else rf"\textbf{{{title}}}\n{body}"
    # Use ordinary bold title text so labels render consistently across Matplotlib builds.
    if body:
        ax.text(x, y + height * 0.18, title, ha="center", va="center", fontweight="bold", fontsize=fontsize)
        ax.text(x, y - height * 0.14, body, ha="center", va="center", fontsize=fontsize - 0.5)
    else:
        ax.text(x, y, title, ha="center", va="center", fontsize=fontsize)
    return rect


def arrow(ax, start, end, label=None, label_offset=(0, 0), dashed=False):
    p = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=10,
                        linewidth=1.0, color="black", linestyle="--" if dashed else "-")
    ax.add_patch(p)
    if label:
        mx = (start[0] + end[0]) / 2 + label_offset[0]
        my = (start[1] + end[1]) / 2 + label_offset[1]
        ax.text(mx, my, label, ha="center", va="center", fontsize=8,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.2})


# Figure 1: pipeline, with a separate routing-control path and offline training note.
fig, ax = plt.subplots(figsize=(13, 5.2))
ax.set_xlim(0, 100)
ax.set_ylim(0, 62)
ax.axis("off")
box(ax, (50, 53), 70, 10, "OFFLINE TRAINING ONLY",
    "Frozen BEATs Fold0 + EMA teachers form averaged logits; both students use CE + KL (T_KD = 4).",
    dashed=True, fontsize=8.2)

pipeline = [
    ((9, 31), 15, "Mixture waveform", "4 s; 16 kHz"),
    ((28, 31), 16, "RMS normalization", "target RMS = 0.07925"),
    ((48, 31), 17, "Log-Mel frontend", "64 x 398; 97 tokens"),
    ((70, 31), 21, "SNR-selected student", "Low: 5.42M (<=10 dB)\nHigh: 5.42M (>=15 dB)"),
    ((92, 31), 15, "Temporal TTA", "three shifts; mean logits\n36 noise classes"),
]
for c, w, title, body in pipeline:
    box(ax, c, w, 13, title, body, fontsize=8.0)
for left, right in zip(pipeline[:-1], pipeline[1:]):
    x1 = left[0][0] + left[1] / 2
    x2 = right[0][0] - right[1] / 2
    arrow(ax, (x1, 31), (x2, 31))

box(ax, (45, 10), 24, 12, "Routing input",
    "metadata SNR r or\ncalibrated proxy r-hat", fontsize=8.2)
box(ax, (70, 10), 20, 12, "Threshold gate",
    "one student selected", fontsize=8.2)
arrow(ax, (57, 10), (60, 10))
arrow(ax, (70, 16), (70, 24), label="select branch", label_offset=(5, 0))
ax.text(70, 1.5, "EDGE INFERENCE: students only (10.84M total)", ha="center", va="center",
        fontsize=8.2, fontweight="bold")
fig.savefig(os.path.join(OUT_DIR, "fig_dual_ced_pipeline.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT_DIR, "fig_dual_ced_pipeline.png"), bbox_inches="tight", dpi=300)
plt.close(fig)


# Figure 2: router estimation and parallel expert selection, without implied serial fusion.
fig, (a, b) = plt.subplots(2, 1, figsize=(12, 7.2), gridspec_kw={"height_ratios": [1, 1.05]})
for axp in (a, b):
    axp.axis("off")
    axp.set_xlim(0, 100)
    axp.set_ylim(0, 42)

a.text(50, 38, "(a) Acoustic proxy estimation and calibrated dispatch", ha="center", va="center",
       fontsize=10, fontweight="bold")
a_nodes = [
    (8, 18, 15, "Mixture waveform", "100-ms frames"),
    (27, 18, 17, "Frame power", r"$P_k = M^{-1}\sum_m x_{k,m}^2$"),
    (47, 18, 17, "Rank powers", "select top/bottom 20%"),
    (67, 18, 19, "Contrast proxy", r"$\hat r=10\log_{10}(P_{top}/(P_{bottom}+\epsilon))$"),
    (88, 18, 17, "Threshold", r"$r$ vs. $\tau$ or $\hat r$ vs. $\hat\tau$"),
]
for x, y, w, title, body in a_nodes:
    box(a, (x, y), w, 11, title, body, fontsize=7.7)
for first, second in zip(a_nodes[:-1], a_nodes[1:]):
    arrow(a, (first[0] + first[2] / 2, 18), (second[0] - second[2] / 2, 18))
box(a, (78, 3.2), 26, 7, "Alternative input: metadata SNR r", fontsize=7.8)
arrow(a, (78, 6.7), (88, 12.4))

b.text(50, 39, "(b) Parallel expert specialization (one active expert per sample)", ha="center", va="center",
       fontsize=10, fontweight="bold")
box(b, (13, 20), 22, 11, "Threshold decision", "one branch per sample", fontsize=8.2)
box(b, (49, 29), 31, 13, "Low-SNR student S_low", "trained <=10 dB; selected below threshold", fontsize=8.2)
box(b, (49, 11), 31, 13, "High-SNR student S_high", "trained >=15 dB; selected at/above threshold", fontsize=8.2)
box(b, (88, 20), 20, 11, "One prediction", "36-class logits", fontsize=8.2)
arrow(b, (24, 24), (33, 29))
arrow(b, (24, 16), (33, 11))
arrow(b, (64.5, 29), (78, 23))
arrow(b, (64.5, 11), (78, 17))
b.text(50, 2, r"Threshold varies by run: $\tau=12.5$ dB (Run 3); $\tau=18.0$ dB (Run 1).",
       ha="center", va="center", fontsize=8)
fig.tight_layout(pad=1.0)
fig.savefig(os.path.join(OUT_DIR, "fig_router_and_calibration.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT_DIR, "fig_router_and_calibration.png"), bbox_inches="tight", dpi=300)
plt.close(fig)
print("Updated monochrome architecture figures in", OUT_DIR)

