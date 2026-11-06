import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

# Set style
plt.rcParams.update({
    "font.size": 9,
    "font.family": "serif",
    "mathtext.fontset": "cm",
})

fig, ax = plt.subplots(figsize=(12, 5.8), dpi=300)
ax.set_xlim(0, 100)
ax.set_ylim(0, 50)
ax.axis("off")

# Colors
c_input = "#E8F4F8"
c_stft = "#D4EDDA"
c_vit = "#D1ECF1"
c_router = "#FFF3CD"
c_shared = "#E2E3E5"
c_routed = "#F8D7DA"
c_out = "#E8DAEF"
c_border = "#495057"

def draw_box(x, y, w, h, title, subtitle="", bg_color="#ffffff", ec=c_border, lw=1.2, r=2):
    rect = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.2,rounding_size={r}",
        facecolor=bg_color, edgecolor=ec, linewidth=lw, zorder=2
    )
    ax.add_patch(rect)
    if subtitle:
        ax.text(x + w/2, y + h*0.62, title, ha="center", va="center", fontweight="bold", fontsize=9, zorder=3)
        ax.text(x + w/2, y + h*0.28, subtitle, ha="center", va="center", fontsize=7.5, color="#333333", zorder=3)
    else:
        ax.text(x + w/2, y + h*0.5, title, ha="center", va="center", fontweight="bold", fontsize=8.5, zorder=3)

def draw_arrow(x1, y1, x2, y2, text="", color=c_border, lw=1.2):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", color=color, lw=lw, shrinkA=2, shrinkB=2),
        zorder=4
    )
    if text:
        ax.text((x1+x2)/2, (y1+y2)/2 + 1.2, text, ha="center", va="bottom", fontsize=7.5, color=color, fontweight="bold", zorder=5)

# 1. Input audio
draw_box(1, 21, 11, 8, "Input Audio", "x in R^{64000} (4s @ 16kHz)", c_input)

# RMS Normalization
draw_box(15, 21, 10, 8, "RMS Norm", "sigma_0 = 0.07925", c_input)
draw_arrow(12, 25, 15, 25)

# Branch 1: STFT Physics features (Top branch)
draw_box(29, 36, 17, 10, "STFT Acoustic Extractor", "257x401 Power Spec P(f,t)\n6 Features f_spec in R^6", c_stft)
draw_arrow(20, 29, 29, 41, text="x_tilde")

# Branch 2: ViT Backbone (Bottom branch)
draw_box(29, 14, 17, 10, "Patch Embed & ViT", "64-Mel -> 96 Tokens + CLS\nBlocks 0-9 (2D GREP Bias)", c_vit)
draw_arrow(25, 25, 29, 19, text="Mel Spec")

# Self-Contained Router
draw_box(52, 31, 18, 15, "Spectral-Token Router", "Proj(h_9) in R^16 || f_spec in R^6\nMLP(22 -> 32 -> 2) + Softmax\nalpha in (0, 1)", c_router, ec="#856404", lw=1.5)

# Connect STFT to Router
draw_arrow(46, 41, 52, 41, text="f_spec in R^6")
# Connect Block 9 CLS to Router
draw_arrow(46, 19, 52, 35, text="h_9 = z_9[CLS]")

# Split-FFN Blocks 10-11
# Shared FFN
draw_box(74, 34, 12, 6, "Shared FFN", "W_shared (0.5x)", c_shared)
# Routed Low FFN
draw_box(74, 25, 12, 6, "Low-SNR FFN", "W_low (0.5(1-alpha))", c_routed, ec="#721c24")
# Routed High FFN
draw_box(74, 16, 12, 6, "High-SNR FFN", "W_high (0.5 alpha)", c_routed, ec="#721c24")

# Group box for Blocks 10-11
group_split = patches.FancyBboxPatch(
    (72, 13), 16, 30,
    boxstyle="round,pad=0.4,rounding_size=3",
    facecolor="none", edgecolor="#0c5460", linewidth=1.2, linestyle="--", zorder=1
)
ax.add_patch(group_split)
ax.text(80, 44.2, "Split-FFN (Blocks 10-11)", ha="center", va="center", fontsize=8, fontweight="bold", color="#0c5460")

# Router output alpha to FFNs
draw_arrow(70, 38.5, 74, 37, text="", color="#856404")
draw_arrow(70, 38.5, 74, 28, text="1-alpha", color="#856404")
draw_arrow(70, 38.5, 74, 19, text="alpha", color="#856404")

# Tokens into Blocks 10-11
draw_arrow(46, 17, 72, 17, text="Tokens z_9")

# Dual Heads & Output
draw_box(91, 20, 8, 10, "Dual Heads", "Soft blend:\n(1-a)H_low + a H_high", c_out)
draw_arrow(86, 25, 91, 25)

# Final classification
ax.text(95, 33, "Y in R^{36}\n(36 Classes)", ha="center", va="center", fontweight="bold", fontsize=8.5, color="#4A148C")
draw_arrow(95, 30, 95, 31.5)

plt.tight_layout()
out_pdf = Path("latex/Springer_CED_SplitFFN/fig_arch.pdf")
out_png = Path("latex/Springer_CED_SplitFFN/fig_arch.png")
fig.savefig(out_pdf, bbox_inches="tight", dpi=300)
fig.savefig(out_png, bbox_inches="tight", dpi=300)
print(f"Generated {out_pdf} and {out_png}")
