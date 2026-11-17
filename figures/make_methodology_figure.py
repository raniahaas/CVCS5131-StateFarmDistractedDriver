"""Draws the methodology figure for the paper (model + evaluation protocol).

Usage:
    python figures/make_methodology_figure.py [output_dir]

Writes ``methodology.pdf`` (vector, for LaTeX) and ``methodology.png`` (preview, 300 dpi)
to ``output_dir`` (default: the folder containing this script). Only needs matplotlib.

Colour code (matches the ablation ladder in novelApproach.ipynb):
    blue   = baseline fine-tuned CNN (A)      orange = + semantic flip (B)
    green  = + driver adversary / GRL (C)     purple = + pose fusion (D, full model)
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

# Fill / edge colours per ablation stage (colour-blind friendly).
BASE = ("#DCE8F5", "#3B5B92")
FLIP = ("#FBE4CC", "#C26A00")
GRL = ("#D8EDD6", "#2E7D32")
POSE = ("#E8DDF2", "#6A3D9A")
LOSS = ("#F0F0F0", "#555555")
PLAIN = ("#FFFFFF", "#333333")

fig, ax = plt.subplots(figsize=(7.2, 4.9))
ax.set_xlim(0, 15.8)
ax.set_ylim(-3.5, 7.6)
ax.axis("off")


def box(x0, y0, x1, y1, text, colors, fs=6.3, rot=0, ls="-", weight="normal", style="round,pad=0.02,rounding_size=0.12"):
    """Draws a rounded, labelled box. Corners are (x0, y0) bottom-left to (x1, y1) top-right."""
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle=style,
                                fc=colors[0], ec=colors[1], lw=1.0, ls=ls))
    ax.text((x0 + x1) / 2, (y0 + y1) / 2, text, ha="center", va="center", fontsize=fs,
            rotation=rot, linespacing=1.15, fontweight=weight)


def arrow(points, color="#333333", ls="-", lw=1.0):
    """Draws a polyline through `points` with an arrow head on the last segment."""
    for (xa, ya), (xb, yb) in zip(points[:-2], points[1:-1]):
        ax.plot([xa, xb], [ya, yb], color=color, lw=lw, ls=ls, solid_capstyle="butt")
    ax.annotate("", xy=points[-1], xytext=points[-2],
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, ls=ls, shrinkA=0, shrinkB=0,
                                mutation_scale=7))


def label(x, y, text, fs=5.6, color="#333333", ha="center", style="normal"):
    ax.text(x, y, text, fontsize=fs, color=color, ha=ha, va="center", style=style, linespacing=1.1)


# ------------------------------------------------------------------ panel (a): model
label(0.1, 7.3, "(a) Model and training", fs=7.5, ha="left", color="#000000")
ax.texts[-1].set_fontweight("bold")

# Input frame and the frozen pose detector (offline, cached).
box(0.2, 3.4, 1.9, 4.6, "Dashcam\nframe\n224×224", PLAIN)
box(2.4, 1.0, 4.6, 2.2, "Keypoint R-CNN\n(frozen, cached)", POSE)
arrow([(1.05, 3.4), (1.05, 1.6), (2.4, 1.6)])
label(3.5, 0.6, "9 upper-body joints\n+ detected flag", fs=5.3)

# Semantic flip: one tall box that acts on both streams (image and keypoints) at train time.
box(5.2, 0.8, 6.5, 4.8, "Semantic flip (train only)\nimage + keypoints + label\nc1↔c3, c2↔c4", FLIP, rot=90, fs=5.8)
arrow([(1.9, 4.0), (5.2, 4.0)])
arrow([(4.6, 1.6), (5.2, 1.6)])

# Visual stream.
box(7.1, 3.4, 9.3, 4.6, "ResNet-18\n(from scratch)", BASE)
box(9.8, 3.55, 10.9, 4.45, "f$_{vis}$\n512-d", BASE)
arrow([(6.5, 4.0), (7.1, 4.0)])
arrow([(9.3, 4.0), (9.8, 4.0)])

# Pose stream.
box(7.1, 1.0, 9.3, 2.2, "Pose MLP\n19 → 64 → 64", POSE)
box(9.8, 1.15, 10.9, 2.05, "f$_{pose}$\n64-d", POSE)
arrow([(6.5, 1.6), (7.1, 1.6)])
arrow([(9.3, 1.6), (9.8, 1.6)])

# Fusion and action classifier.
box(11.4, 2.55, 12.4, 3.45, "concat", BASE, fs=5.6)
arrow([(10.9, 4.0), (11.9, 4.0), (11.9, 3.45)])
arrow([(10.9, 1.6), (11.9, 1.6), (11.9, 2.55)])
box(12.9, 2.5, 14.1, 3.5, "Classifier\n576→10", BASE, fs=5.6)
arrow([(12.4, 3.0), (12.9, 3.0)])
box(14.5, 2.55, 15.7, 3.45, "$L_{class}$\ncross-\nentropy", LOSS, fs=5.4)
arrow([(14.1, 3.0), (14.5, 3.0)])

# Driver adversary (training only).
box(9.5, 5.7, 11.2, 6.9, "Gradient\nreversal\n(×−λ)", GRL, fs=5.8)
box(11.9, 5.7, 13.7, 6.9, "Driver head\n512→256→K", GRL, fs=5.6)
box(14.1, 5.7, 15.7, 6.9, "$L_{driver}$\ncross-\nentropy", LOSS, fs=5.4)
arrow([(10.35, 4.45), (10.35, 5.7)])
arrow([(11.2, 6.3), (11.9, 6.3)])
arrow([(13.7, 6.3), (14.1, 6.3)])
# Reversed gradient flowing back into the backbone (dashed red).
arrow([(9.5, 6.3), (8.2, 6.3), (8.2, 4.6)], color="#B71C1C", ls=(0, (3, 2)))
label(7.4, 6.75, "reversed gradient:\nbackbone hides driver identity", fs=5.2, color="#B71C1C")
label(12.6, 5.35, "training only (removed at inference)", fs=5.2, style="italic")

# Legend for the ablation ladder.
lx, ly = 0.2, 6.35
for i, (name, col) in enumerate([("A  baseline", BASE), ("B  + flip", FLIP), ("C  + driver adversary", GRL), ("D  + pose fusion (full)", POSE)]):
    ax.add_patch(Rectangle((lx, ly - 0.38 * i - 0.14), 0.3, 0.28, fc=col[0], ec=col[1], lw=0.9))
    label(lx + 0.45, ly - 0.38 * i, name, fs=5.8, ha="left")
label(lx, ly + 0.5, "Ablation ladder", fs=6.0, ha="left", color="#000000")
ax.texts[-1].set_fontweight("bold")

# ------------------------------------------------------------------ panel (b): evaluation
label(0.1, -0.55, "(b) Evaluation protocol: driver-grouped K-fold (test on unseen drivers)", fs=7.5, ha="left", color="#000000")
ax.texts[-1].set_fontweight("bold")

n_drivers, held_out = 26, {3, 11, 19}   # illustrative fold: 3 of 26 drivers held out
sq, gap, x0, y0 = 0.40, 0.14, 0.2, -1.45
for i in range(n_drivers):
    held = i in held_out
    ax.add_patch(Rectangle((x0 + i * (sq + gap), y0), sq, sq, fc="#F4A261" if held else "#A9C4E4",
                           ec="#8A4B08" if held else "#3B5B92", lw=0.8))
label(x0, y0 - 0.45, "Each square is a driver: blue = training drivers, orange = held-out validation drivers.\n"
      "All frames of a driver stay in one fold, so no driver appears in both.", fs=5.4, ha="left")

box(0.2, -3.35, 5.0, -2.75, "Metrics: accuracy, macro-F1, log-loss", PLAIN, fs=5.6)
box(5.4, -3.35, 10.2, -2.75, "Driver-ID probe on f$_{vis}$", PLAIN, fs=5.6)
box(10.6, -3.35, 15.7, -2.75, "Grad-CAM + leakage study", PLAIN, fs=5.6)

out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
out_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(out_dir / "methodology.pdf", bbox_inches="tight")
fig.savefig(out_dir / "methodology.png", dpi=300, bbox_inches="tight")
print("wrote", out_dir / "methodology.pdf", "and .png")
