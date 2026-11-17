"""Draw the model-only figure: what each stage sees and how gradient reversal hides the driver.

Usage:
    uv run python figures/make_model_figure.py [output_dir]

Needs ``figures/assets/overview_assets.npz`` (keypoints, Grad-CAM; from make_overview_assets.py)
and ``figures/assets/model_assets.npz`` (stage feature maps; from make_model_assets.py), plus the
saved results in ``outputs-old/``. Writes ``model.pdf`` (vector) and ``model.png`` (300 dpi).
Canvas is 100 x 40 units and the figure has the same ratio, so circles stay round.
"""

import csv
import math
import sys
from collections.abc import Callable
from itertools import pairwise
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image, ImageOps

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FRAME_PATH = (
    ROOT
    / "data"
    / "state-farm-distracted-driver-detection"
    / "imgs"
    / "train"
    / "c2"
    / "img_42941.jpg"
)
HISTORY_CSV = ROOT / "outputs-old" / "figures" / "fig12_adversarial_identity_removal_history.csv"

INK = "#222222"
MUTED = "#6B6B6B"
NEUTRAL = ("#E9EEF5", "#4C6A92")
FLIP = ("#DDF1EA", "#1B7F63")
ADV = ("#E7E5F4", "#5B54A6")
POSE = ("#FCE8D6", "#C2570C")
BAD = ("#FBE3E1", "#B3261E")
LOSS = ("#F1F1F1", "#555555")
FLIP_PERM: list[int] = [0, 2, 1, 4, 3, 6, 5, 8, 7]  # left/right joint swap used when mirroring
LIMBS: list[tuple[int, int]] = [(0, 1), (0, 2), (3, 4), (3, 5), (5, 7), (4, 6), (6, 8)]

W, H = 100.0, 40.0
fig, ax = plt.subplots(figsize=(7.2, 7.2 * H / W))
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")

Point = tuple[float, float]


def box(
    x: float,
    y: float,
    w: float,
    h: float,
    label: str,
    colors: tuple[str, str],
    fs: float = 5.4,
    bold: bool = False,
    ls: str | tuple[int, tuple[int, int]] = "-",
    z: int = 2,
) -> None:
    """Draw a rounded, labeled box with its bottom-left corner at (x, y)."""
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0,rounding_size=0.8",
            fc=colors[0],
            ec=colors[1],
            lw=0.9,
            ls=ls,
            zorder=z,
        )
    )
    if label:
        text(x + w / 2, y + h / 2, label, fs=fs, bold=bold)


def text(
    x: float,
    y: float,
    s: str,
    fs: float = 5.0,
    color: str = INK,
    ha: str = "center",
    bold: bool = False,
    italic: bool = False,
) -> None:
    """Place a text label centered vertically on (x, y)."""
    ax.text(
        x,
        y,
        s,
        fontsize=fs,
        color=color,
        ha=ha,
        va="center",
        fontweight="bold" if bold else "normal",
        style="italic" if italic else "normal",
        linespacing=1.15,
        zorder=6,
    )


def arrow(
    pts: list[Point],
    color: str = INK,
    lw: float = 0.9,
    ls: str | tuple[int, tuple[float, float]] = "-",
) -> None:
    """Draw a polyline through ``pts`` with an arrow head on its last segment."""
    for a, b in pairwise(pts[:-1]):
        ax.plot(
            [a[0], b[0]], [a[1], b[1]], color=color, lw=lw, ls=ls, zorder=4, solid_capstyle="butt"
        )
    ax.add_patch(
        FancyArrowPatch(
            pts[-2],
            pts[-1],
            arrowstyle="-|>",
            mutation_scale=6,
            color=color,
            lw=lw,
            ls=ls,
            shrinkA=0,
            shrinkB=0,
            zorder=4,
        )
    )


def badge(x: float, y: float, n: str, color: str) -> None:
    """Draw a numbered circle marking one of the three ingredients."""
    ax.add_patch(Circle((x, y), 1.2, fc=color, ec="white", lw=0.7, zorder=7))
    ax.text(
        x, y, n, fontsize=5.4, color="white", fontweight="bold", ha="center", va="center", zorder=8
    )


def photo(
    x: float, y: float, w: float, h: float, image: Image.Image, border: str = MUTED
) -> Callable[[float, float], Point]:
    """Draw an image cropped to the box aspect ratio and return a pixel-to-canvas mapper."""
    iw, ih = image.size
    crop_w = min(iw, round(ih * w / h))
    left = (iw - crop_w) // 2
    ax.imshow(
        image.crop((left, 0, left + crop_w, ih)),
        extent=(x, x + w, y, y + h),
        zorder=3,
        interpolation="lanczos",
        aspect="auto",
    )
    ax.add_patch(Rectangle((x, y), w, h, fc="none", ec=border, lw=0.8, zorder=4))
    return lambda px, py: (x + (px - left) / crop_w * w, y + h - py / ih * h)


def skeleton(
    keypoints: np.ndarray, size: tuple[int, int], to_canvas: Callable[[float, float], Point]
) -> None:
    """Draw upper-body keypoints and limbs on a photo."""
    pts = [to_canvas(kx * size[0], ky * size[1]) for kx, ky in keypoints]
    for a, b in LIMBS:
        ax.plot(*zip(pts[a], pts[b], strict=True), color="#FFD400", lw=0.9, zorder=5)
    xs, ys = zip(*pts, strict=True)
    ax.scatter(xs, ys, s=7, c=POSE[1], edgecolors="white", linewidths=0.4, zorder=6)


def feature_bar(x: float, y: float, h: float, colors: tuple[str, str]) -> None:
    """Draw a feature vector as a thin segmented bar."""
    n = 8
    for i in range(n):
        shade = 0.35 + 0.5 * ((i * 5) % n) / n
        ax.add_patch(
            Rectangle(
                (x, y + i * h / n),
                2.5,
                h / n,
                fc=colors[1],
                alpha=shade,
                ec="white",
                lw=0.3,
                zorder=3,
            )
        )
    ax.add_patch(Rectangle((x, y), 2.5, h, fc="none", ec=colors[1], lw=0.8, zorder=4))


def load_history() -> dict[str, dict[str, list[float]]]:
    """Read per-epoch probe accuracy and driver-head loss for both runs of the removal study.

    Returns:
        Mapping model name to {"probe": [...], "adv_loss": [...]} (adv_loss only has values with
        the adversary).
    """
    out: dict[str, dict[str, list[float]]] = {}
    with open(HISTORY_CSV, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        probe_col, loss_col = header.index("probe_acc"), header.index("adv_loss")
        for row in reader:
            entry = out.setdefault(row[0], {"probe": [], "adv_loss": []})
            entry["probe"].append(float(row[probe_col]))
            if row[loss_col]:
                entry["adv_loss"].append(float(row[loss_col]))
    return out


overview = np.load(HERE / "assets" / "overview_assets.npz")
model = np.load(HERE / "assets" / "model_assets.npz")
history = load_history()
k_drivers = int(model["k_drivers"])
with_adv = history["With adversary"]
frame = Image.open(FRAME_PATH).convert("RGB")
mirrored = ImageOps.mirror(frame)
kp = overview["kps"][0]
kp_mirrored = kp[FLIP_PERM].copy()
kp_mirrored[:, 0] = 1 - kp_mirrored[:, 0]  # same transform the notebook applies when it flips

# ------------------------------------------------------------ identity-removal band (bottom)
ax.add_patch(
    FancyBboxPatch(
        (0.4, 1.2),
        99.2,
        13.4,
        boxstyle="round,pad=0,rounding_size=1.0",
        fc="#FBF7F7",
        ec="#E6D6D4",
        lw=0.7,
        zorder=0,
    )
)
badge(1.9, 13.2, "2", ADV[1])
text(
    3.6,
    13.2,
    "Gradient reversal layer (GRL), training only",
    fs=5.4,
    ha="left",
    bold=True,
    color=ADV[1],
)

# ------------------------------------------------------------ inputs
photo(0.5, 17.5, 10.6, 8.0, frame)
ax.text(
    1.0,
    18.4,
    "frame",
    fontsize=4.4,
    color=INK,
    zorder=6,
    bbox={"fc": "white", "ec": "none", "pad": 0.6, "alpha": 0.85},
)
to_canvas = photo(0.5, 28.5, 10.6, 8.0, frame, border=POSE[1])
skeleton(kp, frame.size, to_canvas)
arrow([(2.2, 25.7), (2.2, 28.3)], lw=0.8)
text(3.0, 27.1, "Keypoint R-CNN", fs=4.0, color=POSE[1], ha="left", bold=True)

# ------------------------------------------------------------ 1: semantic flip
box(14.8, 15.4, 13.0, 24.0, "", FLIP, z=1)
badge(16.4, 38.2, "1", FLIP[1])
text(17.9, 38.2, "Semantic flip", fs=4.6, ha="left", bold=True, color=FLIP[1])
to_canvas = photo(16.1, 28.5, 10.4, 7.8, mirrored)
skeleton(kp_mirrored, frame.size, to_canvas)
photo(16.1, 17.5, 10.4, 7.8, mirrored)
text(21.3, 26.9, "label  c2 → c4", fs=4.6, bold=True, color=FLIP[1])
arrow([(11.1, 21.5), (16.1, 21.5)])
arrow([(11.1, 32.4), (16.1, 32.4)])

# ------------------------------------------------------------ 3: pose stream (top lane)
text(37.4, 34.3, "9 joints + flag (19-d)", fs=4.4, color=POSE[1])
arrow([(26.5, 32.4), (47.0, 32.4)], color=POSE[1])
box(47.0, 30.3, 11.0, 4.2, "Pose MLP\n19 → 64 → 64", POSE, fs=4.8)
badge(47.0, 34.5, "3", POSE[1])
arrow([(58.0, 32.4), (60.4, 32.4)])
feature_bar(60.4, 29.7, 5.4, POSE)
text(61.65, 36.5, "f_pose", fs=4.2, color=POSE[1])

# ------------------------------------------------------------ ResNet stages (bottom lane)
text(
    43.6,
    27.0,
    "ResNet-18 stages, fine texture to coarse parts",
    fs=4.6,
    bold=True,
    color=NEUTRAL[1],
)
arrow([(26.5, 21.5), (29.2, 21.5)])
TIMES, MINUS = chr(0xD7), chr(0x2212)  # built from code points to keep the source ASCII-safe
labels = [f"layer {i + 1}\n{n}{TIMES}{n}" for i, n in enumerate([56, 28, 14, 7])]
for i in range(4):
    x = 29.2 + i * 7.6
    photo(x, 18.5, 6.0, 6.0, Image.fromarray(model[f"stage{i + 1}"]), border=NEUTRAL[1])
    text(x + 3.0, 16.9, labels[i], fs=3.8, color=MUTED)
    if i < 3:
        arrow([(x + 6.0, 21.5), (x + 7.6, 21.5)], lw=0.7)
arrow([(58.0, 21.5), (60.4, 21.5)])
feature_bar(60.4, 18.5, 6.0, NEUTRAL)
text(61.65, 25.5, "f_vis", fs=4.2, color=NEUTRAL[1])

# ------------------------------------------------------------ fuse and classify
box(65.4, 24.2, 5.4, 6.6, "concat", NEUTRAL, fs=4.6)
arrow([(62.9, 32.4), (68.1, 32.4), (68.1, 30.8)])
arrow([(62.9, 21.5), (68.1, 21.5), (68.1, 24.2)])
box(73.5, 24.7, 10.0, 5.6, "What?\n10 actions", NEUTRAL, fs=5.0, bold=True)
arrow([(70.8, 27.5), (73.5, 27.5)])
box(73.5, 33.2, 10.0, 2.8, "L_class", LOSS, fs=4.6)
arrow([(78.5, 30.3), (78.5, 33.2)])
arrow([(83.5, 27.5), (86.5, 27.5)], color=MUTED, ls=(0, (2, 1.5)))
photo(
    86.5,
    23.7,
    10.0,
    7.6,
    Image.fromarray(
        np.uint8(
            np.asarray(Image.open(FRAME_PATH).convert("RGB").resize((224, 224))) * 0.5
            + plt.get_cmap("jet")(overview["cam_adversary"][0])[..., :3] * 255 * 0.5
        )
    ),
)
text(91.5, 22.0, "Grad-CAM\nwhere it looks", fs=4.0, bold=True)

# ------------------------------------------------------------ gradient reversal layer (GRL)
GREEN = "#2E7D32"
arrow([(61.65, 18.5), (61.65, 13.4)], color=ADV[1])
box(56.2, 8.6, 10.6, 4.8, "GRL\nbackward \u00d7(\u2212\u03bb)", ADV, fs=4.4, bold=True)
box(70.6, 8.6, 13.0, 4.8, "Driver head\n512\u2192256\u2192K", BAD, fs=4.2)
box(86.5, 8.6, 10.0, 4.8, "L_driver\n(wrong guesses)", LOSS, fs=4.2)
# Forward lane (top): features flow right. Backward lane (bottom): gradients flow left.
arrow([(66.8, 12.0), (70.6, 12.0)], color=ADV[1])
arrow([(83.6, 12.0), (86.5, 12.0)], color=ADV[1])
arrow([(86.5, 9.8), (83.6, 9.8)], color=MUTED, ls=(0, (2.5, 1.5)))
arrow([(70.6, 9.8), (66.8, 9.8)], color=GREEN, lw=1.1, ls=(0, (2.5, 1.5)))
ax.plot([29.2, 58.0], [15.5, 15.5], color=BAD[1], lw=0.7, zorder=3)
arrow([(56.2, 9.8), (43.6, 9.8), (43.6, 15.3)], color=BAD[1], lw=1.2, ls=(0, (2.5, 1.5)))

# Legend that explains the three arrow styles in plain words.
legend = [
    (
        ADV[1],
        "-",
        "Forward: f_vis passes through the GRL unchanged\n"
        "to the driver head, which tries to guess who is driving.",
    ),
    (
        GREEN,
        (0, (2.5, 1.5)),
        "Backward, before the GRL: the normal gradient\ntrains the head to name the driver better.",
    ),
    (
        BAD[1],
        (0, (2.5, 1.5)),
        "Backward, after the GRL: the sign is flipped,\n"
        "so the ResNet learns to make that guess fail.",
    ),
]
for row, (color, style, label) in enumerate(legend):
    y = 11.2 - row * 3.6
    ax.plot([2.4, 5.6], [y, y], color=color, lw=1.2, ls=style, zorder=4)
    text(6.6, y, label, fs=4.2, ha="left")
text(
    44.0,
    5.0,
    "Result in training: the head ends near chance\n"
    f"(loss {with_adv['adv_loss'][-1]:.2f} vs ln K = {math.log(k_drivers):.2f})",
    fs=4.0,
    color=MUTED,
    italic=True,
)

out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE
out_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(out_dir / "model.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(out_dir / "model.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
print("wrote", out_dir / "model.pdf", "and .png")
