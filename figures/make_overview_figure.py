"""Draw the paper's overview figure: the problem, our three ingredients, and the test on strangers.

Usage:
    uv run python figures/make_overview_figure.py [output_dir]

Writes ``overview.pdf`` (vector, for LaTeX) and ``overview.png`` (300 dpi preview).
Canvas is 100 x 28.8 units and the figure is sized to the same ratio, so one unit is the same
length in x and y (circles stay round).
"""

import csv
import sys
from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image

INK = "#222222"
MUTED = "#6B6B6B"
NEUTRAL = ("#E9EEF5", "#4C6A92")  # backbone and classifier
FLIP = ("#DDF1EA", "#1B7F63")  # ingredient 1
ADV = ("#E7E5F4", "#5B54A6")  # ingredient 2
POSE = ("#FCE8D6", "#C2570C")  # ingredient 3
BAD = ("#FBE3E1", "#B3261E")
# Three development drivers, all performing c2 (phone call), chosen for visibly different looks.
PHOTOS: list[tuple[str, str]] = [
    ("p002", "img_42941.jpg"),
    ("p035", "img_56457.jpg"),
    ("p042", "img_69455.jpg"),
]
ASSETS = Path(__file__).resolve().parent / "assets" / "overview_assets.npz"
DATA_DIR = (
    Path(__file__).resolve().parent.parent / "data" / "state-farm-distracted-driver-detection"
)


RESULTS_DIR = Path(__file__).resolve().parent.parent / "outputs-old"
HISTORY_CSV = RESULTS_DIR / "figures" / "fig12_adversarial_identity_removal_history.csv"


def load_results() -> tuple[tuple[float, float], dict[str, list[tuple[float, float]]]]:
    """Read the saved results used by panel C (ImageNet-initialized runs).

    Returns:
        ((random-split accuracy, unseen-driver accuracy) for the baseline, per-epoch history).
        The history maps "Without adversary" and "With adversary" to a list of
        (driver-ID probe accuracy, class accuracy) per epoch on held-out drivers.
    """
    with open(RESULTS_DIR / "table_leakage_study.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    row = next(r for r in rows if r["init"] == "imagenet" and r["metric"] == "accuracy")
    leak = (float(row["random split"]), float(row["driver-grouped (mean)"]))
    history: dict[str, list[tuple[float, float]]] = {}
    with open(HISTORY_CSV, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        probe_col, class_col = header.index("probe_acc"), header.index("class_acc")
        for row in reader:
            # Column 0 is the model name (its header is empty in this file).
            history.setdefault(row[0], []).append((float(row[probe_col]), float(row[class_col])))
    return leak, history


def load_frame(subject: str, name: str) -> Image.Image:
    """Load a c2 training frame from the Kaggle dataset.

    Args:
        subject: Driver id, used only for the error message.
        name: Image file name inside ``imgs/train/c2``.

    Returns:
        The RGB frame.

    Raises:
        FileNotFoundError: If the dataset has not been extracted into ``data/``.
    """
    path = DATA_DIR / "imgs" / "train" / "c2" / name
    if not path.is_file():
        raise FileNotFoundError(f"Missing frame for driver {subject}: {path}")
    return Image.open(path).convert("RGB")


W, H = 100.0, 28.8
fig, ax = plt.subplots(figsize=(7.2, 7.2 * H / W))
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")


def box(
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    colors: tuple[str, str],
    fs: float = 6.2,
    bold: bool = False,
    ls: str = "-",
    z: int = 2,
) -> None:
    """Draw a rounded, labeled box with its bottom-left corner at (x, y)."""
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0,rounding_size=0.9",
            fc=colors[0],
            ec=colors[1],
            lw=1.0,
            ls=ls,
            zorder=z,
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        color=INK,
        zorder=z + 1,
        fontweight="bold" if bold else "normal",
        linespacing=1.15,
    )


def arrow(
    p0: tuple[float, float],
    p1: tuple[float, float],
    color: str = INK,
    ls: str = "-",
    lw: float = 1.0,
    rad: float = 0.0,
) -> None:
    """Draw an arrow from p0 to p1, optionally curved."""
    ax.add_patch(
        FancyArrowPatch(
            p0,
            p1,
            arrowstyle="-|>",
            mutation_scale=7,
            color=color,
            lw=lw,
            ls=ls,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=0,
            shrinkB=0,
            zorder=4,
        )
    )


def text(
    x: float,
    y: float,
    s: str,
    fs: float = 6.0,
    color: str = INK,
    ha: str = "center",
    bold: bool = False,
    italic: bool = False,
) -> None:
    """Place a text label centered on (x, y)."""
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


def badge(x: float, y: float, n: str, color: str) -> None:
    """Draw a numbered circle marking one of the three ingredients."""
    ax.add_patch(Circle((x, y), 1.35, fc=color, ec="white", lw=0.8, zorder=7))
    ax.text(
        x, y, n, color="white", fontsize=6.2, fontweight="bold", ha="center", va="center", zorder=8
    )


def photo(
    x: float, y: float, w: float, h: float, image: Image.Image
) -> Callable[[float, float], tuple[float, float]]:
    """Draw a real dataset frame, center-cropped to the box aspect ratio, with a thin border.

    Args:
        x: Left edge in canvas units.
        y: Bottom edge in canvas units.
        w: Width in canvas units.
        h: Height in canvas units.
        image: The frame to draw.

    Returns:
        A function mapping a pixel (px, py) of ``image`` to canvas coordinates.
    """
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
    ax.add_patch(Rectangle((x, y), w, h, fc="none", ec=MUTED, lw=0.8, zorder=4))
    return lambda px, py: (x + (px - left) / crop_w * w, y + h - py / ih * h)


def overlay_heatmap(image: Image.Image, cam: np.ndarray, alpha: float = 0.5) -> Image.Image:
    """Blend a [0, 1] Grad-CAM map (jet colormap) over a frame resized to the map size."""
    base = np.asarray(image.resize(cam.shape[::-1]), dtype=np.float32) / 255
    heat = plt.get_cmap("jet")(cam)[..., :3]
    blended = (1 - alpha) * base + alpha * heat
    return Image.fromarray((blended * 255).astype(np.uint8))


# COCO-style limbs over the 9 joints (nose, l/r ear, l/r shoulder, l/r elbow, l/r wrist).
LIMBS: list[tuple[int, int]] = [(0, 1), (0, 2), (3, 4), (3, 5), (5, 7), (4, 6), (6, 8)]


def draw_skeleton(
    keypoints: np.ndarray,
    size: tuple[int, int],
    to_canvas: Callable[[float, float], tuple[float, float]],
) -> None:
    """Draw detected upper-body keypoints and limbs on top of a photo.

    Args:
        keypoints: Array [9, 2] of (x, y) in [0, 1] image coordinates.
        size: Image (width, height) in pixels.
        to_canvas: Pixel to canvas mapping returned by ``photo``.
    """
    pts = [to_canvas(kx * size[0], ky * size[1]) for kx, ky in keypoints]
    for a, b in LIMBS:
        ax.plot(*zip(pts[a], pts[b], strict=True), color="#FFD400", lw=1.0, zorder=5)
    xs, ys = zip(*pts, strict=True)
    ax.scatter(xs, ys, s=9, c=POSE[1], edgecolors="white", linewidths=0.4, zorder=6)


def panel_header(x: float, n: str, title: str) -> None:
    """Draw a numbered panel title with a thin rule beneath."""
    ax.text(
        x,
        26.9,
        f"{n}  {title}",
        fontsize=7.2,
        ha="left",
        va="baseline",
        fontweight="bold",
        color=INK,
    )


# ---------------------------------------------------------------- panel A: the shortcut
panel_header(0.5, "A", "The shortcut")
for i, (subject, frame) in enumerate(PHOTOS):
    x = 0.5 + i * 7.2
    photo(x, 16.4, 6.4, 8.0, load_frame(subject, frame))
    text(x + 3.2, 14.8, f"driver {'ABC'[i]}", fs=5.2, color=MUTED)
text(10.9, 12.2, "all labeled  c2: phone call", fs=5.6, bold=True)
arrow((10.9, 10.9), (10.9, 8.3), color=BAD[1])
box(0.5, 1.8, 20.8, 6.4, "", BAD)
text(10.9, 6.3, "Shortcut: memorize", fs=5.8, color=BAD[1], bold=True)
text(10.9, 3.9, "who, fail on new drivers", fs=5.8, color=BAD[1], bold=True)

# ---------------------------------------------------------------- panel B: our model
bx = 24.5
panel_header(bx, "B", "Our model: three ingredients")
ax.add_patch(Rectangle((bx - 0.7, 0.7), 47.6, 25.1, fc="#FAFAFA", ec="#E1E1E1", lw=0.6, zorder=0))

assets = np.load(ASSETS)
frame = load_frame(*PHOTOS[0])
photo(bx, 13.4, 8.4, 6.3, frame)
ax.text(
    bx + 0.5,
    14.0,
    "frame",
    fontsize=4.8,
    color=INK,
    zorder=6,
    bbox={"fc": "white", "ec": "none", "pad": 0.8, "alpha": 0.85},
)
arrow((bx + 4.2, 12.9), (bx + 4.2, 11.0), lw=0.9)
to_canvas = photo(bx, 4.6, 8.4, 6.3, frame)
draw_skeleton(assets["kps"][0], frame.size, to_canvas)
text(bx + 4.2, 2.7, "Keypoint R-CNN\n(frozen)", fs=4.6, color=POSE[1], bold=True)

fx = bx + 10.4
box(fx, 3.4, 5.2, 16.8, "", FLIP)
text(fx + 2.6, 11.8, "Semantic flip\nc1↔c3 · c2↔c4", fs=5.2, color=FLIP[1], bold=True)
ax.texts[-1].set_rotation(90)
badge(fx + 2.6, 21.4, "1", FLIP[1])
arrow((bx + 8.6, 17.0), (fx, 17.0), lw=0.9)
arrow((bx + 8.6, 6.2), (fx, 6.2), lw=0.9)

vx = fx + 7.2
box(vx, 14.4, 9.4, 5.6, "ResNet-18\nbackbone", NEUTRAL, fs=5.6)
box(vx, 3.4, 9.4, 5.6, "Pose MLP\n19 → 64", POSE, fs=5.6)
badge(vx + 9.4, 9.0, "3", POSE[1])
arrow((fx + 5.2, 17.2), (vx, 17.2), lw=0.9)
arrow((fx + 5.2, 6.2), (vx, 6.2), lw=0.9)

cx = vx + 11.6
box(cx, 9.6, 5.6, 5.2, "concat", NEUTRAL, fs=5.4)
arrow((vx + 9.4, 17.2), (cx + 2.8, 17.2), lw=0.9)
arrow((cx + 2.8, 17.2), (cx + 2.8, 14.8), lw=0.9)
arrow((vx + 9.4, 6.2), (cx + 2.8, 6.2), lw=0.9)
arrow((cx + 2.8, 6.2), (cx + 2.8, 9.6), lw=0.9)
ox = cx + 7.8
box(ox, 9.8, 8.8, 4.8, "What?\n10 actions", NEUTRAL, fs=5.6, bold=True)
arrow((cx + 5.6, 12.2), (ox, 12.2), lw=0.9)
cam_img = overlay_heatmap(frame, assets["cam_adversary"][0])
photo(ox + 1.7, 3.9, 5.4, 5.4, cam_img)
text(ox + 4.4, 2.3, "Grad-CAM:\nhand and phone", fs=4.6, color=INK, bold=True)

gx = vx - 0.2
box(gx, 22.0, 8.4, 3.2, "GRL  \u00d7(\u2212\u03bb)", ADV, fs=5.2, ls=(0, (3, 2)))
box(gx + 10.2, 22.0, 13.4, 3.2, "Driver head: who?", BAD, fs=5.0, ls=(0, (3, 2)))
text(gx + 20.4, 20.9, "training only", fs=4.8, color=MUTED, italic=True)
badge(gx - 0.2, 25.2, "2", ADV[1])
arrow((vx + 2.6, 20.0), (vx + 2.6, 22.0), lw=0.9, color=ADV[1])
arrow((vx + 6.4, 22.0), (vx + 6.4, 20.0), lw=0.9, color=BAD[1], ls=(0, (2, 1.5)))
text(vx + 7.2, 21.0, "flipped", fs=4.2, color=BAD[1], ha="left", italic=True)
arrow((gx + 8.4, 23.6), (gx + 10.2, 23.6), lw=0.9, color=ADV[1])

# ---------------------------------------------------------------- panel C: the evidence
tx = 75.8
panel_header(tx, "C", "Why strangers matter")
leak, history = load_results()


def bar(
    x: float,
    base: float,
    scale: float,
    frac: float,
    color: tuple[str, str],
    label: str,
    fmt: str = ".0%",
) -> None:
    """Draw one bar with its value on top and a label underneath."""
    ax.add_patch(
        Rectangle((x, base), 4.2, frac * scale, fc=color[0], ec=color[1], lw=0.9, zorder=3)
    )
    text(x + 2.1, base + frac * scale + 0.9, format(frac, fmt), fs=5.0, bold=True)
    text(x + 2.1, base - 1.0, label, fs=4.8)


# Chart 1: the same baseline, scored on seen drivers vs. unseen drivers.
text(tx + 11.8, 25.4, "Same model, two exams", fs=5.2, bold=True)
base1, scale1 = 16.6, 5.6
ax.plot([tx, tx + 23.4], [base1, base1], color=MUTED, lw=0.7, zorder=2)
bar(tx + 4.0, base1, scale1, leak[0], BAD, "seen drivers", ".1%")
bar(tx + 14.8, base1, scale1, leak[1], NEUTRAL, "new drivers", ".1%")
text(tx + 11.8, 14.0, "a random split inflates accuracy", fs=4.6, color=MUTED, italic=True)

# Chart 2: driver-ID probe accuracy per epoch on held-out drivers, with and without the adversary.
text(tx + 11.8, 12.2, "Who is driving? (probe accuracy)", fs=5.2, bold=True)
base2, scale2, x0, x1 = 5.8, 5.2, tx + 3.2, tx + 19.4  # y axis spans 50% to 100%
ax.plot([x0, x1], [base2, base2], color=MUTED, lw=0.7, zorder=2)
for name, color, dy in [("Without adversary", MUTED, 0.5), ("With adversary", BAD[1], -0.5)]:
    probes = [p for p, _ in history[name]]
    xs = [x0 + (x1 - x0) * i / (len(probes) - 1) for i in range(len(probes))]
    ys = [base2 + (p - 0.5) / 0.5 * scale2 for p in probes]
    ax.plot(xs, ys, color=color, lw=1.2, zorder=4)
    text(x1 + 0.5, ys[-1] + dy, f"{probes[-1]:.0%}", fs=4.8, color=color, ha="left", bold=True)
text(x0 - 0.5, base2, "50%", fs=4.2, color=MUTED, ha="right")
text(x0 - 0.5, base2 + scale2, "100%", fs=4.2, color=MUTED, ha="right")
text(x0, base2 - 1.0, "epoch 1", fs=4.4, color=MUTED, ha="left")
text(x1, base2 - 1.0, "15", fs=4.4, color=MUTED, ha="right")
text(tx, 3.4, "without adversary", fs=4.6, color=MUTED, ha="left", bold=True)
text(tx + 23.4, 3.4, "with adversary", fs=4.6, color=BAD[1], ha="right", bold=True)
kept = {n: h[-1][1] for n, h in history.items()}
text(
    tx + 11.8,
    1.6,
    f"class accuracy kept: {kept['Without adversary']:.0%} vs {kept['With adversary']:.0%}",
    fs=4.4,
    color=MUTED,
)

out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
out_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(out_dir / "overview.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(out_dir / "overview.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
print("wrote", out_dir / "overview.pdf", "and .png")
