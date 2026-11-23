"""Draw the architecture figure of the final model: learn what the driver does, not who they are.

Usage:
    uv run python -m figures.make_model_figure [output_dir]

Needs ``figures/assets/model_assets.npz`` from ``make_model_assets.py``, so every picture, bar and
number in the figure is a real output of the final model F-I on one frame of an unseen test driver.
Writes ``model.pdf`` (vector) and ``model.png`` (300 dpi). The canvas is 100 x 40 units with the
same aspect ratio as the figure, so circles stay round.
"""

import sys
from collections.abc import Callable, Sequence
from itertools import pairwise
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image

from figures.make_model_assets import FRAME_PATH, TRUE_CLASS

HERE = Path(__file__).resolve().parent

INK = "#1F2328"
MUTED = "#6B7280"
WHAT = ("#E6EEF8", "#2563A8")  # action path (fill, edge)
WHO = ("#FBE6E3", "#C0392B")  # driver identity
POSE = ("#FDEBD9", "#D9730D")
HIDE = "#6A4C93"  # gradient that removes identity from the backbone
BAND = ("#FAF7F7", "#D9C9C7")
LIMBS: list[tuple[int, int]] = [(0, 1), (0, 2), (3, 4), (3, 5), (5, 7), (4, 6), (6, 8)]
CLASS_SHORT: list[str] = [
    "safe driving",
    "texting, right",
    "phone, right",
    "texting, left",
    "phone, left",
    "radio",
    "drinking",
    "reaching behind",
    "hair, makeup",
    "passenger",
]
TIMES, ARROW = chr(0xD7), chr(0x2192)  # code points keep the source ASCII

Point = tuple[float, float]
LineStyle = str | tuple[float, tuple[float, float]]

W, H, Y0 = 100.0, 40.0, 14.4  # Y0 crops the empty canvas below the training band
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["mathtext.fontset"] = "dejavusans"
fig, ax = plt.subplots(figsize=(7.0, 7.0 * (H - Y0) / W))
fig.subplots_adjust(0, 0, 1, 1)
ax.set_xlim(0, W)
ax.set_ylim(Y0, H)
ax.axis("off")


def text(
    x: float,
    y: float,
    s: str,
    fs: float = 5.6,
    color: str = INK,
    ha: str = "center",
    bold: bool = False,
    italic: bool = False,
) -> None:
    """Place a label centered vertically on (x, y).

    Args:
        x: Anchor x in canvas units.
        y: Anchor y in canvas units.
        s: Label, may contain mathtext.
        fs: Font size in points.
        color: Text color.
        ha: Horizontal alignment.
        bold: Use a bold weight.
        italic: Use an italic style.
    """
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
        zorder=8,
    )


def box(
    x: float,
    y: float,
    w: float,
    h: float,
    colors: tuple[str, str],
    label: str = "",
    fs: float = 5.8,
    bold: bool = False,
    ls: LineStyle = "-",
    lw: float = 0.9,
    z: int = 2,
) -> None:
    """Draw a rounded box with its bottom-left corner at (x, y) and an optional centered label.

    Args:
        x: Left edge.
        y: Bottom edge.
        w: Width.
        h: Height.
        colors: Fill and edge color.
        label: Centered text.
        fs: Font size of the label.
        bold: Bold label.
        ls: Edge line style.
        lw: Edge line width.
        z: Drawing order.
    """
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0,rounding_size=0.9",
            fc=colors[0],
            ec=colors[1],
            lw=lw,
            ls=ls,
            zorder=z,
        )
    )
    if label:
        text(x + w / 2, y + h / 2, label, fs=fs, bold=bold)


def arrow(pts: Sequence[Point], color: str = INK, lw: float = 0.9, ls: LineStyle = "-") -> None:
    """Draw a polyline through ``pts`` with an arrow head on its last segment.

    Args:
        pts: Corner points, at least two.
        color: Line color.
        lw: Line width.
        ls: Line style.
    """
    for a, b in pairwise(pts[:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw, ls=ls, zorder=5)
    ax.add_patch(
        FancyArrowPatch(
            pts[-2],
            pts[-1],
            arrowstyle="-|>",
            mutation_scale=6.5,
            color=color,
            lw=lw,
            ls=ls,
            shrinkA=0,
            shrinkB=0,
            zorder=5,
        )
    )


def photo(
    x: float,
    y: float,
    w: float,
    h: float,
    image: Image.Image,
    border: str = MUTED,
    lw: float = 0.8,
) -> Callable[[float, float], Point]:
    """Draw an image center-cropped to the box aspect ratio.

    Args:
        x: Left edge.
        y: Bottom edge.
        w: Width.
        h: Height.
        image: Image to draw.
        border: Frame color.
        lw: Frame line width.

    Returns:
        A function that maps pixel coordinates of ``image`` to canvas coordinates.
    """
    iw, ih = image.size
    crop_w, crop_h = min(iw, round(ih * w / h)), min(ih, round(iw * h / w))
    left, top = (iw - crop_w) // 2, (ih - crop_h) // 2
    ax.imshow(
        image.crop((left, top, left + crop_w, top + crop_h)),
        extent=(x, x + w, y, y + h),
        zorder=3,
        interpolation="lanczos",
        aspect="auto",
    )
    ax.add_patch(Rectangle((x, y), w, h, fc="none", ec=border, lw=lw, zorder=4))
    return lambda px, py: (x + (px - left) / crop_w * w, y + h - (py - top) / crop_h * h)


def skeleton(pts: Sequence[Point], dot: float = 6.0) -> None:
    """Draw upper-body joints and limbs at canvas points ``pts``.

    Args:
        pts: Nine canvas points in keypoint order.
        dot: Marker area in points squared.
    """
    for a, b in LIMBS:
        ax.plot(*zip(pts[a], pts[b], strict=True), color="#FFC300", lw=1.0, zorder=6)
    xs, ys = zip(*pts, strict=True)
    ax.scatter(xs, ys, s=dot, c=POSE[1], edgecolors="white", linewidths=0.4, zorder=7)


def feature_bar(x: float, y: float, h: float, color: str, seed: int) -> None:
    """Draw a feature vector as a column of shaded cells.

    Args:
        x: Left edge.
        y: Bottom edge.
        h: Height.
        color: Cell color.
        seed: Seed for the cell shades, so each vector looks different.
    """
    shades = np.random.default_rng(seed).uniform(0.25, 0.95, 9)
    for i, alpha in enumerate(shades):
        ax.add_patch(
            Rectangle(
                (x, y + i * h / 9), 2.4, h / 9, fc=color, alpha=alpha, ec="white", lw=0.3, zorder=3
            )
        )
    ax.add_patch(Rectangle((x, y), 2.4, h, fc="none", ec=color, lw=0.8, zorder=4))


def pill(x: float, y: float, s: str, colors: tuple[str, str], fs: float = 5.2) -> None:
    """Draw a small rounded tag with white bold text centered on (x, y).

    Args:
        x: Center x.
        y: Center y.
        s: Tag text.
        colors: Unused fill and the tag color.
        fs: Font size.
    """
    ax.text(
        x,
        y,
        s,
        fontsize=fs,
        color="white",
        ha="center",
        va="center",
        fontweight="bold",
        zorder=9,
        bbox={"boxstyle": "round,pad=0.28,rounding_size=0.6", "fc": colors[1], "ec": "none"},
    )


def keypoint_crop(
    image: Image.Image, kps: np.ndarray, aspect: float
) -> tuple[Image.Image, int, int]:
    """Crop the frame around the detected joints with a margin, at a fixed aspect ratio.

    Args:
        image: Full frame.
        kps: Keypoints [9, 2] in [0, 1] image coordinates.
        aspect: Width over height of the crop.

    Returns:
        The crop and the pixel offset (left, top) of the crop in the frame.
    """
    iw, ih = image.size
    px = kps * np.array([iw, ih])
    cx, cy = px.mean(axis=0)
    h = min(ih, (np.ptp(px[:, 1]) + np.ptp(px[:, 0]) / aspect) * 0.9)
    w = h * aspect
    left = int(np.clip(cx - w / 2, 0, iw - w))
    top = int(np.clip(cy - h / 2, 0, ih - h))
    return image.crop((left, top, left + int(w), top + int(h))), left, top


def class_bars(probs: np.ndarray, x: float, y: float, w: float, h: float) -> None:
    """Draw the 10 class probabilities as horizontal bars, the true class highlighted.

    Args:
        probs: Softmax output [10].
        x: Left edge of the bar area (labels sit to its left).
        y: Bottom edge.
        w: Width for a probability of 1.
        h: Total height.
    """
    row = h / len(probs)
    for i, p in enumerate(probs):
        cy = y + h - (i + 0.5) * row
        hit = i == TRUE_CLASS
        ax.add_patch(
            Rectangle((x, cy - row * 0.32), w, row * 0.64, fc="#EEF0F3", ec="none", zorder=2)
        )
        ax.add_patch(
            Rectangle(
                (x, cy - row * 0.32), max(p * w, 0.15), row * 0.64,
                fc=WHAT[1] if hit else "#9AA3AE", ec="none", zorder=3,
            )
        )  # fmt: skip
        text(
            x - 0.6,
            cy,
            f"c{i} {CLASS_SHORT[i]}",
            fs=4.9,
            ha="right",
            bold=hit,
            color=INK if hit else MUTED,
        )
        if hit:
            text(x + p * w - 0.5, cy, f"{p:.1%}", fs=5.0, ha="right", bold=True, color="white")
            continue
        # Small bars are too short to hold their label, so it goes just past the bar's end.
        label = "<0.1%" if p < 0.0005 else f"{p:.1%}"
        text(x + max(p * w, 0.15) + 0.5, cy, label, fs=4.6, ha="left", color=MUTED)


def draw(axes: Axes, assets: dict[str, np.ndarray], frame: Image.Image) -> None:
    """Draw the whole figure on ``axes`` (the module-level canvas).

    Args:
        axes: The canvas axes, kept as a parameter so the drawing order reads top to bottom.
        assets: Arrays saved by ``make_model_assets.py``.
        frame: The input frame.
    """
    del axes  # the helpers draw on the module-level ``ax``

    # ---------------------------------------------------------------- training-only band
    # Sits under the classifier, right of f_vis, so the figure needs no extra row for it.
    box(61.2, 14.8, 15.2, 9.6, BAND, ls=(0, (3, 2)), lw=0.8, z=0)
    pill(66.2, 15.9, "TRAINING ONLY", WHO, fs=4.6)

    # ---------------------------------------------------------------- input
    photo(0.5, 22.0, 15.0, 11.25, frame, border=INK, lw=1.0)
    text(8.0, 20.7, f"frame 224{TIMES}224", fs=5.0, color=MUTED)

    # ---------------------------------------------------------------- pose stream (top lane)
    py = 36.0
    arrow([(15.5, 27.5), (16.6, 27.5), (16.6, py), (18.4, py)], color=POSE[1])
    box(18.4, 33.4, 8.6, 5.2, POSE, "Keypoint\nR-CNN", fs=5.4, bold=True)
    text(22.7, 32.2, "frozen, COCO", fs=4.6, color=POSE[1], italic=True)
    arrow([(27.0, py), (28.8, py)], color=POSE[1])
    crop, left, top = keypoint_crop(frame, assets["kps"], 8.0 / 6.4)
    to_canvas = photo(28.8, 32.8, 8.0, 6.4, crop, border=POSE[1])
    fw, fh = frame.size
    skeleton([to_canvas(kx * fw - left, ky * fh - top) for kx, ky in assets["kps"]], dot=5.0)
    text(32.8, 31.6, "9 upper-body joints", fs=4.6, color=POSE[1])

    arrow([(36.8, py), (39.0, py)], color=POSE[1])
    # Body-centered pose: origin at mid-shoulder, unit = shoulder width (image y points down).
    coords = assets["pose_vec"][:18].reshape(9, 2) * np.array([1.0, -1.0])
    scale = 2.6 / np.abs(coords).max()
    ox, oy = 42.4, 35.6
    box(39.0, 32.8, 6.8, 6.4, ("#FFFFFF", POSE[1]), lw=0.7)
    ax.plot([39.4, 45.4], [oy, oy], color="#D1D5DB", lw=0.5, zorder=4)
    ax.plot([ox, ox], [33.2, 38.8], color="#D1D5DB", lw=0.5, zorder=4)
    skeleton([(ox + cx * scale, oy + cy * scale) for cx, cy in coords], dot=4.0)
    text(42.4, 31.6, "body-centered", fs=4.6, color=POSE[1])

    arrow([(45.8, py), (47.6, py)], color=POSE[1])
    box(47.6, 33.4, 8.0, 5.2, POSE, f"Pose MLP\n19{ARROW}64{ARROW}64", fs=5.2)
    arrow([(55.6, py), (57.4, py)], color=POSE[1])
    feature_bar(57.4, 33.0, 6.0, POSE[1], seed=1)
    text(58.6, 31.9, r"$\mathbf{f}_{\mathrm{pose}}$ 64", fs=5.0, color=POSE[1])

    # ---------------------------------------------------------------- visual stream (bottom lane)
    vy = 23.6
    arrow([(15.5, vy), (17.4, vy)])
    box(17.4, 17.4, 38.2, 12.6, WHAT, z=1)
    text(18.4, 28.8, "ResNet-18 backbone", fs=5.8, bold=True, color=WHAT[1], ha="left")
    text(54.8, 28.8, "ImageNet init, fine-tuned", fs=4.8, color=WHAT[1], ha="right", italic=True)
    for i, n in enumerate([56, 28, 14, 7]):
        sx = 19.4 + i * 9.1
        stage = Image.fromarray(assets[f"stage{i + 1}"]).resize(
            (224, 224), Image.Resampling.NEAREST
        )
        photo(sx, 20.3, 6.6, 6.6, stage, border=WHAT[1], lw=0.6)
        text(sx + 3.3, 19.0, f"stage {i + 1}  {n}{TIMES}{n}", fs=4.5, color=MUTED)
        # Forward pass above, the backward pass of the "hide who" loss below.
        nxt = sx + 9.1 if i < 3 else 57.4
        arrow([(sx + 6.6, vy + 0.8), (nxt, vy + 0.8)], color=WHAT[1] if i < 3 else INK, lw=0.7)
        arrow([(nxt, vy - 0.8), (sx + 6.6, vy - 0.8)], color=HIDE, lw=0.9, ls=(0, (2.4, 1.4)))
    feature_bar(57.4, 20.3, 6.6, WHAT[1], seed=2)
    text(58.6, 19.2, r"$\mathbf{f}_{\mathrm{vis}}$ 512", fs=5.0, color=WHAT[1])

    # ---------------------------------------------------------------- fuse and classify
    cx, cy = 63.6, 31.0
    ax.add_patch(Circle((cx, cy), 1.5, fc="white", ec=INK, lw=0.9, zorder=4))
    text(cx, cy + 0.1, "+", fs=8.0, bold=True)
    # Sits a little low so it clears the f_pose label just above-left.
    text(cx - 1.9, cy - 1.0, "concat", fs=4.6, color=MUTED, ha="right")
    arrow([(59.8, py), (cx, py), (cx, cy + 1.5)])
    arrow([(59.8, vy + 3.0), (cx, vy + 3.0), (cx, cy - 1.5)])
    arrow([(cx + 1.5, cy), (67.2, cy)])
    box(67.2, cy - 3.2, 7.6, 6.4, WHAT, f"Dropout\n+ Linear\n576{ARROW}10", fs=5.0)
    text(
        71.0,
        cy - 4.6,
        r"$\mathcal{L}_{\mathrm{cls}} = \mathrm{CE}(y, \hat{y})$",
        fs=4.8,
        color=WHAT[1],
    )
    arrow([(74.8, cy), (76.6, cy)], color=WHAT[1])

    text(88.4, 38.6, "What is the driver doing?", fs=7.4, bold=True, color=WHAT[1])
    class_bars(assets["class_probs"], 87.6, 17.4, 11.6, 19.6)

    # ---------------------------------------------------------------- driver adversary (band)
    hx, hy, hw, hh = 63.8, 19.6, 11.8, 4.0
    arrow([(59.8, 22.4), (hx, 22.4)], color=WHO[1])
    box(hx, hy, hw, hh, WHO)
    text(hx + hw / 2, hy + 2.7, "Driver head", fs=6.0, bold=True)
    text(hx + hw / 2, hy + 1.0, "guesses who is driving", fs=4.5, color=WHO[1])

    # The head learns to name the driver, and the backbone learns to make that guess fail.
    dash = (0, (2.4, 1.4))
    arrow([(71.4, hy), (71.4, 17.6), (74.2, 17.6), (74.2, hy)], color=WHO[1], ls=dash)
    text(70.8, 18.4, "learn who", fs=5.2, color=WHO[1], bold=True, ha="right")
    # The confusion loss enters at f_vis (head frozen) and flows back through every ResNet stage.
    arrow([(hx, 21.0), (59.8, 21.0)], color=HIDE, lw=1.2, ls=dash)


def main() -> None:
    """Load the assets, draw the figure, and save it as PDF and PNG."""
    with np.load(HERE / "assets" / "model_assets.npz") as data:
        assets = {k: data[k] for k in data.files}
    frame = Image.open(FRAME_PATH).convert("RGB")
    draw(ax, assets, frame)
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "model.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_dir / "model.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    print("wrote", out_dir / "model.pdf", "and .png")


if __name__ == "__main__":
    main()
