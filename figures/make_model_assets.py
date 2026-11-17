"""Compute the real per-stage feature visualizations for the model figure.

Usage:
    uv run python -m figures.make_model_assets

Runs the first example frame through the saved ImageNet checkpoint C-I (flip + driver adversary)
and turns the output of each ResNet stage into an RGB image by projecting its channels onto the
top three principal components. Also records the number of training drivers K (the size of the
driver head). Writes ``figures/assets/model_assets.npz``.
"""

from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from sklearn.decomposition import PCA

from figures.make_overview_assets import CKPT_DIR, DATA_DIR, FRAMES, IMG_SIZE, load_model

OUT_FILE = Path(__file__).resolve().parent / "assets" / "model_assets.npz"
STAGE_IDX: list[int] = [4, 5, 6, 7]  # indices of layer1..layer4 inside ``net.features``


def pca_rgb(fmap: np.ndarray) -> np.ndarray:
    """Project a feature map [C, H, W] onto its top 3 principal components as a uint8 image.

    Args:
        fmap: Activations of one stage.

    Returns:
        Array [H, W, 3] in uint8, each component stretched between its 2nd and 98th percentile.
    """
    c, h, w = fmap.shape
    comps = PCA(n_components=3, random_state=0).fit_transform(fmap.reshape(c, h * w).T)
    lo, hi = np.percentile(comps, 2, axis=0), np.percentile(comps, 98, axis=0)
    scaled = np.clip((comps - lo) / (hi - lo + 1e-8), 0, 1)
    return (scaled.reshape(h, w, 3) * 255).astype(np.uint8)


def main() -> None:
    """Capture the four stage outputs for the example frame and save them as RGB images."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = load_model("C-I", False, device)
    img = Image.open(DATA_DIR / FRAMES[0]).convert("RGB")
    x = (TF.resize(TF.pil_to_tensor(img), [IMG_SIZE, IMG_SIZE], antialias=True).float() / 255)[None]
    h = (x.to(device) - net.mean) / net.std
    stages: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for i, layer in enumerate(net.features):
            h = layer(h)
            if i in STAGE_IDX:
                stages[f"stage{STAGE_IDX.index(i) + 1}"] = pca_rgb(h[0].cpu().numpy())
    state = torch.load(CKPT_DIR / "final_model_C-I.pt", map_location="cpu", weights_only=True)
    k_drivers = int(state["driver_head.2.weight"].shape[0])
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_FILE, k_drivers=np.array(k_drivers), **stages)
    print({k: v.shape for k, v in stages.items()}, "K =", k_drivers)


if __name__ == "__main__":
    main()
