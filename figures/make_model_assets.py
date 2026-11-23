"""Compute the real model outputs drawn in the architecture figure.

Usage:
    uv run python -m figures.make_model_assets

Runs one frame of a frozen test driver through the final model F-I (ImageNet ResNet-18, pose
fusion, confusion-loss driver adversary) and saves what the figure shows: the keypoints, the
body-centered pose vector, a PCA image of each ResNet stage, the Grad-CAM map, the class
probabilities, and the driver head's probabilities. Also copies the CV driver-probe accuracy of
A-I and F-I from the ablation table. Writes ``figures/assets/model_assets.npz``.
"""

import csv
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from sklearn.decomposition import PCA
from torch import nn
from torchvision import models

from figures.make_overview_assets import (
    COCO_IDX,
    IMG_SIZE,
    POSE_SCORE_THRESH,
    InferenceNet,
    grad_cam,
    pose_features,
)

ROOT = Path(__file__).resolve().parent.parent
FRAME_PATH = (
    ROOT
    / "data"
    / "state-farm-distracted-driver-detection"
    / "imgs"
    / "train"
    / "c2"
    / "img_52690.jpg"
)  # driver p014, one of the four frozen test drivers, class c2 (phone, right hand)
TRUE_CLASS = 2
CKPT = ROOT / "outputs" / "main" / "final_model_F-I.pt"
ABLATION_CSV = ROOT / "outputs" / "main" / "table_ablation_summary.csv"
OUT_FILE = Path(__file__).resolve().parent / "assets" / "model_assets.npz"
STAGE_IDX: list[int] = [4, 5, 6, 7]  # indices of layer1..layer4 inside ``net.features``


def pca_rgb(fmap: np.ndarray) -> np.ndarray:
    """Project a feature map onto its top 3 principal components as an RGB image.

    Args:
        fmap: Activations of one stage, shape [C, H, W].

    Returns:
        Array [H, W, 3] in uint8, each component stretched between its 2nd and 98th percentile.
    """
    c, h, w = fmap.shape
    comps = PCA(n_components=3, random_state=0).fit_transform(fmap.reshape(c, h * w).T)
    lo, hi = np.percentile(comps, 2, axis=0), np.percentile(comps, 98, axis=0)
    scaled = np.clip((comps - lo) / (hi - lo + 1e-8), 0, 1)
    return (scaled.reshape(h, w, 3) * 255).astype(np.uint8)


def load_final_model(device: torch.device) -> tuple[InferenceNet, nn.Sequential]:
    """Load F-I and its training-only driver head from the saved checkpoint.

    Args:
        device: Torch device.

    Returns:
        The inference network and the driver head, both in eval mode.

    Raises:
        RuntimeError: If the checkpoint does not match the inference network.
    """
    state = torch.load(CKPT, map_location="cpu", weights_only=True)
    net = InferenceNet(use_pose=True)
    missing, unexpected = net.load_state_dict(state, strict=False)
    if missing or any(not k.startswith("driver_head.") for k in unexpected):
        raise RuntimeError(f"F-I: missing={missing} unexpected={unexpected}")
    head_state = {k.removeprefix("driver_head."): v for k, v in state.items() if k in unexpected}
    n_drivers = int(head_state["2.weight"].shape[0])
    head = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, n_drivers))
    head.load_state_dict(head_state)
    return net.to(device).eval(), head.to(device).eval()


def read_probe_accuracy() -> dict[str, float]:
    """Read the mean CV driver-probe accuracy of A-I and F-I from the ablation table.

    Returns:
        Mapping config name to the mean probe accuracy (the table stores "mean ± std").
    """
    with open(ABLATION_CSV, newline="", encoding="utf-8") as f:
        rows = {row["config"]: row["CV driver probe"] for row in csv.DictReader(f)}
    return {name: float(rows[name].split()[0]) for name in ("A-I", "F-I")}


def detect_keypoints(image: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Run the frozen Keypoint R-CNN and return the 9 upper-body joints.

    Args:
        image: Full-resolution frame, uint8 tensor [3, H, W].
        device: Torch device.

    Returns:
        Keypoints [9, 2] in [0, 1] image coordinates.

    Raises:
        RuntimeError: If no person is detected with enough confidence.
    """
    weights = models.detection.KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
    kp_model = models.detection.keypointrcnn_resnet50_fpn(weights=weights).to(device).eval()
    with torch.no_grad():
        out = kp_model([image.to(device).float() / 255])[0]
    if len(out["scores"]) == 0 or out["scores"][0] < POSE_SCORE_THRESH:
        raise RuntimeError(f"No confident person detected in {FRAME_PATH.name}")
    h, w = image.shape[1:]
    xy = out["keypoints"][0][COCO_IDX, :2].cpu()
    return (xy / torch.tensor([w, h], dtype=torch.float32)).clamp(0, 1)


def main() -> None:
    """Compute every model output the figure draws and save them in one file."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, head = load_final_model(device)
    full_res = TF.pil_to_tensor(Image.open(FRAME_PATH).convert("RGB"))
    kp = detect_keypoints(full_res, device)
    x = (TF.resize(full_res, [IMG_SIZE, IMG_SIZE], antialias=True).float() / 255)[None].to(device)
    pose_vec = pose_features(kp[None].to(device), torch.ones(1, device=device))  # [1, 19]

    stages: dict[str, np.ndarray] = {}
    h = (x - net.mean) / net.std
    with torch.no_grad():
        for i, layer in enumerate(net.features):
            h = layer(h)
            if i in STAGE_IDX:
                stages[f"stage{STAGE_IDX.index(i) + 1}"] = pca_rgb(h[0].cpu().numpy())
        f_vis = net.pool(h)
        class_probs = net.classify(f_vis, pose_vec).softmax(1)[0].cpu().numpy()
        driver_probs = head(f_vis).softmax(1)[0].cpu().numpy()
    if int(class_probs.argmax()) != TRUE_CLASS:
        raise RuntimeError(f"F-I predicts c{class_probs.argmax()}, expected c{TRUE_CLASS}")

    probe = read_probe_accuracy()
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_FILE,
        kps=kp.numpy(),
        pose_vec=pose_vec[0].cpu().numpy(),
        cam=grad_cam(net, x, pose_vec, TRUE_CLASS),
        class_probs=class_probs,
        driver_probs=driver_probs,
        probe_baseline=np.array(probe["A-I"]),
        probe_final=np.array(probe["F-I"]),
        **stages,
    )
    print(
        f"p(c{TRUE_CLASS}) = {class_probs[TRUE_CLASS]:.3f}, K = {len(driver_probs)}, probe {probe}"
    )
    print("wrote", OUT_FILE)


if __name__ == "__main__":
    main()
