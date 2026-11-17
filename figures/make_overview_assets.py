"""Compute the real model outputs shown in the overview figure (keypoints and Grad-CAM).

Usage:
    uv run python figures/make_overview_assets.py

Runs the frozen Keypoint R-CNN and Grad-CAM on the three example frames and writes
``figures/assets/overview_assets.npz``. ``make_overview_figure.py`` only reads that file, so it
needs no PyTorch. The Grad-CAM models are the saved last-fold ImageNet checkpoints A-I and C-I in
``outputs-old/`` (C-I has no pose stream, so its map does not depend on the pose convention).
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from torch import nn
from torchvision import models

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "state-farm-distracted-driver-detection" / "imgs" / "train" / "c2"
CKPT_DIR = ROOT / "outputs-old"
OUT_FILE = Path(__file__).resolve().parent / "assets" / "overview_assets.npz"

FRAMES: list[str] = ["img_42941.jpg", "img_56457.jpg", "img_69455.jpg"]  # p002, p035, p042
IMG_SIZE = 224
POSE_SCORE_THRESH = 0.5
MIN_SHOULDER_WIDTH = 0.05
COCO_IDX: list[int] = [0, 3, 4, 5, 6, 7, 8, 9, 10]  # nose, ears, shoulders, elbows, wrists
IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


class InferenceNet(nn.Module):
    """Inference-only copy of ``DriverNet`` in novelApproach.ipynb (no driver head).

    Module names match the notebook so its saved state dict loads directly.
    """

    def __init__(self, use_pose: bool) -> None:
        """Build the ResNet-18 trunk, optional pose MLP, and the classifier.

        Args:
            use_pose: Whether the pose stream is fused into the classifier.
        """
        super().__init__()
        self.use_pose = use_pose
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        resnet = models.resnet18(weights=None)
        self.features = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
        )  # fmt: skip
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
        if use_pose:
            self.pose_mlp = nn.Sequential(
                nn.Linear(19, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU()
            )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.30), nn.Linear(512 + (64 if use_pose else 0), 10)
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the last conv map [B,512,7,7] and the pooled feature [B,512]."""
        fmap = self.features((x - self.mean) / self.std)
        return fmap, self.pool(fmap)

    def classify(self, f_vis: torch.Tensor, pose_f: torch.Tensor) -> torch.Tensor:
        """Return class logits [B,10] from the (optionally pose-fused) features."""
        if self.use_pose:
            f_vis = torch.cat([f_vis, self.pose_mlp(pose_f)], dim=1)
        return self.classifier(f_vis)


def load_model(short: str, use_pose: bool, device: torch.device) -> InferenceNet:
    """Load a saved notebook checkpoint, ignoring only the training-only driver head.

    Args:
        short: Config short name such as ``"D-I"``.
        use_pose: Whether that config fuses pose.
        device: Torch device.

    Returns:
        The model in eval mode.

    Raises:
        RuntimeError: If any weight other than the driver head fails to match.
    """
    net = InferenceNet(use_pose)
    state = torch.load(CKPT_DIR / f"final_model_{short}.pt", map_location="cpu", weights_only=True)
    missing, unexpected = net.load_state_dict(state, strict=False)
    if missing or any(not k.startswith("driver_head") for k in unexpected):
        raise RuntimeError(f"{short}: missing={missing} unexpected={unexpected}")
    return net.to(device).eval()


def pose_features(kp: torch.Tensor, det: torch.Tensor) -> torch.Tensor:
    """Body-centered, shoulder-width-normalized pose vector [B,19] (as in the notebook)."""
    mid = 0.5 * (kp[:, 3:4] + kp[:, 4:5])
    width = (kp[:, 3] - kp[:, 4]).norm(dim=-1).clamp(min=MIN_SHOULDER_WIDTH)
    coords = ((kp - mid) / width[:, None, None]).flatten(1)
    return torch.cat([coords * det[:, None], det[:, None]], dim=1)


def grad_cam(net: InferenceNet, x: torch.Tensor, pf: torch.Tensor, target: int) -> np.ndarray:
    """Grad-CAM heat-map [224,224] in [0,1] for one image (same method as the notebook)."""
    with torch.enable_grad():
        fmap, _ = net.encode(x)
        fmap.retain_grad()
        logits = net.classify(net.pool(fmap), pf)
        net.zero_grad()
        logits[0, target].backward()
        w = fmap.grad.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * fmap).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
    arr = cam[0, 0].detach().cpu().numpy()
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)


def main() -> None:
    """Detect keypoints and compute Grad-CAM for the example frames, then save them."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kp_model = (
        models.detection.keypointrcnn_resnet50_fpn(
            weights=models.detection.KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
        )
        .to(device)
        .eval()
    )
    base, adv = load_model("A-I", False, device), load_model("C-I", False, device)

    kps, cams_a, cams_d = [], [], []
    for name in FRAMES:
        full_res = TF.pil_to_tensor(Image.open(DATA_DIR / name).convert("RGB"))
        h, w = full_res.shape[1:]
        with torch.no_grad():
            out = kp_model([full_res.to(device).float() / 255])[0]
        if len(out["scores"]) == 0 or out["scores"][0] < POSE_SCORE_THRESH:
            raise RuntimeError(f"No confident person detected in {name}")
        xy = out["keypoints"][0][COCO_IDX, :2].cpu()  # pixels in the full-resolution frame
        kp_norm = (xy / torch.tensor([w, h], dtype=torch.float32)).clamp(0, 1)

        x = (TF.resize(full_res, [IMG_SIZE, IMG_SIZE], antialias=True).float() / 255)[None]
        x = x.to(device)
        pf = pose_features(kp_norm[None].to(device), torch.ones(1, device=device))
        with torch.no_grad():
            pred_a = int(base.classify(base.encode(x)[1], pf).argmax())
            pred_c = int(adv.classify(adv.encode(x)[1], pf).argmax())
        print(f"{name}: A-I predicts c{pred_a}, C-I predicts c{pred_c} (true c2)")
        kps.append(kp_norm.numpy())
        cams_a.append(grad_cam(base, x, pf, 2))  # evidence for the true class c2
        cams_d.append(grad_cam(adv, x, pf, 2))

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_FILE, kps=np.stack(kps), cam_baseline=np.stack(cams_a), cam_adversary=np.stack(cams_d)
    )
    print("wrote", OUT_FILE)


if __name__ == "__main__":
    main()
