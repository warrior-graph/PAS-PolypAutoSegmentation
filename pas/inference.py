from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data.dataset import eval_transforms
from .models import build_model
from .utils import amp_dtype, get_device


@torch.no_grad()
def predict_logits(model, images: torch.Tensor, tta: bool = False) -> torch.Tensor:
    """Main-head logits for a batch; with `tta` the horizontal flip is averaged in."""
    logits = model(images)[-1]
    if tta:
        logits = (logits + model(images.flip(-1))[-1].flip(-1)) / 2
    return logits


def to_probability(logits: torch.Tensor, size) -> np.ndarray:
    """Logits [1,1,h,w] -> probability map (H, W) at the original image size."""
    logits = F.interpolate(logits.float(), size=tuple(size), mode="bilinear", align_corners=False)
    return torch.sigmoid(logits)[0, 0].cpu().numpy()


class Predictor:
    """Loads a checkpoint and segments RGB numpy images."""

    def __init__(self, checkpoint: str | Path, device: str = "auto", tta: bool = False):
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.cfg = ckpt["config"]
        self.device = get_device(device)
        self.model = build_model(self.cfg["model"], pretrained=False)
        self.model.load_state_dict(ckpt["model"])
        self.model.to(self.device).eval()
        self.size = self.cfg["data"]["image_size"]
        self.tf = eval_transforms(self.size)
        self.tta = tta
        self.dtype = amp_dtype(self.device)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """image: HxWx3 uint8 RGB -> HxW probability map."""
        x = self.tf(image=image)["image"].unsqueeze(0).to(self.device)
        with torch.autocast(self.device.type, dtype=self.dtype, enabled=self.dtype is not None):
            logits = predict_logits(self.model, x, self.tta)
        return to_probability(logits, image.shape[:2])


def overlay(image: np.ndarray, prob: np.ndarray, thr: float = 0.5, color=(0, 255, 0), alpha=0.4):
    """Blend the binarized mask over the image and draw its contour."""
    import cv2

    mask = (prob >= thr).astype(np.uint8)
    out = image.copy()
    out[mask == 1] = (out[mask == 1] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color, 2)
    return out
