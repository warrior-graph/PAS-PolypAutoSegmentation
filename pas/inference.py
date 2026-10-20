from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data.dataset import eval_transforms
from .models import build_model
from .utils import amp_dtype, get_device


def scale_sizes(base: int, scales=(1.0,)) -> list[int]:
    """Square input sizes for multi-scale inference, rounded to multiples of 32."""
    return [max(32, int(round(base * s / 32)) * 32) for s in scales]


@torch.no_grad()
def predict_logits(model, images: torch.Tensor, tta: bool = False, sizes=None) -> torch.Tensor:
    """Main-head logits for a batch, returned at the resolution of `images`.

    `sizes` lists square input sizes whose logits are averaged (multi-scale TTA); load
    `images` at the largest one so smaller scales are downsampled, not upsampled.
    With `tta` the horizontal flip is averaged in at every size.
    """
    size = images.shape[-2:]
    total, n = 0, 0
    for s in sizes or [None]:
        x = images
        if s is not None and tuple(size) != (s, s):
            x = F.interpolate(images, size=(s, s), mode="bilinear", align_corners=False)
        views = [(x, False), (x.flip(-1), True)] if tta else [(x, False)]
        for v, flipped in views:
            out = model(v)[-1]
            if flipped:
                out = out.flip(-1)
            if out.shape[-2:] != size:
                out = F.interpolate(out.float(), size=size, mode="bilinear", align_corners=False)
            total, n = total + out.float(), n + 1
    return total / n


def filter_components(prob: np.ndarray, thr: float = 0.5, min_rel: float = 0.0) -> np.ndarray:
    """Zero out predicted blobs whose summed probability is below `min_rel` x the strongest blob.

    Encodes a "few polyps per frame" prior: it removes spurious detections (e.g. on ETIS)
    but can drop real secondary polyps, so it is off by default and reported separately.
    """
    if min_rel <= 0:
        return prob
    import cv2

    n, labels = cv2.connectedComponents((prob >= thr).astype(np.uint8))
    if n <= 2:
        return prob
    scores = np.bincount(labels.ravel(), weights=prob.ravel(), minlength=n)[1:]
    drop = np.where(scores < min_rel * scores.max())[0] + 1
    out = prob.copy()
    out[np.isin(labels, drop)] = 0
    return out


def to_probability(logits: torch.Tensor, size) -> np.ndarray:
    """Logits [1,1,h,w] -> probability map (H, W) at the original image size."""
    logits = F.interpolate(logits.float(), size=tuple(size), mode="bilinear", align_corners=False)
    return torch.sigmoid(logits)[0, 0].cpu().numpy()


class Predictor:
    """Loads a checkpoint and segments RGB numpy images."""

    def __init__(
        self, checkpoint: str | Path, device: str = "auto", tta: bool = False, scales=(1.0,), min_component_rel=0.0
    ):
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.cfg = ckpt["config"]
        self.device = get_device(device)
        self.model = build_model(self.cfg["model"], pretrained=False)
        self.model.load_state_dict(ckpt["model"])
        self.model.to(self.device).eval()
        self.sizes = scale_sizes(self.cfg["data"]["image_size"], scales)
        self.tf = eval_transforms(max(self.sizes))
        self.tta = tta
        self.min_component_rel = min_component_rel
        self.dtype = amp_dtype(self.device)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """image: HxWx3 uint8 RGB -> HxW probability map."""
        x = self.tf(image=image)["image"].unsqueeze(0).to(self.device)
        with torch.autocast(self.device.type, dtype=self.dtype, enabled=self.dtype is not None):
            logits = predict_logits(self.model, x, self.tta, self.sizes)
        return filter_components(to_probability(logits, image.shape[:2]), min_rel=self.min_component_rel)


def overlay(image: np.ndarray, prob: np.ndarray, thr: float = 0.5, color=(0, 255, 0), alpha=0.4):
    """Blend the binarized mask over the image and draw its contour."""
    import cv2

    mask = (prob >= thr).astype(np.uint8)
    out = image.copy()
    out[mask == 1] = (out[mask == 1] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color, 2)
    return out
