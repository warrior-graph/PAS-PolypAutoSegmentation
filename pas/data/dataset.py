"""Image/mask datasets in the PraNet layout.

    data/TrainDataset/{image,masks}/*.png          (Kvasir 900 + ClinicDB 550)
    data/TestDataset/<name>/{images,masks}/*.png   (Kvasir, CVC-ClinicDB, CVC-ColonDB,
                                                   ETIS-LaribPolypDB, CVC-300)

Any directory holding an image folder and a sibling mask folder works.
"""

import random
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

IMAGE_DIRS = ("images", "image", "imgs", "img")
MASK_DIRS = ("masks", "mask", "labels", "gt", "GT")
IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
TEST_SETS = ("Kvasir", "CVC-ClinicDB", "CVC-ColonDB", "ETIS-LaribPolypDB", "CVC-300")
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def _find_subdir(root: Path, names) -> Path | None:
    for n in names:
        if (root / n).is_dir():
            return root / n
    return None


def find_pairs(root: str | Path) -> list[tuple[Path, Path]]:
    """Return sorted (image, mask) pairs matched by file stem."""
    root = Path(root)
    img_dir, mask_dir = _find_subdir(root, IMAGE_DIRS), _find_subdir(root, MASK_DIRS)
    if img_dir is None or mask_dir is None:
        raise FileNotFoundError(f"{root} needs an image dir {IMAGE_DIRS} and a mask dir {MASK_DIRS}")
    masks = {p.stem: p for p in mask_dir.iterdir() if p.suffix.lower() in IMG_EXTS}
    pairs = [
        (p, masks[p.stem])
        for p in sorted(img_dir.iterdir())
        if p.suffix.lower() in IMG_EXTS and p.stem in masks
    ]
    if not pairs:
        raise FileNotFoundError(f"No image/mask pairs found under {root}")
    return pairs


def split_pairs(pairs, val_fraction: float, seed: int = 0):
    """Deterministic train/val split."""
    if val_fraction <= 0:
        return list(pairs), []
    idx = list(range(len(pairs)))
    random.Random(seed).shuffle(idx)
    n_val = max(1, int(round(len(pairs) * val_fraction)))
    val = set(idx[:n_val])
    return [p for i, p in enumerate(pairs) if i not in val], [p for i, p in enumerate(pairs) if i in val]


def read_image(path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Cannot read {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_mask(path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise IOError(f"Cannot read {path}")
    return (mask > 127).astype(np.uint8)


def train_transforms(size: int) -> A.Compose:
    return A.Compose(
        [
            A.Resize(size, size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(
                scale=(0.75, 1.25),
                translate_percent=(-0.1, 0.1),
                rotate=(-20, 20),
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                fill_mask=0,
                p=0.5,
            ),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
            A.Normalize(MEAN, STD),
            ToTensorV2(),
        ]
    )


def eval_transforms(size: int) -> A.Compose:
    return A.Compose([A.Resize(size, size), A.Normalize(MEAN, STD), ToTensorV2()])


class PolypDataset(Dataset):
    """Returns (image[3,H,W] float, mask[1,H,W] float) at the training size.

    With `keep_original=True` the original-resolution mask is also returned (for evaluation).
    """

    def __init__(self, pairs, size=352, train=False, keep_original=False):
        self.pairs = list(pairs)
        self.tf = train_transforms(size) if train else eval_transforms(size)
        self.keep_original = keep_original

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        img_path, mask_path = self.pairs[i]
        img, mask = read_image(img_path), read_mask(mask_path)
        out = self.tf(image=img, mask=mask)
        sample = {
            "image": out["image"],
            "mask": out["mask"].unsqueeze(0).float(),
            "name": img_path.name,
        }
        if self.keep_original:
            sample["mask_orig"] = torch.from_numpy(mask)
        return sample
