import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _write_pairs(img_dir: Path, mask_dir: Path, n: int, rng):
    img_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    for i in range(n):
        h, w = rng.integers(60, 100, size=2)
        img = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
        mask = np.zeros((h, w), np.uint8)
        cy, cx, r = h // 2, w // 2, int(min(h, w) * 0.25)
        cv2.circle(mask, (int(cx), int(cy)), r, 255, -1)
        img[mask > 0] = (200, 80, 80)
        cv2.imwrite(str(img_dir / f"{i}.png"), img)
        cv2.imwrite(str(mask_dir / f"{i}.png"), mask)


@pytest.fixture
def synthetic_data(tmp_path):
    """Tiny dataset in the PraNet layout."""
    rng = np.random.default_rng(0)
    root = tmp_path / "data"
    _write_pairs(root / "TrainDataset" / "image", root / "TrainDataset" / "masks", 10, rng)
    _write_pairs(root / "TestDataset" / "Kvasir" / "images", root / "TestDataset" / "Kvasir" / "masks", 3, rng)
    return root
