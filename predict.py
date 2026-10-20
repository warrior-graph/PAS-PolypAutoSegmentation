"""Segment polyps in an image or a folder of images.

    python predict.py --checkpoint runs/emcad_b0/best.pth --input some_frame.png --output preds/
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from pas.data.dataset import IMG_EXTS, read_image
from pas.inference import Predictor, overlay


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--input", required=True, help="image file or directory")
    ap.add_argument("--output", default="predictions")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    predictor = Predictor(args.checkpoint, args.device, args.tta)
    src = Path(args.input)
    files = sorted(p for p in src.iterdir() if p.suffix.lower() in IMG_EXTS) if src.is_dir() else [src]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    for path in files:
        image = read_image(path)
        prob = predictor(image)
        mask = ((prob >= args.threshold) * 255).astype(np.uint8)
        cv2.imwrite(str(out / f"{path.stem}_mask.png"), mask)
        cv2.imwrite(str(out / f"{path.stem}_overlay.png"), cv2.cvtColor(overlay(image, prob, args.threshold), cv2.COLOR_RGB2BGR))
        print(f"{path.name}: polyp area {100 * (prob >= args.threshold).mean():.1f}%")
    print(f"saved {len(files)} predictions to {out}")


if __name__ == "__main__":
    main()
