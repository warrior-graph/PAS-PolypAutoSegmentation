"""Convert the PraNet-layout benchmark to an Ultralytics YOLO detection dataset.

    python scripts/make_yolo_dataset.py --data-root data --out yolo_data

Boxes come from the connected components of the masks (noise blobs filtered as in
pas.detection). Train/val use the same 90/10 split and seed as the segmentation runs,
so both models see exactly the same training images. Test sets are evaluated with
scripts/eval_boxes.py (same metric as the segmentation-derived boxes), not here.
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pas.data import find_pairs, read_mask, split_pairs  # noqa: E402
from pas.detection import mask_to_boxes  # noqa: E402


def write_split(pairs, out, split, copy):
    (out / "images" / split).mkdir(parents=True, exist_ok=True)
    (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    n_boxes = 0
    for img_path, mask_path in pairs:
        mask = read_mask(mask_path)
        h, w = mask.shape
        boxes = mask_to_boxes(mask)
        dst = out / "images" / split / img_path.name
        if not dst.exists():
            if copy:
                shutil.copy2(img_path, dst)
            else:
                os.symlink(img_path.resolve(), dst)
        lines = [
            f"0 {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"
            for x1, y1, x2, y2 in boxes
        ]
        (out / "labels" / split / f"{img_path.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        n_boxes += len(lines)
    return n_boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--train-dir", default="TrainDataset")
    ap.add_argument("--out", default="yolo_data")
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--copy", action="store_true", help="copy images instead of symlinking")
    args = ap.parse_args()

    out = Path(args.out)
    pairs = find_pairs(Path(args.data_root) / args.train_dir)
    train, val = split_pairs(pairs, args.val_fraction, args.seed)
    nt = write_split(train, out, "train", args.copy)
    nv = write_split(val, out, "val", args.copy)
    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: polyp\n"
    )
    print(f"train: {len(train)} images, {nt} boxes | val: {len(val)} images, {nv} boxes -> {out / 'data.yaml'}")


if __name__ == "__main__":
    main()
