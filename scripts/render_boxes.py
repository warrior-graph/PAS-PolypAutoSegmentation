"""Draw ground-truth and predicted boxes side by side for a few test images per dataset.

    python scripts/render_boxes.py --preds seg=results/boxes/seg_emcad_b0_preds.json:0.5 \
        yolo11s=results/boxes/yolo11s_preds.json:0.25 --out results/boxes/figures

Each --preds entry is label=path[:score_threshold]. Per dataset the figure shows the
images where the first model's boxes are worst (lowest IoU with the ground truth)
followed by randomly chosen ones (fixed seed), so successes and failures both appear.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pas.data import find_pairs, read_image, read_mask  # noqa: E402
from pas.detection import box_iou, draw_boxes, mask_to_boxes  # noqa: E402

GT_COLOR = (255, 255, 255)
COLORS = [(0, 220, 0), (255, 0, 200), (0, 160, 255), (255, 200, 0)]


def parse(spec):
    label, rest = spec.split("=", 1)
    path, _, thr = rest.partition(":")
    return label, json.loads(Path(path).read_text()), float(thr or 0.5)


def tile(image, height=300):
    return cv2.resize(image, (int(image.shape[1] * height / image.shape[0]), height))


def legend(width, models):
    bar = np.zeros((34, width, 3), np.uint8)
    items = [("gabarito", GT_COLOR)] + [(f"{lab} (>= {thr:g})", COLORS[i % len(COLORS)]) for i, (lab, _, thr) in enumerate(models)]
    x = 10
    for text, color in items:
        cv2.rectangle(bar, (x, 9), (x + 22, 25), color, 3)
        cv2.putText(bar, text, (x + 30, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
        x += 40 + 10 * len(text)
    return bar


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", nargs="+", required=True, help="label=path[:thr]")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--worst", type=int, default=2)
    ap.add_argument("--random", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    models = [parse(p) for p in args.preds]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in models[0][1]:
        pairs = {p.name: (p, m) for p, m in find_pairs(Path(args.data_root) / "TestDataset" / name)}
        lab0, preds0, thr0 = models[0]
        scored = []
        for img_name, (_, mask_path) in pairs.items():
            gt = mask_to_boxes(read_mask(mask_path))
            p = preds0[name].get(img_name, {"boxes": [], "scores": []})
            boxes = np.array([b for b, s in zip(p["boxes"], p["scores"]) if s >= thr0]).reshape(-1, 4)
            iou = box_iou(gt, boxes)
            scored.append((float(iou.max(axis=1).mean()) if iou.size else 0.0, img_name))
        scored.sort()
        chosen = [n for _, n in scored[: args.worst]]
        rest = [n for _, n in scored[args.worst:]]
        chosen += random.Random(args.seed).sample(rest, min(args.random, len(rest)))

        tiles = []
        for img_name in chosen:
            img_path, mask_path = pairs[img_name]
            image = read_image(img_path)
            t = max(2, image.shape[0] // 150)
            # thicker lines first, so boxes that coincide stay visible underneath each other
            draw_boxes(image, mask_to_boxes(read_mask(mask_path)), GT_COLOR, thickness=t * (len(models) + 2))
            for i, (lab, preds, thr) in enumerate(models):
                p = preds[name].get(img_name, {"boxes": [], "scores": []})
                keep = [k for k, s in enumerate(p["scores"]) if s >= thr]
                draw_boxes(image, np.array([p["boxes"][k] for k in keep]).reshape(-1, 4), COLORS[i % len(COLORS)],
                           scores=[p["scores"][k] for k in keep], thickness=t * (len(models) - i))
            tiles.append(tile(image))
        row = np.hstack(tiles)
        fig = np.vstack([legend(row.shape[1], models), row])
        path = out / f"{name}.jpg"
        cv2.imwrite(str(path), cv2.cvtColor(fig, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(f"{path}: {', '.join(chosen)} (first {args.worst} = worst for {lab0})")


if __name__ == "__main__":
    main()
