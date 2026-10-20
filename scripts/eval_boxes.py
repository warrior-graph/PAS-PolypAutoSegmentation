"""Box-level evaluation on the five test sets, for a segmentation checkpoint or a YOLO model.

    # boxes derived from the segmentation (one box per predicted blob)
    python scripts/eval_boxes.py --checkpoint runs/emcad_b0/best.pth --out results/boxes/seg_b0
    python scripts/eval_boxes.py --checkpoint runs/emcad_b0/best.pth --tta --scales 1.0 1.25 1.5 --out ...
    # dedicated detector (Ultralytics)
    python scripts/eval_boxes.py --yolo runs/yolo/yolo11s/weights/best.pt --imgsz 640 --out results/boxes/yolo11s

Writes <out>.md / .json (metrics) and <out>_preds.json (boxes per image, for drawing).
Ground-truth boxes are the connected components of the ground-truth masks.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pas.data import TEST_SETS, find_pairs, read_image, read_mask  # noqa: E402
from pas.detection import DetectionEvaluator, det_table, mask_to_boxes, prob_to_boxes  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", help="segmentation checkpoint (.pth)")
    src.add_argument("--yolo", help="Ultralytics weights (.pt)")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--test-sets", nargs="+", default=list(TEST_SETS))
    ap.add_argument("--out", required=True, help="output prefix")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--scales", type=float, nargs="+", default=[1.0])
    ap.add_argument("--imgsz", type=int, default=640, help="YOLO inference size")
    ap.add_argument("--conf", type=float, default=0.001, help="YOLO minimum confidence kept for AP")
    ap.add_argument("--score-thr", type=float, default=0.5, help="operating point for precision/recall/F1")
    args = ap.parse_args()

    if args.checkpoint:
        from pas.inference import Predictor

        predictor = Predictor(args.checkpoint, args.device, args.tta, args.scales)

        def predict(image, path):
            return prob_to_boxes(predictor(image))
    else:
        from ultralytics import YOLO

        model = YOLO(args.yolo)
        dev = None if args.device == "auto" else args.device

        def predict(image, path):
            r = model.predict(str(path), imgsz=args.imgsz, conf=args.conf, device=dev, verbose=False)[0]
            return r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()

    results, preds = {}, {}
    for name in args.test_sets:
        set_dir = Path(args.data_root) / "TestDataset" / name
        if not set_dir.is_dir():
            print(f"skipping missing {set_dir}")
            continue
        ev = DetectionEvaluator(score_thr=args.score_thr)
        preds[name] = {}
        for img_path, mask_path in find_pairs(set_dir):
            boxes, scores = predict(read_image(img_path), img_path)
            gt = mask_to_boxes(read_mask(mask_path))
            ev.update(boxes, scores, gt)
            keep = scores >= 0.05  # drawing only; AP above uses every box
            preds[name][img_path.name] = {
                "boxes": boxes[keep].round(1).tolist(),
                "scores": [round(float(s), 4) for s in scores[keep]],
            }
        results[name] = ev.compute()
        print(f"{name}: " + " ".join(f"{k}={v:.3f}" for k, v in results[name].items() if isinstance(v, float)))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    source = args.checkpoint or args.yolo
    settings = f"source={source} tta={args.tta} scales={args.scales} imgsz={args.imgsz} score_thr={args.score_thr}"
    (out.parent / f"{out.name}.md").write_text(det_table(results) + f"\n\n{settings}\n")
    (out.parent / f"{out.name}.json").write_text(json.dumps({"results": results, "settings": settings}, indent=2))
    (out.parent / f"{out.name}_preds.json").write_text(json.dumps(preds))
    print(det_table(results))


if __name__ == "__main__":
    main()
