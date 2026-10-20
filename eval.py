"""Evaluate a checkpoint on the five standard test sets.

    python eval.py --checkpoint runs/emcad_b0/best.pth
    python eval.py --checkpoint runs/emcad_b0/best.pth --tta --save-preds runs/emcad_b0/preds
    python eval.py --checkpoint runs/emcad_b0/best.pth --tta --scales 1.0 1.25 1.5 --min-component-rel 0.3
"""

import argparse
import csv
import json
from pathlib import Path

import torch

from pas.evaluation import evaluate_test_sets, model_complexity, results_table
from pas.metrics import METRIC_NAMES
from pas.models import build_model
from pas.utils import amp_dtype, get_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-root", help="override data.root from the checkpoint config")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--tta", action="store_true", help="average with horizontal flip")
    ap.add_argument("--scales", type=float, nargs="+", default=[1.0], help="multi-scale TTA, multiples of the train size")
    ap.add_argument("--min-component-rel", type=float, default=0.0,
                    help="drop blobs weaker than this fraction of the strongest (0 = off)")
    ap.add_argument("--save-preds", help="directory for predicted probability maps")
    ap.add_argument("--out", help="output prefix (default: <checkpoint dir>/eval[_tta])")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    if args.data_root:
        cfg["data"]["root"] = args.data_root
    device = get_device(args.device)
    dtype = amp_dtype(device)
    model = build_model(cfg["model"], pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    results = evaluate_test_sets(
        model, cfg, device, dtype, args.tta, args.save_preds, args.scales, args.min_component_rel
    )
    complexity = model_complexity(model, cfg["data"]["image_size"], device, dtype)

    tag = "eval" + ("_tta" if args.tta else "") + ("_ms" if args.scales != [1.0] else "")
    tag += f"_cc{args.min_component_rel:g}" if args.min_component_rel > 0 else ""
    out = Path(args.out) if args.out else Path(args.checkpoint).parent / tag
    out.parent.mkdir(parents=True, exist_ok=True)
    table = results_table(results)
    summary = " | ".join(f"{k}: {v:.2f}" for k, v in complexity.items())
    summary += f" (single pass, {device})\n\ntta={args.tta} scales={args.scales} min_component_rel={args.min_component_rel}"
    (out.parent / f"{out.name}.md").write_text(f"{table}\n\n{summary}\n")
    with open(out.parent / f"{out.name}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", *METRIC_NAMES])
        for name, m in results.items():
            w.writerow([name, *(f"{m[k]:.4f}" for k in METRIC_NAMES)])
    (out.parent / f"{out.name}.json").write_text(json.dumps({"results": results, "complexity": complexity}, indent=2))
    print(table)
    print(summary)
    print(f"saved {out}.md/.csv/.json")


if __name__ == "__main__":
    main()
