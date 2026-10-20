import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.flop_counter import FlopCounterMode

from .data import PolypDataset, find_pairs
from .inference import predict_logits, to_probability
from .metrics import METRIC_NAMES, MetricTracker


@torch.no_grad()
def evaluate_pairs(model, pairs, size, device, dtype=None, tta=False, fast=False, num_workers=2, save_dir=None):
    """Evaluate on (image, mask) pairs at the original mask resolution."""
    model.eval()
    loader = DataLoader(
        PolypDataset(pairs, size=size, keep_original=True), batch_size=1, num_workers=num_workers
    )
    tracker = MetricTracker(fast=fast)
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        gt = batch["mask_orig"][0].numpy()
        with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
            logits = predict_logits(model, x, tta)
        prob = to_probability(logits, gt.shape)
        tracker.update(prob, gt)
        if save_dir:
            cv2.imwrite(str(Path(save_dir) / batch["name"][0]), (prob * 255).astype(np.uint8))
    return tracker.compute()


def evaluate_test_sets(model, cfg, device, dtype=None, tta=False, save_dir=None):
    """Evaluate every test set listed in the config; returns {name: metrics}."""
    root = Path(cfg["data"]["root"]) / cfg["data"]["test_dir"]
    results = {}
    for name in cfg["data"]["test_sets"]:
        set_dir = root / name
        if not set_dir.is_dir():
            print(f"[eval] skipping missing test set {set_dir}")
            continue
        results[name] = evaluate_pairs(
            model,
            find_pairs(set_dir),
            cfg["data"]["image_size"],
            device,
            dtype,
            tta,
            num_workers=cfg["data"].get("num_workers", 2),
            save_dir=Path(save_dir) / name if save_dir else None,
        )
        print(f"[eval] {name}: " + " ".join(f"{k}={v:.4f}" for k, v in results[name].items()))
    return results


def results_table(results: dict) -> str:
    """Markdown table: one row per dataset."""
    lines = ["| Dataset | " + " | ".join(METRIC_NAMES) + " |", "|---" * (len(METRIC_NAMES) + 1) + "|"]
    for name, m in results.items():
        lines.append(f"| {name} | " + " | ".join(f"{m.get(k, float('nan')):.4f}" for k in METRIC_NAMES) + " |")
    return "\n".join(lines)


@torch.no_grad()
def model_complexity(model, size, device, dtype=None, n_warmup=10, n_runs=50):
    """Params (M), GFLOPs (multiply-adds, as reported in papers) and FPS at batch size 1."""
    model.eval()
    x = torch.randn(1, 3, size, size, device=device)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    with FlopCounterMode(display=False) as fc:
        model(x)
    gmacs = fc.get_total_flops() / 2e9
    with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
        for _ in range(n_warmup):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_runs):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
    fps = n_runs / (time.perf_counter() - t0)
    return {"params_M": params, "GFLOPs": gmacs, "FPS": fps}
