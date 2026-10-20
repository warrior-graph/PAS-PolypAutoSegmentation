"""Train a polyp segmentation model.

    python train.py --config configs/emcad_b0.yaml
    python train.py --config configs/emcad_b0.yaml --set train.epochs=1 train.batch_size=4
    python train.py --config configs/emcad_b0.yaml --max-steps 20      # smoke test
"""

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from pas.data import PolypDataset, find_pairs, split_pairs
from pas.evaluation import evaluate_pairs, evaluate_test_sets, model_complexity, results_table
from pas.losses import deep_supervision_loss
from pas.models import build_model
from pas.utils import amp_dtype, count_parameters, get_device, load_config, save_checkpoint, seed_everything


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. train.epochs=50")
    ap.add_argument("--max-steps", type=int, default=0, help="stop after N optimizer steps (smoke test)")
    ap.add_argument("--no-test", action="store_true", help="skip the final test-set evaluation")
    return ap.parse_args()


def lr_lambda(warmup_steps, total_steps):
    def fn(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    return fn


def rescale_batch(images, masks, scale):
    """PraNet-style multi-scale training: resize the whole batch, sides rounded to multiples of 32."""
    if scale == 1.0:
        return images, masks
    size = int(round(images.shape[-1] * scale / 32) * 32)
    images = F.interpolate(images, size=(size, size), mode="bilinear", align_corners=False)
    masks = F.interpolate(masks, size=(size, size), mode="bilinear", align_corners=False)
    return images, masks


def main():
    args = parse_args()
    cfg = load_config(args.config, args.set)
    seed_everything(cfg["seed"])
    dcfg, tcfg = cfg["data"], cfg["train"]

    run_dir = Path(cfg["output_dir"]) / cfg["experiment"]
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    device = get_device(cfg.get("device", "auto"))
    dtype = amp_dtype(device) if tcfg.get("amp", True) else None
    print(f"device={device} amp={dtype}")

    pairs = find_pairs(Path(dcfg["root"]) / dcfg["train_dir"])
    train_pairs, val_pairs = split_pairs(pairs, dcfg["val_fraction"], cfg["seed"])
    print(f"train={len(train_pairs)} val={len(val_pairs)} images")
    size = dcfg["image_size"]
    nw = dcfg.get("num_workers", 4)
    loader = DataLoader(
        PolypDataset(train_pairs, size=size, train=True),
        batch_size=tcfg["batch_size"],
        shuffle=True,
        num_workers=nw,
        pin_memory=device.type == "cuda",
        drop_last=len(train_pairs) > tcfg["batch_size"],
        persistent_workers=nw > 0,
    )

    model = build_model(cfg["model"]).to(device)
    print(f"model params: {count_parameters(model) / 1e6:.2f}M")
    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    accum = tcfg.get("grad_accum", 1)
    steps_per_epoch = max(1, len(loader) // accum)
    total_steps = steps_per_epoch * tcfg["epochs"]
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda(steps_per_epoch * tcfg.get("warmup_epochs", 0), total_steps)
    )
    scaler = torch.amp.GradScaler(enabled=dtype == torch.float16)
    scales = tcfg.get("multi_scale", [1.0])

    log_path = run_dir / "log.csv"
    log_fields = ["epoch", "step", "lr", "train_loss", "val_mDice", "val_mIoU", "val_MAE", "time_s"]
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(log_fields)

    best, step, t0 = -1.0, 0, time.time()
    for epoch in range(1, tcfg["epochs"] + 1):
        model.train()
        running, n = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        for i, batch in enumerate(loader):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            images, masks = rescale_batch(images, masks, random.choice(scales))
            with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
                outputs = model(images)
            loss = deep_supervision_loss(outputs, masks, tcfg.get("deep_supervision", "mutation"))
            scaler.scale(loss / accum).backward()
            running += loss.item()
            n += 1
            if (i + 1) % accum == 0:
                if tcfg.get("grad_clip"):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                step += 1
                if args.max_steps and step >= args.max_steps:
                    break

        val = {}
        is_last = epoch == tcfg["epochs"] or (args.max_steps and step >= args.max_steps)
        if val_pairs and (epoch % tcfg.get("val_every", 1) == 0 or is_last):
            val = evaluate_pairs(model, val_pairs, size, device, dtype, fast=True, num_workers=nw)
        row = {
            "epoch": epoch,
            "step": step,
            "lr": scheduler.get_last_lr()[0],
            "train_loss": running / max(n, 1),
            "val_mDice": val.get("mDice", ""),
            "val_mIoU": val.get("mIoU", ""),
            "val_MAE": val.get("MAE", ""),
            "time_s": round(time.time() - t0, 1),
        }
        with open(log_path, "a", newline="") as f:
            csv.DictWriter(f, log_fields).writerow(row)
        print(
            f"epoch {epoch}/{tcfg['epochs']} loss={row['train_loss']:.4f} "
            + (f"val_dice={val['mDice']:.4f} " if val else "")
            + f"lr={row['lr']:.2e} t={row['time_s']}s"
        )

        save_checkpoint(run_dir / "last.pth", model, cfg, epoch=epoch)
        score = val.get("mDice", -row["train_loss"])
        if score > best:
            best = score
            save_checkpoint(run_dir / "best.pth", model, cfg, epoch=epoch, val=val)
            print(f"  new best ({score:.4f}) saved")
        if is_last:
            break

    if args.no_test:
        return
    ckpt = torch.load(run_dir / "best.pth", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    print(f"Evaluating best checkpoint (epoch {ckpt['epoch']}) on test sets")
    results = evaluate_test_sets(model, cfg, device, dtype)
    complexity = model_complexity(model, size, device, dtype)
    with open(run_dir / "test_results.json", "w") as f:
        json.dump({"results": results, "complexity": complexity, "epoch": ckpt["epoch"]}, f, indent=2)
    table = results_table(results)
    (run_dir / "test_results.md").write_text(
        table + "\n\n" + " | ".join(f"{k}: {v:.2f}" for k, v in complexity.items()) + "\n"
    )
    print(table)
    print(complexity)


if __name__ == "__main__":
    main()
