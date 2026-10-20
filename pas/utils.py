import copy
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    """Load a YAML config and apply `section.key=value` overrides (values parsed as YAML)."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for item in overrides or []:
        key, _, value = item.partition("=")
        node = cfg
        *parents, leaf = key.split(".")
        for k in parents:
            node = node.setdefault(k, {})
        node[leaf] = yaml.safe_load(value)
    return cfg


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def amp_dtype(device: torch.device) -> torch.dtype | None:
    """bf16 where supported (Ampere+, DGX Spark), fp16 on older GPUs (T4/P100), none on CPU."""
    if device.type != "cuda":
        return None
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def save_checkpoint(path, model, cfg, **extra):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": copy.deepcopy(cfg), **extra}, path)


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())
