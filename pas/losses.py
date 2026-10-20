from itertools import combinations

import torch
import torch.nn.functional as F


def structure_loss(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Boundary-weighted BCE + weighted IoU (PraNet, MICCAI 2020)."""
    logits, mask = logits.float(), mask.float()
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(logits, mask, reduction="none")
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    pred = torch.sigmoid(logits)
    inter = (pred * mask * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def deep_supervision_loss(outputs: list[torch.Tensor], mask: torch.Tensor, mode: str = "mutation"):
    """Loss over the list of head outputs.

    - "mutation": EMCAD's MUTATION loss, i.e. the loss of every non-empty sum of heads
      (2^n - 1 terms).
    - "sum": sum of the per-head losses.
    - "last": only the main (last) head.
    """
    if mode == "last":
        return structure_loss(outputs[-1], mask)
    if mode == "sum":
        return sum(structure_loss(o, mask) for o in outputs)
    if mode == "mutation":
        return sum(
            structure_loss(sum(combo), mask)
            for r in range(1, len(outputs) + 1)
            for combo in combinations(outputs, r)
        )
    raise ValueError(f"Unknown deep supervision mode: {mode}")
