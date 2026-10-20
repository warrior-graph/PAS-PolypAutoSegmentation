"""Per-image segmentation metrics used in the polyp segmentation literature.

`pred` is a probability map in [0, 1], `gt` a binary mask; both HxW numpy arrays.
S-measure follows Fan et al. (ICCV 2017), E-measure follows Fan et al. (IJCAI 2018);
both mirror the implementation in py_sod_metrics.
"""

import numpy as np

EPS = np.finfo(np.float64).eps


def dice(pred, gt, thr=0.5):
    p, g = pred >= thr, gt > 0.5
    denom = p.sum() + g.sum()
    return 1.0 if denom == 0 else float(2.0 * np.logical_and(p, g).sum() / denom)


def iou(pred, gt, thr=0.5):
    p, g = pred >= thr, gt > 0.5
    union = np.logical_or(p, g).sum()
    return 1.0 if union == 0 else float(np.logical_and(p, g).sum() / union)


def mae(pred, gt):
    return float(np.abs(pred.astype(np.float64) - (gt > 0.5)).mean())


# ---------------------------------------------------------------- S-measure
def _s_object(pred, gt):
    x = pred[gt]
    if x.size == 0:
        return 0.0
    mean = x.mean()
    sigma = x.std(ddof=1) if x.size > 1 else 0.0
    return 2 * mean / (mean**2 + 1 + sigma + EPS)


def _object_score(pred, gt):
    fg = pred * gt
    bg = (1 - pred) * ~gt
    u = gt.mean()
    return u * _s_object(fg, gt) + (1 - u) * _s_object(bg, ~gt)


def _ssim(pred, gt):
    h, w = pred.shape
    n = h * w
    if n <= 1:
        return 0.0
    gt = gt.astype(np.float64)
    x, y = pred.mean(), gt.mean()
    sx = ((pred - x) ** 2).sum() / (n - 1)
    sy = ((gt - y) ** 2).sum() / (n - 1)
    sxy = ((pred - x) * (gt - y)).sum() / (n - 1)
    alpha = 4 * x * y * sxy
    beta = (x**2 + y**2) * (sx + sy)
    if alpha != 0:
        return alpha / (beta + EPS)
    return 1.0 if beta == 0 else 0.0


def _centroid(gt):
    h, w = gt.shape
    area = gt.sum()
    if area == 0:
        return int(round(w / 2)), int(round(h / 2))
    x = np.round((gt.sum(axis=0) * np.arange(w)).sum() / area)
    y = np.round((gt.sum(axis=1) * np.arange(h)).sum() / area)
    return int(x) + 1, int(y) + 1


def _region_score(pred, gt):
    h, w = gt.shape
    x, y = _centroid(gt)
    area = h * w
    w1 = x * y / area
    w2 = y * (w - x) / area
    w3 = (h - y) * x / area
    w4 = 1 - w1 - w2 - w3
    quads = [(slice(0, y), slice(0, x)), (slice(0, y), slice(x, w)), (slice(y, h), slice(0, x)), (slice(y, h), slice(x, w))]
    return sum(wq * _ssim(pred[q], gt[q]) for wq, q in zip((w1, w2, w3, w4), quads) if wq > 0)


def s_measure(pred, gt, alpha=0.5):
    pred = pred.astype(np.float64)
    gt = gt > 0.5
    y = gt.mean()
    if y == 0:
        return float(1 - pred.mean())
    if y == 1:
        return float(pred.mean())
    return float(max(0.0, alpha * _object_score(pred, gt) + (1 - alpha) * _region_score(pred, gt)))


# ---------------------------------------------------------------- E-measure
def _enhanced_alignment(fg, gt):
    """fg: binarized prediction as float64, gt: bool."""
    h, w = gt.shape
    if gt.sum() == 0:
        enhanced = 1.0 - fg
    elif gt.all():
        enhanced = fg.astype(np.float64)
    else:
        fm = fg - fg.mean()
        g = gt.astype(np.float64)
        g = g - g.mean()
        align = 2 * fm * g / (fm**2 + g**2 + EPS)
        enhanced = (align + 1) ** 2 / 4
    return float(enhanced.sum() / (h * w - 1 + EPS))


def e_measure_curve(pred, gt, num_thresholds=256):
    """E-measure for thresholds in [0, 1]; mean and max of this curve are the usual mEφ / maxEφ.

    Vectorized over thresholds: for a binary map the alignment matrix only takes four
    values (one per TP/FP/FN/TN cell), so only the confusion counts are needed.
    Equivalent to calling `_enhanced_alignment` for every threshold.
    """
    gt = gt > 0.5
    n = gt.size
    thresholds = np.linspace(0, 1, num_thresholds)
    pos = np.sort(pred[gt].ravel())
    neg = np.sort(pred[~gt].ravel())
    tp = (pos.size - np.searchsorted(pos, thresholds, side="left")).astype(np.float64)
    fp = (neg.size - np.searchsorted(neg, thresholds, side="left")).astype(np.float64)
    denom = n - 1 + EPS
    if pos.size == 0:
        return (n - fp) / denom
    if neg.size == 0:
        return tp / denom
    fn, tn = pos.size - tp, neg.size - fp
    p = (tp + fp) / n
    q = pos.size / n

    def cell(a, b):
        align = 2 * a * b / (a**2 + b**2 + EPS)
        return (align + 1) ** 2 / 4

    total = (
        tp * cell(1 - p, 1 - q) + fp * cell(1 - p, -q) + fn * cell(-p, 1 - q) + tn * cell(-p, -q)
    )
    return total / denom


METRIC_NAMES = ("mDice", "mIoU", "Sm", "meanEm", "maxEm", "MAE")


def evaluate_image(pred, gt, num_thresholds=256):
    em = e_measure_curve(pred, gt, num_thresholds)
    return {
        "mDice": dice(pred, gt),
        "mIoU": iou(pred, gt),
        "Sm": s_measure(pred, gt),
        "meanEm": float(em.mean()),
        "maxEm": float(em.max()),
        "MAE": mae(pred, gt),
    }


class MetricTracker:
    def __init__(self, num_thresholds=256, fast=False):
        """fast=True only computes Dice/IoU/MAE (used during training validation)."""
        self.num_thresholds = num_thresholds
        self.fast = fast
        self.values = []

    def update(self, pred, gt):
        if self.fast:
            self.values.append({"mDice": dice(pred, gt), "mIoU": iou(pred, gt), "MAE": mae(pred, gt)})
        else:
            self.values.append(evaluate_image(pred, gt, self.num_thresholds))

    def compute(self):
        if not self.values:
            return {}
        return {k: float(np.mean([v[k] for v in self.values])) for k in self.values[0]}
