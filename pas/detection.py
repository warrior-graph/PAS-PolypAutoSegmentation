"""Bounding-box view of polyp localisation.

Ground-truth boxes come from the connected components of the ground-truth masks (the
benchmark has no native boxes). Predicted boxes come either from a segmentation
probability map (`prob_to_boxes`) or from a detector. Both are scored with the same
COCO-style evaluator, so segmentation-derived boxes and a dedicated detector compare
on equal terms.
"""

import cv2
import numpy as np

IOU_THRESHOLDS = np.round(np.arange(0.5, 0.96, 0.05), 2)


def _components(binary, min_area_frac=0.0005, min_rel_area=0.05):
    """Connected components that are not annotation/resize noise.

    Kept: area >= `min_area_frac` of the image and >= `min_rel_area` of the largest one.
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8))
    if n <= 1:
        return labels, []
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = (areas >= min_area_frac * binary.size) & (areas >= min_rel_area * areas.max())
    return labels, [(i + 1, stats[i + 1]) for i in np.where(keep)[0]]


def _box(stat):
    x, y, w, h = stat[cv2.CC_STAT_LEFT], stat[cv2.CC_STAT_TOP], stat[cv2.CC_STAT_WIDTH], stat[cv2.CC_STAT_HEIGHT]
    return [float(x), float(y), float(x + w), float(y + h)]


def mask_to_boxes(mask, **kw) -> np.ndarray:
    """Binary mask (H, W) -> (N, 4) boxes [x1, y1, x2, y2] in pixels."""
    _, comps = _components(mask > 0, **kw)
    return np.array([_box(s) for _, s in comps], dtype=np.float64).reshape(-1, 4)


def prob_to_boxes(prob, thr=0.5, **kw):
    """Probability map -> (boxes (N, 4), scores (N,)); score = mean probability of the blob."""
    labels, comps = _components(prob >= thr, **kw)
    boxes = np.array([_box(s) for _, s in comps], dtype=np.float64).reshape(-1, 4)
    scores = np.array([float(prob[labels == i].mean()) for i, _ in comps])
    return boxes, scores


def box_iou(a, b) -> np.ndarray:
    """IoU matrix between (N, 4) and (M, 4) boxes."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), axis=2)
    area = lambda x: np.prod(x[:, 2:] - x[:, :2], axis=1)
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def _match(pred, scores, gt, thr):
    """Greedy matching by descending score; returns a TP flag per prediction (score order)."""
    order = np.argsort(-scores, kind="stable")
    iou = box_iou(pred[order], gt)
    used = np.zeros(len(gt), bool)
    tp = np.zeros(len(order), bool)
    for k in range(len(order)):
        if len(gt) == 0:
            break
        cand = np.where(~used & (iou[k] >= thr))[0]
        if len(cand):
            j = cand[np.argmax(iou[k, cand])]
            used[j] = tp[k] = True
    return scores[order], tp


def _ap(scores, tp, n_gt):
    """COCO 101-point interpolated average precision."""
    if n_gt == 0:
        return float("nan")
    if len(scores) == 0:
        return 0.0
    order = np.argsort(-scores, kind="stable")
    tp = tp[order]
    ctp, cfp = np.cumsum(tp), np.cumsum(~tp)
    recall = ctp / n_gt
    precision = ctp / np.maximum(ctp + cfp, 1)
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    pts = np.linspace(0, 1, 101)
    idx = np.searchsorted(recall, pts, side="left")
    return float(np.mean([precision[i] if i < len(precision) else 0.0 for i in idx]))


class DetectionEvaluator:
    """Accumulates per-image predictions and reports AP50, AP50-95 and P/R/F1 at a score threshold."""

    def __init__(self, score_thr=0.5):
        self.score_thr = score_thr
        self.images = []  # (pred boxes, scores, gt boxes)

    def update(self, pred_boxes, scores, gt_boxes):
        self.images.append((np.asarray(pred_boxes, float).reshape(-1, 4), np.asarray(scores, float), np.asarray(gt_boxes, float).reshape(-1, 4)))

    def compute(self):
        n_gt = sum(len(g) for _, _, g in self.images)
        aps = {}
        for t in IOU_THRESHOLDS:
            s_all, tp_all = [], []
            for p, s, g in self.images:
                s_sorted, tp = _match(p, s, g, t)
                s_all.append(s_sorted)
                tp_all.append(tp)
            aps[t] = _ap(np.concatenate(s_all) if s_all else np.zeros(0), np.concatenate(tp_all) if tp_all else np.zeros(0, bool), n_gt)
        precision, recall, f1, fp, hit_imgs, best_ious = self._operating_point(self.score_thr, n_gt)
        sweep = [(self._operating_point(t, n_gt)[2], t) for t in np.round(np.arange(0.05, 0.96, 0.05), 2)]
        best_f1, best_thr = max(sweep)
        return {
            "AP50": aps[0.5],
            "AP75": aps[0.75],
            "AP50_95": float(np.nanmean(list(aps.values()))),
            "precision": float(precision),
            "recall": float(recall),
            "F1": float(f1),
            "best_F1": float(best_f1),
            "best_F1_thr": float(best_thr),
            "mean_best_IoU": float(np.mean(best_ious)) if best_ious else float("nan"),
            "images_with_hit": hit_imgs / max(len(self.images), 1),
            "n_images": len(self.images),
            "n_gt": int(n_gt),
            "fp_per_image": float(fp / max(len(self.images), 1)),
        }

    def _operating_point(self, thr, n_gt):
        """Precision, recall, F1, #FP, #images with a hit and best IoU per GT at IoU 0.5."""
        tp = fp = hit_imgs = 0
        best_ious = []
        for p, s, g in self.images:
            keep = s >= thr
            _, flags = _match(p[keep], s[keep], g, 0.5)
            tp += flags.sum()
            fp += (~flags).sum()
            hit_imgs += bool(flags.any())
            iou = box_iou(g, p[keep])
            best_ious += list(iou.max(axis=1)) if iou.size else [0.0] * len(g)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(n_gt, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        return precision, recall, f1, fp, hit_imgs, best_ious


DET_METRICS = ("AP50", "AP50_95", "precision", "recall", "F1", "best_F1", "mean_best_IoU", "fp_per_image")


def det_table(results: dict) -> str:
    cols = DET_METRICS
    lines = ["| Dataset | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for name, m in results.items():
        lines.append(f"| {name} | " + " | ".join(f"{m[c]:.3f}" for c in cols) + " |")
    return "\n".join(lines)


def draw_boxes(image, boxes, color, scores=None, label="", thickness=3, font_scale=None):
    """Draw boxes (and optional scores) on an RGB uint8 image in place; returns it."""
    font_scale = font_scale or max(0.6, image.shape[0] / 600)
    for k, b in enumerate(np.asarray(boxes).reshape(-1, 4)):
        x1, y1, x2, y2 = map(int, b)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        text = label + (f" {scores[k]:.2f}" if scores is not None else "")
        if text.strip():
            org = (x1, max(int(25 * font_scale), y1 - 8))
            cv2.putText(image, text.strip(), org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, max(2, thickness - 1), cv2.LINE_AA)
    return image
