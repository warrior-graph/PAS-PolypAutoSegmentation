import numpy as np
import pytest

from pas.detection import DetectionEvaluator, box_iou, mask_to_boxes, prob_to_boxes


def two_polyps():
    m = np.zeros((100, 100), np.uint8)
    m[10:30, 10:40] = 1
    m[60:90, 60:80] = 1
    return m


def test_mask_to_boxes_and_noise_filter():
    m = two_polyps()
    m[0, 99] = 1  # 1-pixel annotation noise is dropped
    np.testing.assert_array_equal(mask_to_boxes(m), [[10, 10, 40, 30], [60, 60, 80, 90]])
    assert mask_to_boxes(np.zeros((10, 10))).shape == (0, 4)


def test_prob_to_boxes_scores_are_blob_means():
    prob = two_polyps() * 0.8
    boxes, scores = prob_to_boxes(prob)
    assert len(boxes) == 2 and np.allclose(scores, 0.8)


def test_box_iou():
    a = np.array([[0, 0, 10, 10.0]])
    assert box_iou(a, a)[0, 0] == pytest.approx(1)
    assert box_iou(a, np.array([[5, 0, 15, 10.0]]))[0, 0] == pytest.approx(50 / 150)


def test_evaluator_perfect_and_partial():
    gt = mask_to_boxes(two_polyps())
    ev = DetectionEvaluator()
    ev.update(gt, [0.9, 0.8], gt)
    m = ev.compute()
    assert m["AP50"] == pytest.approx(1) and m["AP50_95"] == pytest.approx(1) and m["F1"] == pytest.approx(1)

    ev = DetectionEvaluator()
    ev.update(gt[:1], [0.9], gt)                      # one hit, one miss
    ev.update(np.array([[0, 0, 5, 5.0]]), [0.95], gt)  # a confident false positive
    m = ev.compute()
    assert m["precision"] == pytest.approx(0.5) and m["recall"] == pytest.approx(0.25)
    assert 0 < m["AP50"] < 0.25
