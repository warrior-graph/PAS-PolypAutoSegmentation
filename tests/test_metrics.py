import numpy as np
import pytest

from pas.metrics import _enhanced_alignment, dice, e_measure_curve, evaluate_image, iou, mae, s_measure


@pytest.fixture
def square():
    gt = np.zeros((50, 60), np.float64)
    gt[10:25, 20:40] = 1
    return gt


def test_perfect_prediction(square):
    m = evaluate_image(square, square)
    assert m["mDice"] == 1 and m["mIoU"] == 1 and m["MAE"] == 0
    assert m["Sm"] == pytest.approx(1, abs=1e-6)
    assert m["maxEm"] == pytest.approx(1, abs=1e-3)


def test_inverted_prediction(square):
    m = evaluate_image(1 - square, square)
    assert m["mDice"] == 0 and m["mIoU"] == 0 and m["MAE"] == 1
    assert m["Sm"] == pytest.approx(0, abs=1e-6)


def test_dice_iou_known_values():
    gt = np.zeros((10, 10))
    gt[:, :5] = 1
    pred = np.zeros((10, 10))
    pred[:, :4] = 1
    assert dice(pred, gt) == pytest.approx(2 * 40 / 90)
    assert iou(pred, gt) == pytest.approx(40 / 50)
    assert mae(pred, gt) == pytest.approx(0.1)


def test_empty_gt():
    gt = np.zeros((20, 20))
    assert dice(gt, gt) == 1
    assert s_measure(gt, gt) == 1


@pytest.mark.parametrize("kind", ["empty", "full", "box", "random"])
def test_vectorized_e_measure_matches_naive(kind):
    rng = np.random.default_rng(1)
    gt = np.zeros((40, 50), bool)
    if kind == "full":
        gt[:] = True
    elif kind == "box":
        gt[5:20, 10:30] = True
    elif kind == "random":
        gt = rng.random(gt.shape) > 0.6
    pred = np.clip(gt * 0.6 + rng.random(gt.shape) * 0.5, 0, 1)
    naive = [_enhanced_alignment((pred >= t).astype(np.float64), gt) for t in np.linspace(0, 1, 256)]
    np.testing.assert_allclose(e_measure_curve(pred, gt), naive, atol=1e-10)
