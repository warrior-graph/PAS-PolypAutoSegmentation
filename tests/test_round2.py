"""Strong augmentation, multi-scale TTA, component filter and the experiment runner."""

import subprocess
import sys

import numpy as np
import pytest
import torch
import yaml

from conftest import ROOT
from pas.data import PolypDataset, find_pairs
from pas.inference import filter_components, predict_logits, scale_sizes
from pas.models import build_model


def test_strong_aug_sample(synthetic_data):
    ds = PolypDataset(find_pairs(synthetic_data / "TrainDataset"), size=64, train=True, aug="strong")
    for i in range(len(ds)):
        s = ds[i]
        assert s["image"].shape == (3, 64, 64) and s["mask"].shape == (1, 64, 64)
        assert set(s["mask"].unique().tolist()) <= {0.0, 1.0}


def test_scale_sizes():
    assert scale_sizes(352, (1.0, 1.25, 1.5)) == [352, 448, 512]
    assert scale_sizes(448, (1.0,)) == [448]


def test_multiscale_logits_shape_and_identity():
    torch.manual_seed(0)
    model = build_model({"name": "emcad", "encoder": "pvt_v2_b0"}, pretrained=False).eval()
    x = torch.randn(1, 3, 96, 96)
    single = predict_logits(model, x)
    assert torch.allclose(predict_logits(model, x, sizes=[96]), single)
    multi = predict_logits(model, x, tta=True, sizes=[64, 96])
    assert multi.shape == single.shape == (1, 1, 96, 96)


def test_filter_components_drops_weak_blobs():
    prob = np.zeros((40, 40))
    prob[5:20, 5:20] = 0.9  # strong blob
    prob[30:33, 30:33] = 0.6  # weak spurious blob
    out = filter_components(prob, min_rel=0.3)
    assert out[10, 10] == 0.9 and out[31, 31] == 0
    assert filter_components(prob, min_rel=0.0) is prob  # off by default


def test_runner_end_to_end(synthetic_data, tmp_path):
    matrix = tmp_path / "m.yaml"
    matrix.write_text(yaml.safe_dump({
        "defaults": {
            "config": "configs/emcad_b0.yaml",
            "set": ["device=cpu", "model.pretrained=false", "data.image_size=64", "data.val_fraction=0.2",
                    "data.num_workers=0", "train.epochs=1", "train.batch_size=4", "train.grad_accum=1",
                    "data.test_sets=[Kvasir]"],
            "evals": {"tta_ms_cc": ["--tta", "--scales", 1.0, 1.5, "--min-component-rel", 0.3, "--device", "cpu"]},
        },
        "experiments": {"a_basic": {}, "b_strong": {"set": ["data.aug=strong"]}},
    }))
    out = tmp_path / "runs"
    cmd = [sys.executable, "scripts/run_experiments.py", str(matrix), "--out", str(out), "--data-root", str(synthetic_data)]
    subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True, text=True)
    for name in ("a_basic", "b_strong"):
        assert (out / name / "test_results.json").exists()
        assert (out / name / "eval_tta_ms_cc.json").exists()
    summary = (out / "summary.md").read_text()
    assert "a_basic" in summary and "b_strong" in summary and "tta_ms_cc" in summary

    # relaunch: everything is skipped
    r = subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True, text=True)
    assert "already trained" in r.stdout


@pytest.mark.parametrize("flag", ["--dry-run"])
def test_round2_matrix_parses(flag):
    r = subprocess.run([sys.executable, "scripts/run_experiments.py", "experiments/round2.yaml", flag],
                       cwd=ROOT, check=True, capture_output=True, text=True)
    assert "e5_b2_strong_448" in r.stdout and "data.aug=strong" in r.stdout
