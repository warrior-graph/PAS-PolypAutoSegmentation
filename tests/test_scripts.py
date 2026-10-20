"""End-to-end: train -> eval -> predict on a synthetic dataset, on CPU."""

import json
import subprocess
import sys

from conftest import ROOT


def run(*args):
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True, capture_output=True, text=True)


def test_train_eval_predict(synthetic_data, tmp_path):
    out = tmp_path / "runs"
    run(
        "train.py",
        "--config", "configs/emcad_b0.yaml",
        "--set",
        f"data.root={synthetic_data}",
        f"output_dir={out}",
        "experiment=test",
        "device=cpu",
        "model.pretrained=false",
        "data.image_size=64",
        "data.val_fraction=0.2",
        "data.num_workers=0",
        "train.epochs=2",
        "train.batch_size=4",
        "train.grad_accum=1",
        "data.test_sets=[Kvasir]",
    )
    run_dir = out / "test"
    for f in ("best.pth", "last.pth", "log.csv", "test_results.json", "test_results.md"):
        assert (run_dir / f).exists(), f
    assert "Kvasir" in json.loads((run_dir / "test_results.json").read_text())["results"]

    run("eval.py", "--checkpoint", str(run_dir / "best.pth"), "--device", "cpu", "--tta")
    assert (run_dir / "eval_tta.csv").exists()

    preds = tmp_path / "preds"
    images = synthetic_data / "TestDataset" / "Kvasir" / "images"
    run("predict.py", "--checkpoint", str(run_dir / "best.pth"), "--input", str(images), "--output", str(preds), "--device", "cpu")
    assert len(list(preds.glob("*_mask.png"))) == 3
    assert len(list(preds.glob("*_overlay.png"))) == 3
