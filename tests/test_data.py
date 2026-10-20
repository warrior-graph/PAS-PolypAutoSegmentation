import pytest

from pas.data import PolypDataset, find_pairs, split_pairs


def test_find_pairs_both_layouts(synthetic_data):
    assert len(find_pairs(synthetic_data / "TrainDataset")) == 10  # image/ + masks/
    assert len(find_pairs(synthetic_data / "TestDataset" / "Kvasir")) == 3  # images/ + masks/


def test_find_pairs_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        find_pairs(tmp_path)


def test_split_is_deterministic_and_disjoint(synthetic_data):
    pairs = find_pairs(synthetic_data / "TrainDataset")
    tr1, va1 = split_pairs(pairs, 0.2, seed=3)
    tr2, va2 = split_pairs(pairs, 0.2, seed=3)
    assert tr1 == tr2 and va1 == va2
    assert len(va1) == 2 and not set(tr1) & set(va1)


@pytest.mark.parametrize("train", [True, False])
def test_dataset_sample(synthetic_data, train):
    ds = PolypDataset(find_pairs(synthetic_data / "TrainDataset"), size=64, train=train, keep_original=True)
    s = ds[0]
    assert s["image"].shape == (3, 64, 64)
    assert s["mask"].shape == (1, 64, 64)
    assert set(s["mask"].unique().tolist()) <= {0.0, 1.0}
    assert s["mask_orig"].ndim == 2
