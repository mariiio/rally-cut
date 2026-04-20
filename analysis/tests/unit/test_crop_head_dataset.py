"""Unit tests for CropContactDataset."""
from pathlib import Path

import numpy as np
import pytest

from rallycut.ml.crop_head.dataset import CropContactDataset


def _make_fake_npz(path: Path, label: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        rally_id="rally-fake-0001",
        frame=100,
        player_crop=np.random.rand(9, 3, 64, 64).astype(np.float32),
        ball_patch=np.random.rand(9, 3, 32, 32).astype(np.float32),
        label=label,
        gbm_conf=0.4,
        source="gt_positive" if label else "random_negative",
    )


def test_dataset_loads_and_yields_expected_shapes(tmp_path: Path) -> None:
    _make_fake_npz(tmp_path / "vid_a" / "001.npz", 1)
    _make_fake_npz(tmp_path / "vid_a" / "002.npz", 0)
    _make_fake_npz(tmp_path / "vid_b" / "001.npz", 1)

    ds = CropContactDataset(tmp_path, ["vid_a", "vid_b"])
    assert len(ds) == 3
    item = ds[0]
    assert item["player_crop"].shape == (9, 3, 64, 64)
    assert item["ball_patch"].shape == (9, 3, 32, 32)
    assert item["label"] in (0, 1)


def test_dataset_filters_by_video_id(tmp_path: Path) -> None:
    _make_fake_npz(tmp_path / "vid_a" / "001.npz", 1)
    _make_fake_npz(tmp_path / "vid_b" / "001.npz", 1)
    ds = CropContactDataset(tmp_path, ["vid_a"])
    assert len(ds) == 1
    assert ds[0]["video_id"] == "vid_a"


def test_dataset_raises_on_empty_cache(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No crops found"):
        CropContactDataset(tmp_path, ["does-not-exist"])
