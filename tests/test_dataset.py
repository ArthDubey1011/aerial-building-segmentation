from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from src.data.dataset import TileDataset, TrainDataset, blank_fraction, list_pairs, tile_starts


@pytest.fixture
def tiny_dataset(tmp_path):
    """Two synthetic 600x700 images with a square 'building' in the mask; no real data needed."""
    (tmp_path / "images").mkdir()
    (tmp_path / "masks").mkdir()
    rng = np.random.default_rng(0)
    for name in ("a.png", "b.png"):
        img = rng.integers(1, 254, size=(600, 700, 3), dtype=np.uint8)
        mask = np.zeros((600, 700, 3), dtype=np.uint8)
        mask[100:300, 100:300] = 255
        cv2.imwrite(str(tmp_path / "images" / name), img)
        cv2.imwrite(str(tmp_path / "masks" / name), mask)
    return list_pairs(tmp_path / "images", tmp_path / "masks")


def test_train_sample_shapes_and_values(tiny_dataset):
    ds = TrainDataset(tiny_dataset, crop_size=256, length=5)
    img, mask = ds[0]
    assert len(ds) == 5
    assert img.shape == (3, 256, 256) and img.dtype == torch.float32
    assert mask.shape == (1, 256, 256) and mask.dtype == torch.float32
    assert set(torch.unique(mask).tolist()) <= {0.0, 1.0}


def test_tile_starts_cover_length_with_full_tiles():
    starts = tile_starts(1500, 512)
    assert starts == [0, 512, 988]
    assert starts[-1] + 512 == 1500  # last tile ends exactly at the image edge
    assert tile_starts(300, 512) == [0]


def test_tile_dataset_covers_image(tiny_dataset):
    ds = TileDataset(tiny_dataset, tile_size=256)
    # 600 -> starts [0,256,344] (3), 700 -> [0,256,444] (3) => 9 tiles per image, 2 images
    assert len(ds) == 18
    img, mask = ds[0]
    assert img.shape == (3, 256, 256) and mask.shape == (1, 256, 256)


def test_blank_fraction():
    img = np.full((10, 10, 3), 128, dtype=np.uint8)
    img[:5] = 255
    assert blank_fraction(img) == 0.5


def test_mostly_blank_crops_are_skipped(tmp_path):
    """Left half of the image is pure white. With enough tries, crops should come from the valid right side."""
    (tmp_path / "images").mkdir()
    (tmp_path / "masks").mkdir()
    img = np.full((512, 1024, 3), 100, dtype=np.uint8)
    img[:, :512] = 255
    cv2.imwrite(str(tmp_path / "images" / "x.png"), img)
    cv2.imwrite(str(tmp_path / "masks" / "x.png"), np.zeros((512, 1024, 3), dtype=np.uint8))
    pairs = list_pairs(tmp_path / "images", tmp_path / "masks")
    ds = TrainDataset(pairs, crop_size=256, length=1, blank_max_fraction=0.5, max_tries=200, augment=False)
    for _ in range(10):
        image, _ = ds[0]
        # normalised white would be ~2.2-2.6; valid pixels (100/255) are ~ -0.3..0.1
        assert (image[0] > 2.0).float().mean() <= 0.5
