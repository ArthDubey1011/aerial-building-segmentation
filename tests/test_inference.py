import numpy as np
import pytest

from src.inference.postprocess import (compute_stats, density_heatmap, remove_small_components,
                                       separate_buildings, threshold_mask)
from src.inference.tiling import gaussian_window, predict_tiled


def identity_model(batch):
    """Fake 'model' that returns the first channel unchanged. If tiling+blending is correct, stitching the
    tiles must give back exactly that channel of the original image."""
    return batch[..., 0]


@pytest.mark.parametrize("shape,tile,overlap", [
    ((700, 900, 3), 256, 64),     # not a multiple of the stride
    ((512, 512, 3), 512, 128),    # exactly one tile
    ((300, 200, 3), 512, 128),    # smaller than a tile (needs padding)
    ((1500, 1500, 3), 512, 128),  # the real image size
])
def test_tiling_identity_reconstructs_input(shape, tile, overlap):
    rng = np.random.default_rng(0)
    image = rng.random(shape).astype(np.float32)
    out = predict_tiled(image, identity_model, tile_size=tile, overlap=overlap, batch_size=3)
    assert out.shape == shape[:2]
    np.testing.assert_allclose(out, image[..., 0], atol=1e-5)


def test_gaussian_window_is_positive_and_peaks_in_centre():
    w = gaussian_window(64)
    assert w.min() > 0
    assert w[32, 32] > w[0, 0]


def test_blending_removes_seams_between_disagreeing_tiles():
    """Tiles that disagree (each returns its own constant) must blend smoothly, not jump at tile borders."""
    calls = iter(range(1, 100))
    # Height 100 fits in a single row of tiles, so neighbouring tiles differ by exactly 1 (values 1, 2, 3, ...).
    out = predict_tiled(np.zeros((100, 600, 3), dtype=np.float32),
                        lambda b: np.stack([np.full(b.shape[1:3], float(next(calls))) for _ in b]),
                        tile_size=256, overlap=128, batch_size=1)
    assert out.max() - out.min() > 1.5  # tiles really disagree...
    assert np.abs(np.diff(out, axis=1)).max() < 0.05  # ...but the blend has no jump (a hard stitch would jump by 1)


def _circles(centers, radius, size=(120, 160)):
    yy, xx = np.mgrid[:size[0], :size[1]]
    mask = np.zeros(size, dtype=np.uint8)
    for cy, cx in centers:
        mask |= (((yy - cy) ** 2 + (xx - cx) ** 2) <= radius ** 2).astype(np.uint8)
    return mask


def test_watershed_splits_two_touching_circles():
    mask = _circles([(60, 50), (60, 85)], radius=20)  # overlapping circles (centres 35 px apart, radius 20)
    labels = separate_buildings(mask, h=2.0)
    assert len(np.unique(labels[labels > 0])) == 2


def test_watershed_keeps_single_circle_and_rectangle_whole():
    assert len(np.unique(separate_buildings(_circles([(60, 80)], 25))[_circles([(60, 80)], 25) > 0])) == 1
    rect = np.zeros((100, 200), dtype=np.uint8)
    rect[30:70, 20:180] = 1  # long rectangle: must not be over-segmented
    labels = separate_buildings(rect)
    assert len(np.unique(labels[labels > 0])) == 1


def test_tiny_building_still_counted_and_noise_removed():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:15, 10:15] = 1      # 25 px blob
    mask[50:80, 50:80] = 1      # 900 px building
    cleaned = remove_small_components(mask, min_area=30)
    assert cleaned.sum() == 900
    labels = separate_buildings(mask)  # without cleaning both must still get labels
    assert len(np.unique(labels[labels > 0])) == 2


def test_stats_and_area_percentage():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[0:20, 0:50] = 1  # 1000 px = 10 %
    labels = separate_buildings(mask)
    stats = compute_stats(mask, labels, pixel_size_m=0.5)
    assert stats["building_count"] == 1
    assert abs(stats["built_up_area_pct"] - 10.0) < 1e-6
    assert abs(stats["built_up_area_m2"] - 250.0) < 1e-6  # 1000 px * 0.25 m2/px


def test_threshold_mask():
    assert threshold_mask(np.array([0.2, 0.5, 0.8]), 0.5).tolist() == [0, 0, 1]


def test_heatmap_peaks_where_buildings_cluster():
    labels = np.zeros((256, 256), dtype=np.int32)
    for i, (y, x) in enumerate([(30, 30), (40, 45), (50, 35), (200, 200)], start=1):
        labels[y:y + 6, x:x + 6] = i
    heat = density_heatmap(labels, cell_px=32)
    assert heat.shape == labels.shape
    assert 0.0 <= heat.min() and abs(heat.max() - 1.0) < 1e-6
    assert heat[40, 40] > heat[200, 200]
