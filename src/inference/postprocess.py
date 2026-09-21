"""Turn a probability map into building instances and statistics.

Steps: threshold -> remove tiny blobs -> distance transform + watershed (split touching buildings) ->
count / area % / per-building areas -> density heatmap.
"""
import math

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import h_maxima
from skimage.segmentation import watershed


def threshold_mask(prob, threshold=0.5):
    return (prob > threshold).astype(np.uint8)


def remove_small_components(mask, min_area):
    """Drop connected blobs smaller than min_area pixels (typically false-positive speckle)."""
    n, comp, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area  # index 0 is the background
    return keep[comp].astype(np.uint8)


def separate_buildings(mask, h=2.0):
    """Split touching buildings with a marker-based watershed. Returns an int label image (0 = background).

    Idea: the distance transform gives each foreground pixel its distance to the nearest background. Each
    building has a "peak" at its centre, and two touching buildings are separated by a "valley" (the narrow
    neck). Flooding the inverted distance map from one marker per peak makes the waters meet at the neck.
    h-maxima keeps only peaks that rise at least `h` above their surroundings, which avoids splitting one
    building with a slightly bumpy outline into several pieces (over-segmentation).
    """
    mask = mask.astype(bool)
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    dist = cv2.GaussianBlur(dist, (0, 0), 1.0)  # smooth pixel-level jitter
    markers, _ = ndi.label(h_maxima(dist, h))
    labels = watershed(-dist, markers, mask=mask)

    # Small blobs whose peak is lower than h get no marker; give each of them its own label so they still count.
    orphans, n_orphans = ndi.label(mask & (labels == 0))
    labels[orphans > 0] = orphans[orphans > 0] + labels.max()
    return labels


def compute_stats(mask, labels, pixel_size_m=1.0):
    """Building count, built-up area %, and per-building areas (pixels and m2)."""
    ids, areas_px = np.unique(labels[labels > 0], return_counts=True)
    pixel_area = pixel_size_m ** 2
    return {
        "building_count": int(len(ids)),
        "built_up_area_pct": float(mask.mean() * 100.0),
        "built_up_area_m2": float(mask.sum() * pixel_area),
        "mean_building_area_m2": float(areas_px.mean() * pixel_area) if len(ids) else 0.0,
        "median_building_area_m2": float(np.median(areas_px) * pixel_area) if len(ids) else 0.0,
        "building_areas_px": [int(a) for a in areas_px],
    }


def density_heatmap(labels, cell_px=64):
    """Building density in [0,1]: count building centroids per grid cell, smooth, upsample to image size."""
    h, w = labels.shape
    counts = np.zeros((math.ceil(h / cell_px), math.ceil(w / cell_px)), dtype=np.float32)
    ids = np.unique(labels[labels > 0])
    if len(ids):
        centroids = np.array(ndi.center_of_mass(labels > 0, labels, ids))  # (n, 2) as (y, x)
        np.add.at(counts, ((centroids[:, 0] // cell_px).astype(int), (centroids[:, 1] // cell_px).astype(int)), 1)
    counts = cv2.GaussianBlur(counts, (0, 0), 1.0)  # smooth over neighbouring cells (1 cell std)
    heat = cv2.resize(counts, (w, h), interpolation=cv2.INTER_CUBIC)  # smooth upsampling to full size
    heat = np.clip(heat, 0, None)
    return heat / heat.max() if heat.max() > 0 else heat

