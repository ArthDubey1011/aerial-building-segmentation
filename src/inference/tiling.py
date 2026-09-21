"""Sliding-window inference for images larger than the model's tile size.

Why tiles: a 1500x1500 image doesn't fit in memory as one batch on small GPUs, and the model was trained on
512x512 crops. Why overlap + blending: predictions are least reliable at a tile's border (less context), and
hard tile boundaries leave visible seams. We overlap tiles and average predictions with a Gaussian weight that
is highest at the tile centre, so each pixel is dominated by the tiles where it is far from the border.
"""
import math

import numpy as np
import torch

from src.data.dataset import IMAGENET_MEAN, IMAGENET_STD


def gaussian_window(size, sigma_frac=0.15):
    """2D weight window, 1 at the centre and ~0.005 (1D) at the tile edge. Never exactly 0, so every pixel has
    total weight > 0. The edge weight must be near 0: a tile that suddenly starts contributing with a large
    weight where its neighbour ends would itself create a visible seam (a wider sigma of 0.25 did, in a test)."""
    x = np.arange(size) - (size - 1) / 2
    g = np.exp(-0.5 * (x / (sigma_frac * size)) ** 2)
    return np.outer(g, g).astype(np.float32)


def _grid(length, tile, stride):
    """Number of tiles needed along one axis and the padded length that makes the grid fit exactly."""
    n = 1 if length <= tile else math.ceil((length - tile) / stride) + 1
    return n, tile + (n - 1) * stride


def predict_tiled(image, predict_fn, tile_size=512, overlap=128, batch_size=4):
    """Run `predict_fn` on overlapping tiles and blend the results into one full-size probability map.

    image:      (H, W, C) array.
    predict_fn: takes a batch (B, tile, tile, C) and returns probabilities (B, tile, tile). Keeping this a plain
                function makes the stitching testable with an "identity model" (no neural network needed).
    Returns:    (H, W) float32 array.
    """
    stride = tile_size - overlap
    if stride <= 0:
        raise ValueError("overlap must be smaller than tile_size")
    h, w = image.shape[:2]

    # Reflect-pad: `margin` on every side so border pixels also get context and a full-weight tile, plus extra
    # bottom/right padding so the tile grid fits exactly. Reflection (mirroring) avoids fake black borders.
    margin = overlap // 2
    ny, ph = _grid(h + 2 * margin, tile_size, stride)
    nx, pw = _grid(w + 2 * margin, tile_size, stride)
    padded = np.pad(image, ((margin, ph - h - margin), (margin, pw - w - margin), (0, 0)), mode="reflect")

    window = gaussian_window(tile_size)
    acc = np.zeros((ph, pw), dtype=np.float32)      # sum of weight * prediction
    weights = np.zeros((ph, pw), dtype=np.float32)  # sum of weights
    positions = [(iy * stride, ix * stride) for iy in range(ny) for ix in range(nx)]

    for start in range(0, len(positions), batch_size):
        chunk = positions[start:start + batch_size]
        batch = np.stack([padded[y:y + tile_size, x:x + tile_size] for y, x in chunk])
        preds = predict_fn(batch)
        for (y, x), pred in zip(chunk, preds):
            acc[y:y + tile_size, x:x + tile_size] += pred * window
            weights[y:y + tile_size, x:x + tile_size] += window

    # Weighted average; crop the padding away.
    return (acc / weights)[margin:margin + h, margin:margin + w]


def make_model_predict_fn(model, device):
    """Wrap a torch model as a predict_fn: uint8 RGB tiles in -> sigmoid probabilities out."""
    model.eval()
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)

    @torch.no_grad()
    def predict_fn(batch):
        x = (batch.astype(np.float32) / 255.0 - mean) / std
        x = torch.from_numpy(x.transpose(0, 3, 1, 2)).to(device)
        return torch.sigmoid(model(x))[:, 0].cpu().numpy()

    return predict_fn
