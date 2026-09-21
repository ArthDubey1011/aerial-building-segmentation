"""Overlays for visualising predictions. All functions take/return RGB uint8 images."""
import cv2
import numpy as np
from skimage.segmentation import find_boundaries


def mask_overlay(image, mask, color=(255, 0, 0), alpha=0.4):
    out = image.astype(np.float32)
    m = mask.astype(bool)
    out[m] = (1 - alpha) * out[m] + alpha * np.array(color, dtype=np.float32)
    return out.astype(np.uint8)


def instance_overlay(image, labels, alpha=0.5):
    """Each building gets its own random colour (so touching buildings visibly separate) + white outlines."""
    rng = np.random.default_rng(0)
    colors = rng.integers(60, 255, size=(labels.max() + 1, 3)).astype(np.float32)
    out = image.astype(np.float32)
    m = labels > 0
    out[m] = (1 - alpha) * out[m] + alpha * colors[labels[m]]
    out[find_boundaries(labels, mode="inner")] = 255
    return out.astype(np.uint8)


def heatmap_overlay(image, heat, alpha=0.5):
    """Blend a [0,1] density map (JET colormap) over the image."""
    colored = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(image, 1 - alpha, colored, alpha, 0)
