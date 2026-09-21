"""Datasets for building segmentation.

Two datasets:
- TrainDataset: random 512x512 crops from the big 1500x1500 images (+ augmentation).
  Crops that are mostly blank (pure white/black no-data borders) are re-drawn.
- TileDataset: deterministic grid of tiles covering each image, used for validation/testing.
"""
import random
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

# The ResNet34 encoder was pretrained on ImageNet with these statistics, so we must normalise the same way.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_SUFFIXES = {".png", ".tif", ".tiff", ".jpg"}


def list_pairs(image_dir, mask_dir):
    """Match images and masks by file name. Returns a sorted list of (image_path, mask_path)."""
    image_dir, mask_dir = Path(image_dir), Path(mask_dir)
    pairs = []
    for img in sorted(image_dir.iterdir()):
        if img.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        mask = mask_dir / img.name
        if not mask.exists():
            raise FileNotFoundError(f"No mask for {img.name} in {mask_dir}")
        pairs.append((img, mask))
    if not pairs:
        raise FileNotFoundError(f"No images found in {image_dir}")
    return pairs


def read_image(path):
    """Read as RGB uint8 (OpenCV loads BGR, so convert)."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Could not read image {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_mask(path, threshold=127):
    """Read a mask and return a binary uint8 array (1 = building).

    The masks are 3-channel PNGs with values {0, 255}. Reading as grayscale collapses the 3 identical
    channels into one; thresholding (instead of `== 255`) is robust to any compression artefacts.
    """
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise IOError(f"Could not read mask {path}")
    return (mask > threshold).astype(np.uint8)


def blank_fraction(img):
    """Fraction of pixels that are pure white or pure black in all channels (= 'no data' areas)."""
    white = (img == 255).all(axis=-1)
    black = (img == 0).all(axis=-1)
    return float((white | black).mean())


def _normalize_and_tensor():
    return [A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]


def train_transform():
    """Geometric augmentations are safe for aerial imagery (no fixed 'up'), so flips + 90 degree rotations."""
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            # Light colour jitter: imagery differs in season/lighting between tiles.
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02, p=0.5),
            *_normalize_and_tensor(),
        ]
    )


def eval_transform():
    return A.Compose(_normalize_and_tensor())


def _to_sample(transform, image, mask):
    out = transform(image=image, mask=mask)
    # Mask -> float tensor of shape (1, H, W) with values 0.0/1.0 (what BCE/Dice expect).
    return out["image"], out["mask"].float().unsqueeze(0)


class TrainDataset(Dataset):
    """Random crops. `length` is the number of crops per epoch (not the number of images)."""

    def __init__(self, pairs, crop_size, length, blank_max_fraction=0.5, max_tries=10,
                 mask_threshold=127, augment=True):
        self.pairs = pairs
        self.crop_size = crop_size
        self.length = length
        self.blank_max_fraction = blank_max_fraction
        self.max_tries = max_tries
        self.mask_threshold = mask_threshold
        self.transform = train_transform() if augment else eval_transform()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # `random` (not numpy) is used because PyTorch seeds it differently in each DataLoader worker.
        img_path, mask_path = random.choice(self.pairs)
        image = read_image(img_path)
        mask = read_mask(mask_path, self.mask_threshold)
        h, w = mask.shape
        c = self.crop_size
        if h < c or w < c:
            raise ValueError(f"Image {img_path.name} ({h}x{w}) is smaller than crop_size {c}")

        for _ in range(self.max_tries):
            y = random.randint(0, h - c)
            x = random.randint(0, w - c)
            img_crop = image[y:y + c, x:x + c]
            if blank_fraction(img_crop) <= self.blank_max_fraction:
                break  # good crop; if we run out of tries we just keep the last one
        return _to_sample(self.transform, img_crop, mask[y:y + c, x:x + c])


def tile_starts(length, tile):
    """Start offsets of tiles covering [0, length). The last tile is shifted back to end exactly at `length`,
    so every tile is full size (small overlap with its neighbour instead of padding)."""
    if length <= tile:
        return [0]
    return list(range(0, length - tile, tile)) + [length - tile]


class TileDataset(Dataset):
    """Deterministic grid of tiles over each image, for validation/test.

    Note: the last row/column of tiles overlaps its neighbour (1500 is not a multiple of 512), so a few
    pixels are counted twice in the metrics. This is a small, consistent bias and keeps the code simple.
    Proper overlap-blended full-image inference lives in src/inference/tiling.py.
    """

    def __init__(self, pairs, tile_size, mask_threshold=127):
        self.pairs = pairs
        self.tile_size = tile_size
        self.mask_threshold = mask_threshold
        self.transform = eval_transform()
        self.tiles = []  # (pair_index, y, x)
        for i, (_, mask_path) in enumerate(pairs):
            h, w = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE).shape
            for y in tile_starts(h, tile_size):
                for x in tile_starts(w, tile_size):
                    self.tiles.append((i, y, x))
        self._cache_idx = None  # single-slot cache: tiles of one image are consecutive, so read it once
        self._cache = None

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        i, y, x = self.tiles[idx]
        if self._cache_idx != i:
            img_path, mask_path = self.pairs[i]
            self._cache = (read_image(img_path), read_mask(mask_path, self.mask_threshold))
            self._cache_idx = i
        image, mask = self._cache
        t = self.tile_size
        return _to_sample(self.transform, image[y:y + t, x:x + t], mask[y:y + t, x:x + t])


def build_datasets(cfg, debug=False):
    """Create (train, val, test) datasets from the config. Paths are relative to cfg['data']['root']."""
    d = cfg["data"]
    root = Path(d["root"])
    train_pairs = list_pairs(root / d["train_images"], root / d["train_masks"])
    val_pairs = list_pairs(root / d["val_images"], root / d["val_masks"])
    test_pairs = list_pairs(root / d["test_images"], root / d["test_masks"])
    if debug:  # keep --debug tiny so it runs in < 2 minutes on CPU
        val_pairs, test_pairs = val_pairs[:1], test_pairs[:1]

    train_ds = TrainDataset(
        train_pairs, d["crop_size"], d["crops_per_epoch"], d["blank_max_fraction"],
        d["max_resample_tries"], d["mask_threshold"],
    )
    val_ds = TileDataset(val_pairs, d["crop_size"], d["mask_threshold"])
    test_ds = TileDataset(test_pairs, d["crop_size"], d["mask_threshold"])
    return train_ds, val_ds, test_ds
