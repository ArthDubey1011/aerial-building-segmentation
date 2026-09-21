"""Save side-by-side prediction panels: image | ground truth | prediction."""
from pathlib import Path

import cv2
import numpy as np
import torch

from src.data.dataset import IMAGENET_MEAN, IMAGENET_STD


def denormalize(img_tensor):
    """(3,H,W) normalised tensor -> (H,W,3) uint8 RGB."""
    img = img_tensor.cpu().numpy().transpose(1, 2, 0)
    img = img * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


@torch.no_grad()
def save_predictions(model, dataset, device, out_dir, n=4, threshold=0.5):
    """Write n evenly spaced samples from `dataset` as PNG panels into out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    indices = np.linspace(0, len(dataset) - 1, num=min(n, len(dataset)), dtype=int)
    for k, idx in enumerate(indices):
        image, mask = dataset[int(idx)]
        prob = torch.sigmoid(model(image.unsqueeze(0).to(device)))[0, 0].cpu().numpy()
        rgb = denormalize(image)
        gt = np.repeat((mask[0].numpy() * 255).astype(np.uint8)[..., None], 3, axis=-1)
        pred = np.repeat(((prob > threshold) * 255).astype(np.uint8)[..., None], 3, axis=-1)
        panel = np.concatenate([rgb, gt, pred], axis=1)
        cv2.imwrite(str(out_dir / f"sample_{k}.png"), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
