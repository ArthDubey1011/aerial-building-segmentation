"""One training epoch and one evaluation pass."""
import torch
from tqdm import tqdm

from src.eval.metrics import ConfusionAccumulator


def train_one_epoch(model, loader, loss_fn, optimizer, scaler, device, use_amp):
    model.train()
    total, count = 0.0, 0
    for images, masks in tqdm(loader, desc="train", leave=False):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad(set_to_none=True)
        # Mixed precision: forward in float16 where safe (faster on GPU); GradScaler prevents tiny fp16
        # gradients from underflowing to zero. With use_amp=False (CPU) both are no-ops.
        with torch.autocast(device_type=device.type, enabled=use_amp):
            loss = loss_fn(model(images), masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += loss.item() * images.size(0)
        count += images.size(0)
    return total / count


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, threshold=0.5, use_amp=False):
    """Returns dict(loss, iou, dice) over the whole loader (metrics pooled over all pixels)."""
    model.eval()
    acc = ConfusionAccumulator(threshold)
    total, count = 0.0, 0
    for images, masks in tqdm(loader, desc="eval", leave=False):
        images, masks = images.to(device), masks.to(device)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
        total += loss_fn(logits, masks).item() * images.size(0)
        count += images.size(0)
        acc.update(logits, masks)
    return {"loss": total / count, **acc.compute()}
