"""IoU and Dice, accumulated over the WHOLE dataset.

Averaging per-batch IoU is biased (a batch with few buildings counts as much as one with many, and empty
batches give 0/0). Instead we sum true positives / false positives / false negatives over all pixels of all
batches and compute IoU and Dice once at the end.
"""
import torch


class ConfusionAccumulator:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.tp = 0
        self.fp = 0
        self.fn = 0

    @torch.no_grad()
    def update(self, logits, targets):
        preds = torch.sigmoid(logits.float()) > self.threshold
        targets = targets > 0.5
        self.tp += int((preds & targets).sum())
        self.fp += int((preds & ~targets).sum())
        self.fn += int((~preds & targets).sum())

    def compute(self, eps=1e-7):
        iou = self.tp / (self.tp + self.fp + self.fn + eps)
        dice = 2 * self.tp / (2 * self.tp + self.fp + self.fn + eps)
        return {"iou": iou, "dice": dice}
