"""Segmentation losses. All take raw logits (B,1,H,W) and float targets in {0,1} of the same shape.

Why combine a pixel-wise loss with Dice? Buildings are ~5-10% of pixels (class imbalance). BCE alone is
dominated by the easy background. Dice measures region overlap (like the F1 score) so it directly optimises
what we evaluate, but its gradients are noisy early on; BCE/Focal stabilise it.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """1 - soft Dice, computed over the whole batch (not per image, so empty images don't blow up)."""

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth  # avoids 0/0 when there are no building pixels

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits.float())  # .float(): stay accurate under mixed precision
        inter = (probs * targets).sum()
        denom = probs.sum() + targets.sum()
        return 1.0 - (2.0 * inter + self.smooth) / (denom + self.smooth)


class FocalLoss(nn.Module):
    """Binary focal loss: BCE * (1 - p_t)^gamma. Down-weights easy pixels (p_t near 1) so training focuses on
    hard ones (building edges, small buildings). gamma=0 recovers plain BCE. No alpha term: Dice already
    handles the class imbalance."""

    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits.float(), targets, reduction="none")
        p_t = torch.exp(-bce)  # probability assigned to the correct class
        return ((1.0 - p_t) ** self.gamma * bce).mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=1.0, dice_weight=1.0):
        super().__init__()
        self.bce_weight, self.dice_weight = bce_weight, dice_weight
        self.dice = DiceLoss()

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits.float(), targets)
        return self.bce_weight * bce + self.dice_weight * self.dice(logits, targets)


class FocalDiceLoss(nn.Module):
    def __init__(self, focal_weight=1.0, dice_weight=1.0, gamma=2.0):
        super().__init__()
        self.focal_weight, self.dice_weight = focal_weight, dice_weight
        self.focal = FocalLoss(gamma)
        self.dice = DiceLoss()

    def forward(self, logits, targets):
        return self.focal_weight * self.focal(logits, targets) + self.dice_weight * self.dice(logits, targets)


def build_loss(cfg):
    name = cfg["loss"]["name"]
    if name == "bce_dice":
        return BCEDiceLoss()
    if name == "focal_dice":
        return FocalDiceLoss()
    raise ValueError(f"Unknown loss '{name}' (expected bce_dice or focal_dice)")
