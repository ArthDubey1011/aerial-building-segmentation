import torch

from src.eval.metrics import ConfusionAccumulator
from src.losses.losses import BCEDiceLoss, DiceLoss, FocalDiceLoss, FocalLoss


def _target():
    t = torch.zeros(1, 1, 8, 8)
    t[..., 2:6, 2:6] = 1.0
    return t


def _perfect_logits(target):
    return (target * 2 - 1) * 20.0  # +20 where building, -20 elsewhere -> probability ~1 / ~0


def test_perfect_prediction_gives_near_zero_loss():
    t = _target()
    logits = _perfect_logits(t)
    for loss in (DiceLoss(), FocalLoss(), BCEDiceLoss(), FocalDiceLoss()):
        assert loss(logits, t).item() < 1e-3


def test_wrong_prediction_gives_large_loss():
    t = _target()
    logits = -_perfect_logits(t)  # exactly inverted
    for loss in (DiceLoss(), FocalLoss(), BCEDiceLoss(), FocalDiceLoss()):
        assert loss(logits, t).item() > 0.9


def test_focal_with_gamma_zero_equals_bce():
    t = _target()
    logits = torch.randn(1, 1, 8, 8)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, t)
    assert torch.allclose(FocalLoss(gamma=0.0)(logits, t), bce, atol=1e-6)


def test_loss_handles_empty_target():
    t = torch.zeros(1, 1, 8, 8)
    logits = torch.full((1, 1, 8, 8), -20.0)
    assert DiceLoss()(logits, t).item() < 1e-3  # smoothing avoids 0/0


def test_metrics_known_example():
    # target: 4 building pixels. prediction: 3 correct + 1 false positive -> tp=3 fp=1 fn=1
    target = torch.zeros(1, 1, 4, 4)
    target[0, 0, 0, :] = 1
    pred = torch.zeros(1, 1, 4, 4)
    pred[0, 0, 0, :3] = 1
    pred[0, 0, 1, 0] = 1
    acc = ConfusionAccumulator()
    acc.update((pred * 2 - 1) * 10, target)
    m = acc.compute()
    assert abs(m["iou"] - 3 / 5) < 1e-6
    assert abs(m["dice"] - 6 / 8) < 1e-6


def test_metrics_pool_over_batches():
    """Pooled IoU over two updates must equal IoU of the concatenated data (not the mean of per-batch IoUs)."""
    t1 = torch.zeros(1, 1, 4, 4); t1[0, 0, 0, :] = 1
    t2 = torch.zeros(1, 1, 4, 4)
    l1 = (t1 * 2 - 1) * 10           # perfect on batch 1
    l2 = torch.full((1, 1, 4, 4), 10.0)  # predicts everything as building on empty batch 2
    acc = ConfusionAccumulator()
    acc.update(l1, t1)
    acc.update(l2, t2)
    # tp=4, fp=16, fn=0 -> IoU 4/20
    assert abs(acc.compute()["iou"] - 0.2) < 1e-6
