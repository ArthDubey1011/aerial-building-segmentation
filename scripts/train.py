"""Train a U-Net for building segmentation.

Usage (PowerShell):
    python scripts/train.py --config configs/local.yaml --debug          # tiny CPU smoke test
    python scripts/train.py --config configs/kaggle_bce_dice.yaml        # full run (Kaggle GPU)
    python scripts/train.py --config configs/kaggle_bce_dice.yaml --resume   # continue after a crash
"""
import argparse
import csv
import json
import sys
from pathlib import Path

# Make `import src...` work when running `python scripts/train.py` (Windows, Kaggle, anywhere).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader

from src.data.dataset import build_datasets
from src.eval.visualize import save_predictions
from src.losses.losses import build_loss
from src.models.unet import build_model
from src.train.engine import evaluate, train_one_epoch
from src.train.utils import get_device, load_config, save_config, set_seed

METRIC_FIELDS = ["epoch", "train_loss", "val_loss", "val_iou", "val_dice", "lr"]


def apply_debug_overrides(cfg):
    """--debug: 2 epochs, 4 crops per epoch, batch 2, single-process loading (1 val/test image: see build_datasets)."""
    cfg["experiment_name"] += "_debug"
    cfg["train"].update(epochs=2, batch_size=2, num_workers=0, mixed_precision=False)
    cfg["data"]["crops_per_epoch"] = 4
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--resume", action="store_true", help="continue from results/<name>/checkpoints/last.pth")
    parser.add_argument("--data-root", help="override data.root from the config (Kaggle mounts datasets at varying paths)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.data_root:
        cfg["data"]["root"] = args.data_root
    if args.debug:
        cfg = apply_debug_overrides(cfg)
    set_seed(cfg["seed"])
    device = get_device()
    use_amp = bool(cfg["train"]["mixed_precision"]) and device.type == "cuda"
    print(f"device={device} amp={use_amp} experiment={cfg['experiment_name']}")

    out_dir = Path(cfg["output"]["results_dir"]) / cfg["experiment_name"]
    ckpt_dir = out_dir / "checkpoints"  # gitignored
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out_dir / "config.yaml")

    train_ds, val_ds, test_ds = build_datasets(cfg, debug=args.debug)
    t = cfg["train"]
    loader_kw = dict(batch_size=t["batch_size"], num_workers=t["num_workers"], pin_memory=device.type == "cuda")
    train_loader = DataLoader(train_ds, shuffle=False, **loader_kw)  # dataset already samples randomly
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kw)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kw)

    model = build_model(cfg).to(device)
    loss_fn = build_loss(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=t["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t["epochs"])
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    start_epoch, best_iou = 0, -1.0
    metrics_path = out_dir / "metrics.csv"
    last_path, best_path = ckpt_dir / "last.pth", ckpt_dir / "best.pth"
    if args.resume and last_path.exists():
        ckpt = torch.load(last_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch, best_iou = ckpt["epoch"] + 1, ckpt["best_iou"]
        print(f"Resumed from epoch {start_epoch}, best val IoU so far {best_iou:.4f}")
    elif metrics_path.exists():
        metrics_path.unlink()  # fresh run: don't append to an old log

    if not metrics_path.exists():
        with open(metrics_path, "w", newline="") as f:
            csv.writer(f).writerow(METRIC_FIELDS)

    for epoch in range(start_epoch, t["epochs"]):
        lr = optimizer.param_groups[0]["lr"]
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, scaler, device, use_amp)
        val = evaluate(model, val_loader, loss_fn, device, t["threshold"], use_amp)
        scheduler.step()
        print(f"epoch {epoch + 1}/{t['epochs']} train_loss={train_loss:.4f} val_loss={val['loss']:.4f} "
              f"val_iou={val['iou']:.4f} val_dice={val['dice']:.4f}")

        with open(metrics_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, train_loss, val["loss"], val["iou"], val["dice"], lr])

        if val["iou"] > best_iou:
            best_iou = val["iou"]
            torch.save(model.state_dict(), best_path)  # weights only: what inference needs
        # Full training state every epoch so a stopped Kaggle session can resume.
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": epoch, "best_iou": best_iou}, last_path)

    # Final: reload the best model (by val IoU), evaluate once on the test split, save sample predictions.
    model.load_state_dict(torch.load(best_path, map_location=device))
    test = evaluate(model, test_loader, loss_fn, device, t["threshold"], use_amp)
    print(f"TEST iou={test['iou']:.4f} dice={test['dice']:.4f} (best val iou {best_iou:.4f})")
    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump({"experiment": cfg["experiment_name"], "best_val_iou": best_iou, **test}, f, indent=2)
    save_predictions(model, test_ds, device, out_dir / "samples", n=6, threshold=t["threshold"])


if __name__ == "__main__":
    main()
