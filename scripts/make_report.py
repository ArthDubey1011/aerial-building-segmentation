"""Build the README material from finished experiments in results/.

Produces (in docs/figures/ and docs/):
  - results_table.md          comparison table (best val IoU, test IoU/Dice per experiment; copy into the README)
  - training_curves.png       train/val loss and val IoU per epoch
  - good_*.png / failure_*.png  panels [image | ground truth | prediction | errors] for the best experiment
  - per_image_<exp>.csv       IoU/Dice of every full test image (tiled + blended inference)

Usage (PowerShell):
    python scripts/make_report.py --config configs/local.yaml --experiments kaggle_bce_dice kaggle_focal_dice
    python scripts/make_report.py --config configs/local.yaml --debug
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import matplotlib

matplotlib.use("Agg")  # no display needed (works on Kaggle/CI too)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.dataset import list_pairs, read_image, read_mask
from src.inference.pipeline import analyze_image, load_model
from src.train.utils import get_device, load_config


def error_map(pred, gt):
    """Green = correct building (TP), red = false alarm (FP), blue = missed building (FN), black = background."""
    out = np.zeros((*gt.shape, 3), dtype=np.uint8)
    out[(pred == 1) & (gt == 1)] = (0, 200, 0)
    out[(pred == 1) & (gt == 0)] = (230, 0, 0)
    out[(pred == 0) & (gt == 1)] = (0, 90, 255)
    return out


def save_panel(path, image, gt, pred, title, width=420):
    """Image | ground truth | prediction | error map, resized to `width` each so README figures stay small."""
    gt_rgb = np.repeat(gt[..., None] * 255, 3, axis=-1).astype(np.uint8)
    pr_rgb = np.repeat(pred[..., None] * 255, 3, axis=-1).astype(np.uint8)
    tiles = [image, gt_rgb, pr_rgb, error_map(pred, gt)]
    h = int(image.shape[0] * width / image.shape[1])
    tiles = [cv2.resize(t, (width, h), interpolation=cv2.INTER_AREA) for t in tiles]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.4))
    for ax, t, name in zip(axes, tiles, ["Image", "Ground truth", "Prediction", "Errors (green TP, red FP, blue FN)"]):
        ax.imshow(t)
        ax.set_title(name, fontsize=9)
        ax.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def scores(pred, gt, eps=1e-7):
    tp = int(((pred == 1) & (gt == 1)).sum())
    fp = int(((pred == 1) & (gt == 0)).sum())
    fn = int(((pred == 0) & (gt == 1)).sum())
    return {"iou": tp / (tp + fp + fn + eps), "dice": 2 * tp / (2 * tp + fp + fn + eps), "tp": tp, "fp": fp, "fn": fn}


def evaluate_full_images(exp, cfg, pairs, device, results_dir, keep_images=False):
    """Blended tiled inference on every test image; returns per-image rows (+ pooled metrics, + images if kept)."""
    ckpt = results_dir / exp / "checkpoints" / "best.pth"
    model = load_model(ckpt, cfg, device)
    thr = cfg["infer"]["threshold"]
    rows, cache = [], {}
    for img_path, mask_path in pairs:
        image = read_image(img_path)
        gt = read_mask(mask_path, cfg["data"]["mask_threshold"])
        result = analyze_image(image, model, device, cfg["infer"])
        pred = (result["prob"] > thr).astype(np.uint8)  # raw threshold, no post-processing, for a fair metric
        rows.append({"image": img_path.name, **scores(pred, gt), "gt_area_pct": 100 * gt.mean()})
        if keep_images:
            cache[img_path.name] = (image, gt, pred)
        print(f"  {exp} {img_path.name}: IoU={rows[-1]['iou']:.3f}", flush=True)
    tp, fp, fn = (sum(r[k] for r in rows) for k in ("tp", "fp", "fn"))
    pooled = {"full_iou": tp / (tp + fp + fn), "full_dice": 2 * tp / (2 * tp + fp + fn)}
    return rows, pooled, cache


def plot_curves(results_dir, experiments, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    for exp in experiments:
        df = pd.read_csv(results_dir / exp / "metrics.csv")
        axes[0].plot(df["epoch"], df["train_loss"], label=exp)
        axes[1].plot(df["epoch"], df["val_loss"], label=exp)
        axes[2].plot(df["epoch"], df["val_iou"], label=exp)
    for ax, t in zip(axes, ["Train loss", "Val loss", "Val IoU"]):
        ax.set_title(t)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("Note: losses differ in scale between experiments (BCE+Dice vs Focal+Dice); compare IoU, not loss.",
                 fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--experiments", nargs="+", default=None)
    parser.add_argument("--test-images", default="data/test/images")
    parser.add_argument("--test-masks", default="data/test/masks")
    parser.add_argument("--out", default="docs/figures")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    results_dir = Path(cfg["output"]["results_dir"])
    out = Path(args.out)
    if args.debug:
        experiments = ["local_debug"]
        pairs = list_pairs(cfg["data"]["root"] + "/" + cfg["data"]["test_images"],
                           cfg["data"]["root"] + "/" + cfg["data"]["test_masks"])[:2]
        out = Path("results/report_debug")
        cfg["infer"]["tile_size"], cfg["infer"]["overlap"] = 512, 128
    else:
        experiments = args.experiments or sorted(p.name for p in results_dir.glob("kaggle_*") if p.is_dir())
        pairs = list_pairs(args.test_images, args.test_masks)
    out.mkdir(parents=True, exist_ok=True)
    device = get_device()
    print(f"experiments={experiments} test images={len(pairs)} device={device}")

    # 1) table: tile-based numbers logged by train.py + full-image numbers from blended inference
    table, per_exp_rows, per_exp_cache = [], {}, {}
    for exp in experiments:
        test = json.load(open(results_dir / exp / "test_metrics.json"))
        rows, pooled, cache = evaluate_full_images(exp, cfg, pairs, device, results_dir, keep_images=True)
        per_exp_rows[exp], per_exp_cache[exp] = rows, cache
        table.append({"experiment": exp, "best_val_iou": test["best_val_iou"], "test_iou_tiles": test["iou"],
                      "test_dice_tiles": test["dice"], "test_iou_full": pooled["full_iou"],
                      "test_dice_full": pooled["full_dice"]})
        with open(out / f"per_image_{exp}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    df = pd.DataFrame(table)
    md = ["| Experiment | Best val IoU | Test IoU (tiles) | Test Dice (tiles) | Test IoU (full, blended) | Test Dice (full, blended) |",
          "|---|---|---|---|---|---|"]
    for r in table:
        md.append(f"| {r['experiment']} | {r['best_val_iou']:.4f} | {r['test_iou_tiles']:.4f} | {r['test_dice_tiles']:.4f} "
                  f"| {r['test_iou_full']:.4f} | {r['test_dice_full']:.4f} |")
    (out / "results_table.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))

    # 2) curves
    plot_curves(results_dir, experiments, out / "training_curves.png")

    # 3) good + failure panels from the experiment with the best full-image test IoU
    best = df.sort_values("test_iou_full", ascending=False).iloc[0]["experiment"]
    ranked = sorted(per_exp_rows[best], key=lambda r: r["iou"], reverse=True)
    n_good = min(6, len(ranked))
    for k, r in enumerate(ranked[:n_good]):
        image, gt, pred = per_exp_cache[best][r["image"]]
        save_panel(out / f"good_{k}.png", image, gt, pred, f"{best}: {r['image']}  IoU={r['iou']:.3f}")
    for k, r in enumerate(ranked[::-1][:2]):
        image, gt, pred = per_exp_cache[best][r["image"]]
        kind = "false positives dominate" if r["fp"] > r["fn"] else "false negatives (missed buildings) dominate"
        save_panel(out / f"failure_{k}.png", image, gt, pred,
                   f"{best}: {r['image']}  IoU={r['iou']:.3f}  ({kind}; FP={r['fp']}, FN={r['fn']})")
    print(f"best experiment: {best}. Figures written to {out}")


if __name__ == "__main__":
    main()
