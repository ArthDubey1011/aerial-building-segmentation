r"""Run tiled inference on full-size image(s): saves mask, instance overlay, heatmap overlay and stats JSON.

Usage (PowerShell):
    python scripts/infer.py --config configs/local.yaml --image path\to\image.png
    python scripts/infer.py --config configs/local.yaml --image path\to\folder --checkpoint results\x\checkpoints\best.pth
    python scripts/infer.py --config configs/local.yaml --debug     # 1 sample image (cropped), debug checkpoint
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2

from src.data.dataset import IMAGE_SUFFIXES, read_image
from src.inference.pipeline import analyze_image, load_model
from src.train.utils import get_device, load_config


def save_rgb(path, image):
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--image", help="image file or folder of images")
    parser.add_argument("--checkpoint", help="weights file (default: infer.checkpoint from the config)")
    parser.add_argument("--out", help="output folder (default: <results_dir>/infer)")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    infer_cfg = cfg["infer"]
    device = get_device()

    if args.debug:
        checkpoint = Path(args.checkpoint or infer_cfg["debug_checkpoint"])
        image_paths = sorted((Path(cfg["data"]["root"]) / cfg["data"]["test_images"]).iterdir())[:1]
        out_dir = Path(args.out or Path(cfg["output"]["results_dir"]) / "infer_debug")
    else:
        if not args.image:
            parser.error("--image is required (unless --debug)")
        checkpoint = Path(args.checkpoint or infer_cfg["checkpoint"])
        src = Path(args.image)
        image_paths = sorted(p for p in src.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES) if src.is_dir() else [src]
        out_dir = Path(args.out or Path(cfg["output"]["results_dir"]) / "infer")

    model = load_model(checkpoint, cfg, device)
    print(f"device={device} checkpoint={checkpoint} images={len(image_paths)}")

    for path in image_paths:
        image = read_image(path)
        if args.debug:
            image = image[:1024, :1024]  # keep the CPU smoke test fast (9 tiles instead of 16)
        result = analyze_image(image, model, device, infer_cfg)

        folder = out_dir / path.stem
        folder.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(folder / "mask.png"), result["mask"] * 255)
        save_rgb(folder / "overlay_instances.png", result["instance_overlay"])
        save_rgb(folder / "overlay_heatmap.png", result["heatmap_overlay"])
        save_rgb(folder / "overlay_mask.png", result["mask_overlay"])
        stats = {k: v for k, v in result["stats"].items() if k != "building_areas_px"}
        with open(folder / "stats.json", "w") as f:
            json.dump({"image": path.name, **result["stats"]}, f, indent=2)
        print(path.name, stats)


if __name__ == "__main__":
    main()
