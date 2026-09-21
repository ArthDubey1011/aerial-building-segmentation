"""Publish the model + Space to Hugging Face. DRY RUN by default: nothing is uploaded without --publish.

Needs a write token in the environment (never put it in a file or in git):
    $env:HF_TOKEN = "hf_..."          # PowerShell, get it at https://huggingface.co/settings/tokens

Usage:
    python scripts/publish_space.py --user YOUR_HF_USERNAME                 # dry run: shows what would happen
    python scripts/publish_space.py --user YOUR_HF_USERNAME --publish       # creates PUBLIC repos and uploads
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MODEL_CARD = """---
library_name: pytorch
tags:
- image-segmentation
- remote-sensing
---
# Aerial building segmentation (U-Net, ResNet34)

U-Net with an ImageNet-pretrained ResNet34 encoder (segmentation-models-pytorch), trained to segment buildings
in aerial images at ~1 m/pixel. Weights only (`best.pth`, state dict); build the model with
`smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=1)` and apply a sigmoid to the logits.

- Data: Massachusetts Buildings dataset (137 train / 4 val / 10 test images, 1500x1500).
- Loss: BCE + Dice, 40 epochs x 400 random 512x512 crops, AdamW, one seed.
- Test (full images, tiled inference): IoU 0.695, Dice 0.820 (Focal+Dice: 0.690 / 0.817, effectively a tie).
- Limitations: one region only, no cross-city evaluation; 1 m/pixel imagery; single image, no change detection.

Code, training and failure analysis: https://github.com/ArthDubey1011/urban-growth-satellite
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", required=True)
    parser.add_argument("--checkpoint", default=str(ROOT / "results" / "kaggle_bce_dice" / "checkpoints" / "best.pth"))
    parser.add_argument("--publish", action="store_true", help="actually create repos and upload")
    parser.add_argument("--private", action="store_true", help="create private repos instead of public")
    args = parser.parse_args()

    model_id, space_id = f"{args.user}/aerial-building-unet", f"{args.user}/aerial-building-segmentation"
    space_dir, ckpt = ROOT / "hf_space", Path(args.checkpoint)
    for p in (space_dir / "space_app.py", ckpt):
        if not p.exists():
            sys.exit(f"Missing {p}. Run scripts/build_space.py first (and train / extract results).")

    visibility = "PRIVATE" if args.private else "PUBLIC"
    print(f"Would create {visibility} model repo  {model_id}  <- {ckpt.name} ({ckpt.stat().st_size / 1e6:.0f} MB) + model card")
    print(f"Would create {visibility} Space       {space_id}  <- {space_dir}")
    if not args.publish:
        print("Dry run only. Re-run with --publish to upload.")
        return
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN is not set.")

    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.create_repo(model_id, repo_type="model", private=args.private, exist_ok=True)
    api.upload_file(path_or_fileobj=str(ckpt), path_in_repo="best.pth", repo_id=model_id)
    api.upload_file(path_or_fileobj=MODEL_CARD.encode("utf-8"), path_in_repo="README.md", repo_id=model_id)
    api.create_repo(space_id, repo_type="space", space_sdk="gradio", private=args.private, exist_ok=True)
    # The checkpoint lives in the model repo (downloaded at start-up), so keep it and caches out of the Space.
    api.upload_folder(folder_path=str(space_dir), repo_id=space_id, repo_type="space",
                      ignore_patterns=["checkpoints/*", "__pycache__/*", "**/__pycache__/*", "*.pyc"])
    print(f"Model: https://huggingface.co/{model_id}")
    print(f"Space: https://huggingface.co/spaces/{space_id}  (first build takes a few minutes)")


if __name__ == "__main__":
    main()
