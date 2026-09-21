"""End-to-end analysis of one image, shared by scripts/infer.py and app.py."""
from pathlib import Path

import torch

from src.inference.postprocess import (compute_stats, density_heatmap, remove_small_components,
                                       separate_buildings, threshold_mask)
from src.inference.render import heatmap_overlay, instance_overlay, mask_overlay
from src.inference.tiling import make_model_predict_fn, predict_tiled
from src.models.unet import build_model


def pick_checkpoint(infer_cfg):
    """Find the weights to use. Order: local trained checkpoint -> download from the Hugging Face Hub
    (if hf_repo is set) -> debug checkpoint with a warning. Returns (path, warning_or_None)."""
    trained = Path(infer_cfg["checkpoint"])
    if trained.exists():
        return trained, None
    if infer_cfg.get("hf_repo"):
        from huggingface_hub import hf_hub_download  # downloaded once, then cached
        path = hf_hub_download(repo_id=infer_cfg["hf_repo"], filename=infer_cfg["hf_filename"])
        return Path(path), None
    debug = Path(infer_cfg.get("debug_checkpoint") or "")
    if debug.is_file():
        return debug, (f"WARNING: trained checkpoint '{trained}' not found. Using the DEBUG checkpoint "
                       f"(2 tiny epochs), so predictions are essentially meaningless.")
    raise FileNotFoundError(f"No checkpoint found: '{trained}' or '{debug}'. Run scripts/train.py first.")


def load_model(checkpoint, cfg, device):
    """Build the U-Net and load weights saved by train.py (best.pth = weights only).
    encoder_weights=None: no need to download ImageNet weights, the checkpoint already contains everything."""
    model_cfg = {"model": {**cfg["model"], "encoder_weights": None}}
    model = build_model(model_cfg)
    model.load_state_dict(torch.load(Path(checkpoint), map_location=device))
    return model.to(device).eval()


def analyze_image(image, model, device, infer_cfg):
    """image: (H,W,3) uint8 RGB. Returns a dict with the probability map, mask, instance labels, stats and overlays."""
    predict_fn = make_model_predict_fn(model, device)
    prob = predict_tiled(image, predict_fn, infer_cfg["tile_size"], infer_cfg["overlap"])
    mask = threshold_mask(prob, infer_cfg["threshold"])
    mask = remove_small_components(mask, infer_cfg["min_building_area_px"])
    labels = separate_buildings(mask, infer_cfg["watershed_h"])
    stats = compute_stats(mask, labels, infer_cfg["pixel_size_m"])
    heat = density_heatmap(labels, infer_cfg["heatmap_cell_px"])
    return {
        "prob": prob, "mask": mask, "labels": labels, "stats": stats, "heat": heat,
        "mask_overlay": mask_overlay(image, mask),
        "instance_overlay": instance_overlay(image, labels),
        "heatmap_overlay": heatmap_overlay(image, heat),
    }
