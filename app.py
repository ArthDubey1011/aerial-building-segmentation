"""Gradio demo: upload an aerial image -> building mask, instance overlay, density heatmap, count, area %.

Usage (PowerShell):
    python app.py --config configs/local.yaml
    python app.py --config configs/local.yaml --debug    # smoke test: analyse one sample image, no web server
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gradio as gr

from src.data.dataset import read_image
from src.inference.pipeline import analyze_image, load_model
from src.train.utils import get_device, load_config


def pick_checkpoint(infer_cfg):
    """Use the trained checkpoint from the config if it exists, otherwise fall back to the debug one.
    Returns (path, warning_or_None)."""
    trained = Path(infer_cfg["checkpoint"])
    if trained.exists():
        return trained, None
    # On Hugging Face Spaces the 98 MB checkpoint lives in a model repo and is downloaded at start-up.
    if infer_cfg.get("hf_repo"):
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id=infer_cfg["hf_repo"], filename=infer_cfg["hf_filename"])
        return Path(path), None
    debug = Path(infer_cfg.get("debug_checkpoint") or "")
    if debug.is_file():
        return debug, (f"WARNING: trained checkpoint '{trained}' not found. Using the DEBUG checkpoint "
                       f"(2 tiny epochs), so predictions are essentially meaningless.")
    raise FileNotFoundError(f"No checkpoint found: '{trained}' or '{debug}'. Run scripts/train.py first.")


FAST_SIZE = 1024  # fast mode analyses the centre 1024x1024 crop: ~9 tiles instead of ~16 (matters on free CPU)


def to_rgb(image):
    """Uploads may be grayscale or RGBA (PNG with transparency); the model needs exactly 3 channels."""
    if image.ndim == 2:
        image = image[..., None]
    if image.shape[-1] == 1:
        image = image.repeat(3, axis=-1)
    return image[..., :3]


def center_crop(image, size):
    h, w = image.shape[:2]
    y, x = max(0, (h - size) // 2), max(0, (w - size) // 2)
    return image[y:y + size, x:x + size]


def build_app(cfg):
    device = get_device()
    checkpoint, warning = pick_checkpoint(cfg["infer"])
    model = load_model(checkpoint, cfg, device)

    def run(image, fast):
        if image is None:
            raise gr.Error("Please upload an image.")
        image = to_rgb(image)
        if min(image.shape[:2]) < 64:
            raise gr.Error("Image is too small (need at least 64x64 pixels).")
        if fast:
            image = center_crop(image, FAST_SIZE)
        result = analyze_image(image, model, device, cfg["infer"])
        s = result["stats"]
        return (result["mask_overlay"], result["instance_overlay"], result["heatmap_overlay"],
                s["building_count"], round(s["built_up_area_pct"], 2))

    with gr.Blocks(title="Aerial Building Segmentation") as demo:
        gr.Markdown("# Building detection from aerial imagery\n"
                    "U-Net (ResNet34 encoder) with tiled inference, watershed building separation and a density heatmap.")
        if warning:
            gr.Markdown(f"**{warning}**")
        else:
            gr.Markdown(f"Model: `{checkpoint}`")
        gr.Markdown("Trained on Massachusetts aerial imagery at ~1 m/pixel, so it works best on similar "
                    "resolution. Analyses one image at a time (no change over time).")
        inp = gr.Image(type="numpy", label="Aerial image (RGB)")
        fast = gr.Checkbox(value=True, label=f"Fast mode: analyse only the centre {FAST_SIZE}x{FAST_SIZE} crop "
                                             "(untick for the full image; slower on CPU)")
        btn = gr.Button("Analyse", variant="primary")
        with gr.Row():
            out_mask = gr.Image(label="Building mask overlay")
            out_inst = gr.Image(label="Individual buildings (watershed)")
            out_heat = gr.Image(label="Building density heatmap")
        with gr.Row():
            out_count = gr.Number(label="Building count")
            out_area = gr.Number(label="Built-up area (%)")
        btn.click(run, [inp, fast], [out_mask, out_inst, out_heat, out_count, out_area])
        examples = sorted(Path("examples").glob("*.jpg"))  # only present if bundled (see scripts/build_space.py)
        if examples:
            gr.Examples(examples=[[str(p), True] for p in examples], inputs=[inp, fast], label="Try an example")
    return demo, model, device, warning


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/local.yaml")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    demo, model, device, warning = build_app(cfg)

    if args.debug:
        first = sorted((Path(cfg["data"]["root"]) / cfg["data"]["test_images"]).iterdir())[0]
        result = analyze_image(read_image(first)[:1024, :1024], model, device, cfg["infer"])
        print("warning:", warning)
        print(first.name, {k: v for k, v in result["stats"].items() if k != "building_areas_px"})
        return
    demo.launch()


if __name__ == "__main__":
    main()
