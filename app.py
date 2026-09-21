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
    debug = Path(infer_cfg["debug_checkpoint"])
    if debug.exists():
        return debug, (f"WARNING: trained checkpoint '{trained}' not found. Using the DEBUG checkpoint "
                       f"(2 tiny epochs), so predictions are essentially meaningless.")
    raise FileNotFoundError(f"No checkpoint found: '{trained}' or '{debug}'. Run scripts/train.py first.")


def build_app(cfg):
    device = get_device()
    checkpoint, warning = pick_checkpoint(cfg["infer"])
    model = load_model(checkpoint, cfg, device)

    def run(image):
        if image is None:
            raise gr.Error("Please upload an image.")
        result = analyze_image(image, model, device, cfg["infer"])
        s = result["stats"]
        return (result["mask_overlay"], result["instance_overlay"], result["heatmap_overlay"],
                s["building_count"], round(s["built_up_area_pct"], 2))

    with gr.Blocks(title="Urban Growth Monitor") as demo:
        gr.Markdown("# Building detection from aerial imagery\n"
                    "U-Net (ResNet34 encoder) with tiled inference, watershed building separation and a density heatmap.")
        if warning:
            gr.Markdown(f"**{warning}**")
        else:
            gr.Markdown(f"Model: `{checkpoint}`")
        inp = gr.Image(type="numpy", label="Aerial image (RGB)")
        btn = gr.Button("Analyse", variant="primary")
        with gr.Row():
            out_mask = gr.Image(label="Building mask overlay")
            out_inst = gr.Image(label="Individual buildings (watershed)")
            out_heat = gr.Image(label="Building density heatmap")
        with gr.Row():
            out_count = gr.Number(label="Building count")
            out_area = gr.Number(label="Built-up area (%)")
        btn.click(run, inp, [out_mask, out_inst, out_heat, out_count, out_area])
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
