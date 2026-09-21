"""Entry point on Hugging Face Spaces: same app as app.py, with the Space config (checkpoint from the Hub)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app import build_app
from src.train.utils import load_config

demo, _model, _device, _warning = build_app(load_config("configs/space.yaml"))

if __name__ == "__main__":
    demo.launch()
