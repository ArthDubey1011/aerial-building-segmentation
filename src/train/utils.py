"""Small helpers shared by the scripts: config loading, seeding, device choice."""
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config(cfg, path):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def set_seed(seed):
    """Seed every RNG we use so runs are reproducible (GPU kernels may still be slightly non-deterministic)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def project_root():
    return Path(__file__).resolve().parents[2]
