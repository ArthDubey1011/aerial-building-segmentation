"""Kaggle script kernel: clone the repo, run the debug check + both experiments, zip results.
Same steps as notebooks/kaggle_run.md, but as a plain script so `kaggle kernels push` can run it unattended.
Everything is also logged to /kaggle/working/run_log.txt so the log always comes back with the output."""
import glob
import os
import subprocess
import traceback

REPO = "https://github.com/ArthDubey1011/urban-growth-satellite.git"
LOG = "/kaggle/working/run_log.txt"


def sh(cmd):
    print(">>", cmd, flush=True)
    # bash + pipefail so a failing python command isn't hidden by `tee`
    subprocess.run(f"set -o pipefail; {cmd} 2>&1 | tee -a {LOG}", shell=True, check=True, executable="/bin/bash")


def find_data_root():
    """Kaggle mounts datasets either at /kaggle/input/<slug> or /kaggle/input/datasets/<owner>/<slug>.
    Find the folder that contains png/train wherever it is."""
    hits = glob.glob("/kaggle/input/**/png/train", recursive=True)
    if not hits:
        sh("find /kaggle/input -maxdepth 4 | head -50")
        raise FileNotFoundError("Could not find png/train under /kaggle/input")
    return os.path.dirname(hits[0])


try:
    sh(f"git clone {REPO} repo")
    os.chdir("repo")
    sh("pip install -q -r requirements.txt")

    import torch
    print("CUDA available:", torch.cuda.is_available(), flush=True)
    root = find_data_root()
    print("data root:", root, flush=True)

    train = f"python scripts/train.py --data-root {root}"
    sh(f"{train} --config configs/kaggle_bce_dice.yaml --debug")  # fail fast before spending GPU time
    sh(f"{train} --config configs/kaggle_bce_dice.yaml")
    sh(f"{train} --config configs/kaggle_focal_dice.yaml")
except Exception:
    traceback.print_exc()
finally:
    # Always package whatever exists. last.pth (optimizer state) is only needed for resuming: skip it (size).
    if os.path.isdir("results"):
        sh("rm -rf results/*_debug; zip -qr /kaggle/working/results.zip results -x '*/last.pth'; ls -lh /kaggle/working/results.zip")
