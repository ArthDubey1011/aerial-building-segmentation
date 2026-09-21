"""Assemble the Hugging Face Space folder (hf_space/) from this repo. Publishes nothing.

Usage (PowerShell):
    python scripts/build_space.py --user YOUR_HF_USERNAME
    python scripts/build_space.py --user YOUR_HF_USERNAME --with-examples   # bundle 2 sample crops (check dataset licence!)
Then test locally, and publish with scripts/publish_space.py.
"""
import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2

OUT = ROOT / "hf_space"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", required=True, help="your Hugging Face username")
    parser.add_argument("--with-examples", action="store_true",
                        help="bundle 2 example crops from data/sample (only if the dataset licence allows redistribution)")
    args = parser.parse_args()

    OUT.mkdir(exist_ok=True)
    shutil.copy(ROOT / "app.py", OUT / "app.py")
    shutil.copytree(ROOT / "src", OUT / "src", dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
    for name in ("README.md", "requirements.txt", "space_app.py"):
        shutil.copy(ROOT / "space" / name, OUT / name)

    (OUT / "configs").mkdir(exist_ok=True)
    cfg = (ROOT / "configs" / "space.yaml").read_text(encoding="utf-8").replace("HF_USER", args.user)
    (OUT / "configs" / "space.yaml").write_text(cfg, encoding="utf-8")

    if args.with_examples:
        (OUT / "examples").mkdir(exist_ok=True)
        for i, path in enumerate(sorted((ROOT / "data" / "sample" / "images").glob("*.png"))[:2]):
            img = cv2.imread(str(path))[:1024, :1024]
            cv2.imwrite(str(OUT / "examples" / f"example_{i}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])

    files = sorted(p.relative_to(OUT) for p in OUT.rglob("*") if p.is_file())
    print(f"Built {OUT} with {len(files)} files:")
    for f in files:
        print("  ", f)


if __name__ == "__main__":
    main()
