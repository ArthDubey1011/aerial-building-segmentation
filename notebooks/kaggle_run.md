# Kaggle run: copy-paste cells

## One-time notebook setup (right-hand "Session options" panel)
1. **Accelerator**: GPU T4 x2 (or P100). Only one GPU is used.
2. **Internet**: ON (needed for `git clone` and the ImageNet encoder weights download).
3. **Add Input**: search "Massachusetts Buildings Dataset" by *balraj98* and add it. It appears at
   `/kaggle/input/massachusetts-buildings-dataset/` (check with cell 2; if the folder name differs, change
   `data.root` in both `configs/kaggle_*.yaml`).
4. When everything works: **Save Version -> Save & Run All (Commit)** so it keeps running if you close the browser.

## Cell 1: clone the repo and install requirements
```python
!git clone https://github.com/ArthDubey1011/urban-growth-satellite.git repo
%cd repo
!pip install -q -r requirements.txt
```

## Cell 2: sanity checks (GPU + dataset paths)
```python
import torch
print("CUDA available:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
!ls /kaggle/input/
!ls /kaggle/input/massachusetts-buildings-dataset/png
!ls /kaggle/input/massachusetts-buildings-dataset/png/train | head -3
```

## Cell 3 (optional but recommended): 1-minute debug run of the real code path on the GPU machine
Uses the Kaggle config with `--debug` (2 tiny epochs, 1 val/test image), so a crash costs seconds, not 15 minutes.
```python
!python scripts/train.py --config configs/kaggle_bce_dice.yaml --debug
```

## Cell 4: experiment 1, BCE + Dice
```python
!python scripts/train.py --config configs/kaggle_bce_dice.yaml
```
Look at the time per epoch in the first lines. Target: the whole run takes ~15-20 min. If it will be much
longer or shorter, edit `train.epochs` (or `data.crops_per_epoch`) in **both** configs so the experiments stay comparable.

If the session stops, re-run the same cell with `--resume` appended (continues from `last.pth`).

## Cell 5: experiment 2, Focal + Dice
```python
!python scripts/train.py --config configs/kaggle_focal_dice.yaml
```

## Cell 6: print the results and zip `results/` (includes best.pth of each run) for download
```python
import json, glob
for f in sorted(glob.glob("results/kaggle_*/test_metrics.json")):
    print(f, json.load(open(f)))
!rm -rf results/*_debug
!zip -qr /kaggle/working/results.zip results
!ls -lh /kaggle/working/results.zip
```
Download `results.zip` from the notebook's **Output** tab (or after "Save & Run All" from the version's Output),
then extract it into the project's `results/` folder on your laptop.

