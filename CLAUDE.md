# Project: Urban Growth Monitoring from Satellite Imagery

## Purpose
Portfolio computer vision project for an AI/ML Trainee Engineer role (medical image analysis company).
Interviews will deep-dive into every design choice, so the owner must understand ALL the code.
Explain non-obvious choices in comments and in your summaries. Prefer simple, readable code over clever code.

## CURRENT FOCUS: 3-hour sprint scope (do ONLY this now)
- Dataset: Massachusetts Buildings (1500x1500 aerial images, on Kaggle). Local sample in data/sample/.
- Model: U-Net with ImageNet-pretrained resnet34 encoder (segmentation-models-pytorch).
- Losses: BCE+Dice vs Focal+Dice (2 short experiments, ~15-20 min each on Kaggle GPU).
- Metrics: IoU, Dice on test split.
- infer.py: tiled sliding-window inference with overlap blending, watershed building separation,
  building count, built-up area %, density heatmap overlay.
- app.py: Gradio demo (upload image -> mask, count, area %, heatmap).
- README with results table, sample predictions, 2 failure cases, and future work.
- Speed over perfection: during the sprint, skip plan approval unless something is risky.

## Full scope (LATER extensions, not part of the sprint; switch datasets to Inria/LEVIR-CD then)
1. Baseline: U-Net building segmentation on Inria Aerial Image Labeling dataset (metrics: IoU, Dice).
   Loss ablation: BCE vs Dice vs Focal vs BCE+Dice.
2. Model comparison: SegFormer vs U-Net (accuracy, parameters, inference speed).
3. Self-supervised pretraining (SimCLR or MAE) on unlabelled patches, then fine-tune with 10% of labels.
   Compare against training from scratch / ImageNet weights with the same 10%.
4. Cross-city generalization: train on some cities, test on a held-out city. Improve with colour
   augmentation and normalization. Report before/after.
5. Large-image inference: tiled sliding window with overlap and smooth blending (no seams),
   watershed to separate touching buildings, building count, built-up area %, density heatmap,
   MC-dropout uncertainty map.
6. Change detection on LEVIR-CD (before/after image pairs): highlight new buildings, report growth stats.
Extras: Gradio demo app, ONNX export + speed benchmark.

## Tech stack
Python 3.10+, PyTorch, segmentation-models-pytorch, timm, albumentations, OpenCV, scikit-image,
tifffile or rasterio, YAML configs, pytest, Gradio, onnxruntime.

## Project structure
- configs/        YAML config per experiment
- data/           datasets (gitignored, never commit)
- src/data/       datasets, patching, augmentations
- src/models/     model builders
- src/losses/     loss functions
- src/train/      training loop
- src/eval/       metrics and evaluation
- src/inference/  tiled inference, post-processing, quantification
- scripts/        entry points (train.py, evaluate.py, infer.py)
- tests/          pytest tests
- results/        one folder per experiment (metrics CSV, config copy, sample images)
- docs/           learning_notes.md, figures for README

## Rules
- Work on ONE phase (or sub-task) at a time. Always propose a plan and wait for approval before writing code.
- All hyperparameters and paths go in config files. No hard-coded paths.
- Every script must support a `--debug` flag that runs on a tiny subset on CPU in under 2 minutes.
- Set random seeds for reproducibility.
- Save each experiment's metrics, config copy, and sample predictions to results/<experiment_name>/.
- Never download large datasets, delete files, or install system packages without asking first.
- Write pytest tests for losses, metrics, and tile/stitch logic (stitching must reconstruct the original image).
- Code must also run on Kaggle/Colab (no local-only assumptions).
- After each task: summarize what changed and why, list likely interview questions about it,
  and append key concepts to docs/learning_notes.md.

## Hardware and workflow
- Local: Windows laptop (PowerShell), AMD Radeon graphics, NO CUDA GPU. Local runs are CPU-only.
- Local use: write code, debug, run tests, and run `--debug` mode on 2-3 sample images only.
- Real training: Kaggle notebooks (free GPU, ~30 hrs/week, so avoid wasting GPU time on crashes).
  Workflow: push to GitHub -> `git clone` / `git pull` in Kaggle -> run scripts with a config -> download results/.
- Datasets live on Kaggle (attached as Kaggle datasets). Dataset root paths must come from config,
  with a local config (small sample in data/sample/) and a Kaggle config (/kaggle/input/...).
- Device selection must be automatic: use CUDA if available, else CPU.

## Windows-specific rules
- Use pathlib for all paths (no hard-coded "/" or "\\").
- All entry scripts must use `if __name__ == "__main__":` (required for DataLoader workers on Windows).
- `num_workers` comes from config; default 0 locally, 2-4 on Kaggle.
- Give terminal commands in PowerShell syntax when running locally.

## GPU budget rules
- Save checkpoints every epoch and support resuming training (Kaggle sessions can stop).
- Use mixed precision (torch.cuda.amp) on GPU to train faster.
- Keep Phase 3 (self-supervised pretraining) small: limited epochs and a subset of unlabelled patches.
- Before any full Kaggle run, the same config must pass in `--debug` mode locally.
