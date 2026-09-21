# HANDOVER: 3-hour sprint, execution instructions for Claude Code

Read CLAUDE.md first (project rules + sprint scope). This file tells you WHAT to do and in WHAT ORDER.
Work through the stages in order. You run the commands yourself. Stop only at the points marked
**STOP**, where the user must do something you cannot do.

## Context
- User is on Windows, VS Code terminal = PowerShell. AMD Radeon graphics, NO CUDA. Local = CPU only.
- Real training happens on Kaggle GPU. Local runs are only `--debug` runs on data/sample/.
- The user is preparing for AI/CV interviews. Keep code simple and explain choices.

## Progress tracking (important)
- Create and maintain `PROGRESS.md`: which stages are done, what's next, key decisions
  (dataset slug, folder structure, mask format), and any open issues.
- Update it at the end of every stage. The user may run `/clear` between stages; a fresh session
  must be able to continue by reading CLAUDE.md, HANDOVER.md and PROGRESS.md.
- At the end of every stage: `git commit` with a clear message, append key concepts + likely
  interview questions to `docs/learning_notes.md`, and tell the user in 3-5 lines what was done
  and what happens next.

---

## Stage 0: Environment setup
1. Check `python --version` (need 3.10-3.12) and `git --version`. If either is missing/wrong, **STOP** and tell the user.
2. Create venv: `python -m venv .venv` and activate with `.\.venv\Scripts\Activate.ps1`.
   If script execution is blocked, **STOP** and ask the user to run
   `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` (don't change system settings yourself).
3. Install CPU PyTorch: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`
4. Create `requirements.txt` (WITHOUT torch/torchvision, since Kaggle already has GPU torch):
   segmentation-models-pytorch, albumentations, opencv-python-headless, scikit-image, pyyaml,
   tqdm, pandas, matplotlib, gradio, pytest, kaggle. Install it locally.
5. `git init`, create `.gitignore` (.venv/, data/, *.pth, *.pt, __pycache__/, results/**/checkpoints/, .ipynb_checkpoints/).
6. Create the folder structure from CLAUDE.md with placeholder `__init__.py` files.

## Stage 1: Get a small data sample (do NOT download the full dataset)
1. Check for the Kaggle API token at `%USERPROFILE%\.kaggle\kaggle.json`.
   If missing, **STOP** and tell the user: Kaggle -> Settings -> API -> "Create New Token",
   then move the downloaded kaggle.json to `C:\Users\<name>\.kaggle\`.
2. Find the dataset: `kaggle datasets list -s "massachusetts buildings"`. Pick the most popular
   version that has images + building masks in PNG or TIFF.
3. List its files: `kaggle datasets files <slug>` (page through if needed). Figure out the folder
   structure (train/val/test images and labels).
4. Download ONLY 3 image/mask pairs (use `kaggle datasets download <slug> -f <path>`) into
   `data/sample/images/` and `data/sample/masks/` with matching names. Unzip if needed.
5. Inspect them with a small Python snippet: image size, dtype, channels, mask unique values
   (masks may be 0/255 or RGB; decide how to convert to binary 0/1). Also check for
   blank/white regions in images (this dataset has some; decide how to handle them).
6. Record slug, folder structure and mask format in PROGRESS.md.
7. Create `configs/local.yaml` (data/sample, CPU, num_workers 0, tiny settings) and
   `configs/kaggle_bce_dice.yaml` + `configs/kaggle_focal_dice.yaml` (paths under /kaggle/input/<slug>/...,
   num_workers 2, mixed precision on).

## Stage 2: Training pipeline
1. `src/data/`: dataset returning random 512x512 crops for training (skip crops that are mostly
   blank), center/tiled crops for validation, albumentations (flips, rot90, light colour jitter),
   ImageNet normalization, mask converted to 0/1 float.
2. `src/models/`: U-Net with ImageNet-pretrained resnet34 encoder via segmentation-models-pytorch.
3. `src/losses/`: BCE+Dice and Focal+Dice, chosen by config.
4. `src/eval/`: IoU and Dice metrics (computed over the whole val set, not averaged per batch).
5. `scripts/train.py`: config-driven, seeds, automatic device choice, mixed precision on CUDA,
   checkpoint every epoch + best model by val IoU, resume support, logs metrics.csv + config copy
   + a few prediction images to `results/<experiment_name>/`. `--debug` flag = 2 tiny epochs on
   data/sample on CPU, under 2 minutes.
6. `tests/`: tests for losses (perfect prediction -> ~0 loss), metrics (known small examples),
   dataset output shapes/values.
7. Run `pytest` and `python scripts/train.py --config configs/local.yaml --debug`. Fix until both pass.
   Note: with a pretrained encoder, the first run downloads weights (~85 MB); that's expected.

## Stage 3: Kaggle handoff
1. Write `notebooks/kaggle_run.md` with copy-paste cells for a Kaggle notebook:
   - `!git clone <repo_url>` and `%cd` into it
   - `!pip install -q -r requirements.txt`
   - quick GPU check (`torch.cuda.is_available()`)
   - run experiment 1, then experiment 2 (each should finish in ~15-20 min; tune epochs/crops-per-epoch in the configs so they do)
   - zip `results/` (including the best checkpoint of each run) for download
   - short setup notes: Accelerator = GPU T4, Internet = ON, add the dataset via "Add Input",
     and use "Save Version -> Save & Run All" so it keeps running if the browser closes.
2. Commit. Ask the user for their GitHub repo URL (it should be a PUBLIC empty repo, so Kaggle can clone
   it without a token). Add the remote and push. If push fails due to auth, **STOP** and give the user
   the exact commands to run.
3. **STOP**: tell the user to start the Kaggle run following notebooks/kaggle_run.md.
   Say that you'll continue with Stage 4 while it trains.

## Stage 4: Inference, quantification, demo (while Kaggle trains)
1. `src/inference/tiling.py`: sliding-window inference with overlap (e.g. 512 tiles, 128 overlap),
   smooth blending (Gaussian or linear weight window) so there are no seams; reflect-pad edges.
2. `src/inference/postprocess.py`: threshold -> clean small noise -> distance transform +
   watershed to separate touching buildings -> building count, built-up area %, per-building areas.
3. Density heatmap: count building centroids on a grid, smooth, overlay on the image.
4. `scripts/infer.py`: image in -> saves mask, instance overlay, heatmap, stats JSON.
5. Tests: tiling with an identity "model" must reconstruct the input exactly (within float tolerance);
   watershed on two touching synthetic circles must give count 2.
6. `app.py`: Gradio app: upload image -> mask overlay, heatmap, building count, area %.
   Load the checkpoint path from config; if no trained checkpoint exists yet, use the debug checkpoint
   and show a warning in the UI.
7. Test everything locally on data/sample. Commit and push.

## Stage 5: Results and README (after the user returns with results)
**STOP** at the start of this stage if `results/` from Kaggle isn't in the project yet. Ask the user to
download the zip from Kaggle and extract it into the project's `results/` folder (checkpoints stay gitignored).
1. `scripts/make_report.py`: comparison table (val/test IoU and Dice per experiment), training curves,
   6 good predictions, and 2 clear failure cases with a one-line reason for each.
2. Run infer.py on 2 full test images with the best model; save figures to `docs/figures/`.
3. README.md: problem, dataset, approach (why U-Net, why pretrained encoder, why Dice-based losses
   for class imbalance, why tiling), results table + figures, failure analysis, how to run
   (local + Kaggle + demo), future work (SegFormer comparison, self-supervised pretraining with
   fewer labels, cross-city generalization on Inria, change detection on LEVIR-CD, ONNX speed-up).
4. Final commit + push. Give the user a 10-question interview self-check based on this project.
