# PROGRESS

## Stage status
- [x] Stage 0: Environment setup
- [x] Stage 1: Data sample
- [x] Stage 2: Training pipeline
- [~] Stage 3: Kaggle handoff (notebook written; GitHub repo NOT created/pushed yet, see open issues)
- [x] Stage 4: Inference + demo (code done + tested locally; not pushed yet)
- [ ] Stage 5: Results + README

## Key decisions
- Venv uses Python 3.12.6 (`py -3.12 -m venv .venv`) because the default `python` is 3.13.5, outside the 3.10-3.12 range.
  Always use `.\.venv\Scripts\python.exe` or activate the venv.
- Local torch is 2.14.0+cpu. requirements.txt deliberately excludes torch/torchvision (Kaggle has GPU torch).
- segmentation-models-pytorch 0.5.0.

- Dataset: Kaggle `balraj98/massachusetts-buildings-dataset` (auth via `~/.kaggle/access_token`, new-style token string).
  Kaggle path: `/kaggle/input/massachusetts-buildings-dataset/png/{train,train_labels,val,val_labels,test,test_labels}`.
  Image and mask files share the same name (e.g. `22678915_15.png`).
- Images: 1500x1500 RGB uint8. Masks: 1500x1500x3 uint8 with values {0,255} -> binary = channel 0 > 127.
- Sample (data/sample/): train images 22678915_15, 22678930_15, 22678945_15 (building fraction 5-10%). No blank regions in
  these, but the full dataset has some -> training crops skipped if >50% pure white/black pixels.
- Configs: local.yaml, kaggle_bce_dice.yaml, kaggle_focal_dice.yaml (same schema; data folders relative to data.root).

- Stage 2 code: src/data/dataset.py, src/models/unet.py, src/losses/losses.py, src/eval/metrics.py, src/eval/visualize.py, src/train/{engine,utils}.py, scripts/train.py. 11 pytest tests pass; python scripts/train.py --config configs/local.yaml --debug runs in ~50 s on CPU and --resume works.
- train.py outputs to results/<name>/: config.yaml, metrics.csv, test_metrics.json, samples/, checkpoints/{best,last}.pth (checkpoints gitignored).
- Kaggle configs: 40 epochs x 400 crops, batch 8, AMP. Runtime is an estimate (~15 min on T4) -- unverified on GPU.

- Stage 3: notebooks/kaggle_run.md written (REPO_URL_HERE placeholder must be replaced after the GitHub remote exists).

- Stage 4 code: src/inference/{tiling,postprocess,render,pipeline}.py, scripts/infer.py, app.py, tests/test_inference.py (23 tests total pass). infer.py and app.py both have working --debug modes.
- Infer config keys live under infer: (threshold, min_building_area_px, watershed_h, heatmap_cell_px, pixel_size_m, checkpoint, debug_checkpoint).

## Open issues
- GitHub repo creation via gh repo create --public was blocked by the permission classifier. The user must create/allow it (or create an empty public repo and give the URL). Then: add remote, push, replace REPO_URL_HERE in notebooks/kaggle_run.md.
- CLAUDE.md and HANDOVER.md were not found on disk in the project folder (they were only provided in chat context). Consider saving them there so a fresh session can read them.



