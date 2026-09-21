# PROGRESS

## Stage status
- [x] Stage 0: Environment setup
- [ ] Stage 1: Data sample
- [ ] Stage 2: Training pipeline
- [ ] Stage 3: Kaggle handoff
- [ ] Stage 4: Inference + demo
- [ ] Stage 5: Results + README

## Key decisions
- Venv uses Python 3.12.6 (`py -3.12 -m venv .venv`) because the default `python` is 3.13.5, outside the 3.10-3.12 range.
  Always use `.\.venv\Scripts\python.exe` or activate the venv.
- Local torch is 2.14.0+cpu. requirements.txt deliberately excludes torch/torchvision (Kaggle has GPU torch).
- segmentation-models-pytorch 0.5.0.

## Open issues
- CLAUDE.md and HANDOVER.md were not found on disk in the project folder (they were only provided in chat context). Consider saving them there so a fresh session can read them.
- Stage 1 needs the Kaggle API token at `%USERPROFILE%\.kaggle\kaggle.json` (not checked yet).
