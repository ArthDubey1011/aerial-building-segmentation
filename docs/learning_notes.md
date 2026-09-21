# Learning notes

## Stage 0: Environment
- **Why a venv?** Isolates project dependencies so versions don't clash with other projects.
- **Why is torch not in requirements.txt?** Kaggle ships a GPU build of torch; reinstalling from PyPI could replace it with a mismatched one. Locally we install the CPU wheel from the PyTorch CPU index.
- **Why Python 3.12, not 3.13?** Many ML libraries lag on wheels for the newest Python; 3.10-3.12 is the safe range.
- **Why data/ and checkpoints are gitignored?** Large binaries bloat the repo; datasets live on Kaggle, checkpoints are shipped as results zips.

Likely interview questions:
- Why pin/choose a Python version for ML projects?
- How do you keep local (CPU) and Kaggle (GPU) environments consistent?

## Stage 1: Data
- **Dataset**: Massachusetts Buildings, 1500x1500 aerial RGB images (~1 m/px) with binary building masks. Split: 137 train / 4 val / 10 test.
- **Mask format**: 3-channel PNG with values {0,255}; we threshold channel 0 at 127 to get a 0/1 mask. Thresholding (not `== 255`) is robust to PNG/resize artefacts.
- **Class imbalance**: buildings are only ~5-10% of pixels, so plain accuracy/BCE is misleading (predicting all background gives ~92% accuracy). Hence IoU/Dice metrics and Dice-based losses.
- **No-data regions**: some images have white/black borders; random crops that are mostly blank are skipped so the model doesn't waste training on them.
- **Kaggle data access**: datasets are attached as `/kaggle/input/<slug>`; paths come from config, so the same code runs locally and on Kaggle.

Likely interview questions:
- Why not just report pixel accuracy for segmentation?
- How did you convert the masks to binary and why threshold instead of equality?
- How do you keep the dataset paths portable between laptop and Kaggle?
