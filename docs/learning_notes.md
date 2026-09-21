# Learning notes

## Stage 0: Environment
- **Why a venv?** Isolates project dependencies so versions don't clash with other projects.
- **Why is torch not in requirements.txt?** Kaggle ships a GPU build of torch; reinstalling from PyPI could replace it with a mismatched one. Locally we install the CPU wheel from the PyTorch CPU index.
- **Why Python 3.12, not 3.13?** Many ML libraries lag on wheels for the newest Python; 3.10-3.12 is the safe range.
- **Why data/ and checkpoints are gitignored?** Large binaries bloat the repo; datasets live on Kaggle, checkpoints are shipped as results zips.

Likely interview questions:
- Why pin/choose a Python version for ML projects?
- How do you keep local (CPU) and Kaggle (GPU) environments consistent?
