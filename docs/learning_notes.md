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

## Stage 2: Training pipeline
- **U-Net**: encoder-decoder with skip connections; the encoder downsamples to capture context, the decoder upsamples back to per-pixel predictions, and skips restore fine spatial detail (building edges). Good for small datasets.
- **Pretrained ResNet34 encoder**: ImageNet features (edges, textures) transfer to aerial imagery and converge much faster with only 137 training images. So input must use ImageNet mean/std normalisation.
- **Random 512 crops**: 1500x1500 images don't fit in GPU memory at batch size 8; crops also act as augmentation. Crops that are mostly white/black no-data are re-drawn.
- **Losses**: BCE+Dice vs Focal+Dice. Dice = 1 - 2|P∩T|/(|P|+|T|) computed on the whole batch; it fights class imbalance (buildings ~5-10% of pixels). Focal = BCE*(1-p_t)^gamma down-weights easy background pixels. Losses take logits (BCEWithLogits is more numerically stable than sigmoid+BCE).
- **Metrics pooled over the dataset**: accumulate TP/FP/FN over all pixels, then IoU=TP/(TP+FP+FN), Dice=2TP/(2TP+FP+FN). Averaging per-batch IoU is biased. Dice >= IoU always; Dice = 2*IoU/(1+IoU).
- **Validation tiles**: 3x3 grid, last tile shifted to the image edge (small overlap counted twice). Full-image blended inference comes in Stage 4.
- **Mixed precision** (autocast + GradScaler): fp16 forward is faster on GPU; the scaler stops small gradients underflowing. Losses cast logits to float32.
- **Checkpointing**: last.pth (full state incl. optimizer/scheduler/scaler) each epoch for resume; best.pth = weights of best val IoU. Test evaluated once, with the best model.
- **Reproducibility**: seeds for random/numpy/torch; `random` (not numpy) is used for cropping because PyTorch seeds it per DataLoader worker.
- **Cosine LR schedule** with AdamW (lr 3e-4): smooth decay helps final convergence in short runs.

Likely interview questions:
- Why Dice + BCE instead of BCE alone? What does Focal add? What are gamma and p_t?
- Why compute IoU over the whole val set rather than per batch?
- What does GradScaler do? Why cast logits to float in the loss?
- Why ImageNet normalisation with a pretrained encoder? Would you freeze the encoder?
- How does resuming from a checkpoint work, and what must be saved?
- Why choose the best checkpoint by val IoU and evaluate test only once?

## Stage 4: Inference and post-processing
- **Tiled inference**: the model is trained on 512 crops; a 1500x1500 image is split into overlapping 512 tiles (stride 384), predicted in batches, and merged. Reflect-padding (mirror) gives border pixels context without fake black edges.
- **Blending**: prediction = sum(w*p)/sum(w) with a Gaussian window w (peak at tile centre, ~0 at tile edge). A window that is still large at the tile edge causes a seam where a tile starts contributing (a test caught this at sigma=0.25*tile, so we use 0.15). Test: with an identity "model" the stitched result must equal the input.
- **Watershed**: threshold -> remove blobs < 30 px -> distance transform (distance of each building pixel to background) -> h-maxima markers (one per building "peak") -> watershed on the inverted distance. Touching buildings meet at a narrow neck = a valley, so the flood separates them. h controls over- vs under-segmentation (bumpy outline -> too many peaks without it).
- **Stats**: count = number of labels, built-up area % = mask mean, areas in m2 via pixel_size_m (Massachusetts is ~1 m/px).
- **Density heatmap**: building centroids counted on a 64 px grid, Gaussian-smoothed, bicubic-upsampled, shown with a JET colormap.
- **Known limits**: watershed struggles with attached row houses (under-split) and complex roofs (over-split); very large images at 97% "built-up" from the debug model show why we always check the checkpoint (app shows a warning if it only has the debug one).

Likely interview questions:
- Why overlap and Gaussian blending instead of just cutting tiles? What causes seams?
- Explain watershed with a distance transform. Why h-maxima? What fails with connected rows of houses?
- How do you unit-test the stitching without a trained model?
- Why does the debug model report 97% built-up area?
- How would you convert pixels to real-world area? What would change for another dataset?
- How would you evaluate the instance count (not just the mask)?
