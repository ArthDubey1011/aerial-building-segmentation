# Interview self-check (answer out loud, then check against the code and docs/learning_notes.md)

1. Why is pixel accuracy a bad metric here? What do IoU and Dice measure, and how are they related (Dice = 2·IoU/(1+IoU))?
2. Why compute IoU by summing TP/FP/FN over the whole validation set instead of averaging per batch?
3. Explain BCE+Dice vs Focal+Dice. What does gamma do? Why did you (not) see a clear winner, and what would you do to decide (seeds, more val data)?
4. Why a U-Net with a pretrained ResNet34 encoder? What does the pretrained encoder need from the input (normalisation)? Would you freeze it?
5. Why train on random 512 crops but predict on full images? How does the tiling with Gaussian blending avoid seams, and how did you test it? (Hint: the seam your own test caught with sigma = 0.25.)
6. How does watershed on the distance transform separate touching buildings? What does h-maxima prevent, and where does it still fail (large complex roofs, row houses)?
7. Your test IoU (0.69) is higher than validation IoU (0.64). Is that a problem? (Think: 4 val images, different scenes, no test-set tuning.)
8. What does mixed precision do (autocast + GradScaler), and why are the losses computed in float32?
9. How does resuming a killed Kaggle session work? What is in last.pth vs best.pth and why are they different?
10. What are the weaknesses of this project, and how would you extend it to urban *growth* (change detection on LEVIR-CD)? What would you need to report growth reliably?

Bonus - a real bug story: the dataset module was missing on the first Kaggle run because `.gitignore` had `data/`, which also matched `src/data/`. How did you find it, and how did you make sure it can't recur (fresh-clone test)?
