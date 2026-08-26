# RTMDet-Tiny NIR Configuration

These configs inherit the official MMYOLO RTMDet-Tiny 640-pixel recipe at the
pinned commit, preserve negative frames, and replace only dataset, class count,
budget, batch/accumulation, checkpoint, and single-GPU settings. Strong cached
Mosaic/MixUp augmentation switches to the official weak stage at epoch 93,
proportional to the final 20 of the original 300 epochs.
