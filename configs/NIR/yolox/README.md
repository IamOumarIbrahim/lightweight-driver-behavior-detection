# YOLOX-Nano NIR configuration

The official Nano architecture is retained, including its depthwise backbone,
augmentation policy, SGD optimizer, warmup-cosine schedule, and EMA. Its native
416-pixel input is deliberately adapted to the benchmark's shared 640x640 input.
Training uses physical batch 8 with four-step gradient accumulation and evaluates
each epoch so `best_ckpt.pth` remains resume-safe.
