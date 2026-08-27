# SSDLite-MobileNetV3-Large NIR configuration

Torchvision's official COCO checkpoint uses a fixed 320x320 transform. This
benchmark loads every compatible pretrained detector tensor, replaces the
classification predictors for the two NIR classes, and fine-tunes the official
architecture at the shared 640x640 input. The adaptation is explicit in both the
protocol and backend manifests.

The zero-based training annotations are shifted to detector labels 1 and 2
because Torchvision reserves label 0 for background. Validation uses the
benchmark's one-based COCO annotations.

The native trainer follows the Torchvision reference SGD/cosine family. Its
reference learning rate 0.15 is linearly scaled from the official global batch
192 to this benchmark's effective batch 32, giving 0.025. Training uses FP32
because the 640-pixel adaptation is numerically unstable under CUDA FP16 AMP;
non-finite losses, gradients, or model state now stop the run immediately.
Physical batch 8, four-step gradient accumulation, deterministic seed 13,
validation-selected `best.pt`, per-epoch `last.pt`, and the 100-epoch retention
milestone remain frozen.
