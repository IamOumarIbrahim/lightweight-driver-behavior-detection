# Model Adapters

Ultralytics and D-FINE outputs are normalized in-process to one-based COCO
predictions. RTMDet-Tiny and EfficientDet-D1 use the same normalization contract
inside their pinned CUDA container, then return a checksum-ready JSON envelope
to the shared evaluator.
