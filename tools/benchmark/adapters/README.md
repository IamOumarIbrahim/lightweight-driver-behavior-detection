# Model Adapters

Ultralytics, D-FINE, Torchvision SSDLite, RT-DETRv2, and YOLOX outputs are
normalized in-process to one-based COCO predictions for the shared evaluator.
Every adapter exposes the same 640x640 FP32 tensor boundary for CUDA AMP
profiling.
