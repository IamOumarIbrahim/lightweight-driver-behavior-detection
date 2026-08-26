# EfficientDet-D1 NIR Configuration

The pinned `tf_efficientdet_d1` implementation retains its 640-pixel
EfficientNet-B1/BiFPN architecture and COCO-pretrained weights. Training uses
physical batch 8, four-step sample-correct accumulation, native AMP, EMA, and a
100-epoch cosine schedule. The container entrypoint converts the frozen
zero-based training COCO file to the one-based labels required by this backend;
the source artifact is never modified.
