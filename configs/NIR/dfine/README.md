# D-FINE-N NIR Configuration

`base.yml` freezes the two-class recipe. `ratio_1to2.yml` and `ratio_1to6.yml`
override only the training annotation and output paths. The NIR schedule runs for
100 epochs, with the augmentation transition proportionally placed at epoch 67.
