# RT-DETRv2-S NIR configuration

The official RT-DETRv2 R18VD architecture is the smallest upstream v2 model and
is named `RT-DETRv2-S` by its authors. These files keep its official 640-pixel
recipe while applying the shared NIR budget: 100 epochs, physical batch 8,
four-step gradient accumulation, AMP, seed 13, and no early stopping.

Run it through `tools.workflow.train`; do not invoke the upstream trainer
directly because setup verifies and applies the accumulation/resume patch.
