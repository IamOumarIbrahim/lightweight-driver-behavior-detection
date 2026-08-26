"""Frozen MMYOLO RTMDet-Tiny recipe for the NIR experiment."""

_base_ = "/opt/mmyolo/configs/rtmdet/rtmdet_tiny_syncbn_fast_8xb32-300e_coco.py"

custom_imports = dict(imports=["mmyolo"], allow_failed_imports=False)

classes = ("drinking", "phone_use")
metainfo = dict(classes=classes)
image_root = "/workspace/data/processed/NIR/images/"
validation_annotations = "/workspace/data/processed/NIR/coco/dfine/evaluation/instances_val.json"
pretrained_checkpoint = "/workspace/third_party/weights/rtmdet_tiny_coco.pth"

model = dict(
    backbone=dict(init_cfg=None),
    bbox_head=dict(head_module=dict(num_classes=2)),
    train_cfg=dict(assigner=dict(num_classes=2)),
)

train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    dataset=dict(
        metainfo=metainfo,
        data_prefix=dict(img=image_root),
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
    ),
)
val_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    dataset=dict(
        metainfo=metainfo,
        ann_file=validation_annotations,
        data_prefix=dict(img=image_root),
        batch_shapes_cfg=None,
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(ann_file=validation_annotations)
test_evaluator = val_evaluator

optim_wrapper = dict(
    type="AmpOptimWrapper",
    accumulative_counts=4,
    loss_scale="dynamic",
    optimizer=dict(lr=0.002),
)
param_scheduler = [
    dict(type="LinearLR", start_factor=1.0e-5, by_epoch=False, begin=0, end=1000),
    dict(
        type="CosineAnnealingLR",
        eta_min=0.0001,
        begin=50,
        end=100,
        T_max=50,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
]

max_epochs = 100
stage2_epoch = 93
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=100, val_interval=1)
default_hooks = dict(
    checkpoint=dict(
        type="CheckpointHook",
        interval=1,
        max_keep_ckpts=2,
        save_best="coco/bbox_mAP",
        rule="greater",
        save_last=True,
    )
)
custom_hooks = [
    dict(
        type="EMAHook",
        ema_type="ExpMomentumEMA",
        momentum=0.0002,
        update_buffers=True,
        strict_load=False,
        priority=49,
    ),
    dict(
        type="mmdet.PipelineSwitchHook",
        switch_epoch=stage2_epoch,
        switch_pipeline=_base_.train_pipeline_stage2,
    ),
]

load_from = pretrained_checkpoint
randomness = dict(seed=13, deterministic=True)
