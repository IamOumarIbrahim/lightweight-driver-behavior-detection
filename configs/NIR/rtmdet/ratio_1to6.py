_base_ = "./base.py"

train_annotations = "/workspace/data/processed/NIR/coco/dfine/ratio_1to6/instances_train.json"
train_dataloader = dict(dataset=dict(ann_file=train_annotations))
