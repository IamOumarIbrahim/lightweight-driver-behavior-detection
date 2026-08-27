"""YOLOX-Nano adapted from its official 416 recipe to the frozen 640 protocol."""

from __future__ import annotations

from pathlib import Path

from exps.default.yolox_nano import Exp as NanoExp


REPO_ROOT = Path(__file__).resolve().parents[3]


class Exp(NanoExp):
    def __init__(self, ratio: str = "1to6") -> None:
        super().__init__()
        if ratio not in {"1to2", "1to6"}:
            raise ValueError(f"Unsupported NIR ratio: {ratio}")
        self.num_classes = 2
        self.input_size = (640, 640)
        self.test_size = (640, 640)
        self.random_size = (20, 20)
        self.multiscale_range = 0
        self.max_epoch = 100
        self.gradient_accumulation_steps = 4
        self.reset_prefetcher_each_epoch = True
        self.eval_interval = 1
        self.save_history_ckpt = False
        self.seed = 13
        self.data_num_workers = 4
        self.test_conf = 0.001
        self.data_dir = str(REPO_ROOT / "data" / "processed" / "NIR")
        self.train_ann = f"../coco/dfine/ratio_{ratio}/instances_train.json"
        self.val_ann = "../coco/dfine/evaluation/instances_val.json"
        self.output_dir = str(REPO_ROOT / "runs" / "NIR" / "yolox_nano" / f"ratio_{ratio}")
        self.exp_name = "training"

    def get_dataset(self, cache: bool = False, cache_type: str = "ram"):
        from yolox.data import COCODataset, TrainTransform

        return COCODataset(
            data_dir=self.data_dir,
            json_file=self.train_ann,
            name="images",
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=50,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob,
            ),
            cache=cache,
            cache_type=cache_type,
        )

    def get_eval_dataset(self, **kwargs):
        from yolox.data import COCODataset, ValTransform

        return COCODataset(
            data_dir=self.data_dir,
            json_file=self.val_ann,
            name="images",
            img_size=self.test_size,
            preproc=ValTransform(legacy=kwargs.get("legacy", False)),
        )

    def get_data_loader(
        self,
        batch_size: int,
        is_distributed: bool,
        no_aug: bool = False,
        cache_img: str | None = None,
    ):
        if is_distributed:
            raise RuntimeError("The frozen Windows YOLOX path is single-GPU only")
        if cache_img is not None:
            raise RuntimeError("YOLOX image caching is disabled on Windows")

        import torch
        from yolox.data import (
            DataLoader,
            MosaicDetection,
            TrainTransform,
            YoloBatchSampler,
            worker_init_reset_seed,
        )

        if self.dataset is None:
            self.dataset = self.get_dataset(cache=False)
        self.dataset = MosaicDetection(
            dataset=self.dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=120,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob,
            ),
            degrees=self.degrees,
            translate=self.translate,
            mosaic_scale=self.mosaic_scale,
            mixup_scale=self.mixup_scale,
            shear=self.shear,
            enable_mixup=self.enable_mixup,
            mosaic_prob=self.mosaic_prob,
            mixup_prob=self.mixup_prob,
        )
        generator = torch.Generator().manual_seed(self.seed)
        sampler = torch.utils.data.RandomSampler(self.dataset, generator=generator)
        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            mosaic=not no_aug,
        )
        return DataLoader(
            self.dataset,
            num_workers=self.data_num_workers,
            pin_memory=True,
            batch_sampler=batch_sampler,
            worker_init_fn=worker_init_reset_seed,
        )
