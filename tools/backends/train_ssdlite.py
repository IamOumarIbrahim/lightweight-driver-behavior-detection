"""Train the 640x640 Torchvision SSDLite NIR adaptation with safe resume."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as transform_functional

from tools.backends.ssdlite import (
    build_ssdlite,
    checkpoint_state,
    load_matching_pretrained,
)
from tools.benchmark.evaluation import coco_metrics
from tools.benchmark.paths import REPO_ROOT
from tools.benchmark.protocol import ProtocolError, resolve_repo_path
from tools.benchmark.training import accumulation_loss_scale


class CocoDetectionDataset(Dataset):
    def __init__(
        self,
        images: str | Path,
        annotations: str | Path,
        *,
        horizontal_flip_probability: float = 0.0,
        category_id_offset: int = 0,
    ) -> None:
        self.image_root = resolve_repo_path(images)
        self.annotation_path = resolve_repo_path(annotations)
        self.coco = json.loads(self.annotation_path.read_text(encoding="utf-8"))
        self.images = list(self.coco["images"])
        self.annotations: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for annotation in self.coco["annotations"]:
            self.annotations[int(annotation["image_id"])].append(annotation)
        self.horizontal_flip_probability = float(horizontal_flip_probability)
        self.category_id_offset = int(category_id_offset)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        record = self.images[index]
        image_id = int(record["id"])
        with Image.open(self.image_root / record["file_name"]) as source:
            image = transform_functional.pil_to_tensor(source.convert("RGB"))
        image = transform_functional.convert_image_dtype(image, torch.float32)
        boxes = []
        labels = []
        areas = []
        crowds = []
        for annotation in self.annotations[image_id]:
            x, y, width, height = map(float, annotation["bbox"])
            if width <= 0.0 or height <= 0.0:
                continue
            boxes.append([x, y, x + width, y + height])
            labels.append(int(annotation["category_id"]) + self.category_id_offset)
            areas.append(float(annotation.get("area", width * height)))
            crowds.append(int(annotation.get("iscrowd", 0)))
        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor(image_id, dtype=torch.int64),
            "area": torch.as_tensor(areas, dtype=torch.float32),
            "iscrowd": torch.as_tensor(crowds, dtype=torch.int64),
        }
        if (
            self.horizontal_flip_probability > 0.0
            and torch.rand(()) < self.horizontal_flip_probability
        ):
            image = transform_functional.hflip(image)
            width = float(image.shape[-1])
            boxes_tensor = target["boxes"]
            if boxes_tensor.numel():
                boxes_tensor[:, [0, 2]] = width - boxes_tensor[:, [2, 0]]
        return image, target


def collate_detection(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


def _seed_worker(worker_id: int) -> None:
    del worker_id
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def load_config(path: str | Path) -> dict[str, Any]:
    path = resolve_repo_path(path)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    include = config.pop("__include__", None)
    if include:
        base_path = path.with_name(str(include))
        base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
        base.update(config)
        config = base
    if not isinstance(config, dict):
        raise ProtocolError(f"Invalid SSDLite config: {path}")
    return config


def _atomic_torch_save(payload: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)


def _predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> list[dict[str, Any]]:
    model.eval()
    predictions: list[dict[str, Any]] = []
    with torch.inference_mode():
        for images, targets in loader:
            outputs = model([image.to(device, non_blocking=True) for image in images])
            for target, output in zip(targets, outputs):
                image_id = int(target["image_id"].item())
                for box, label, score in zip(
                    output["boxes"].cpu(),
                    output["labels"].cpu(),
                    output["scores"].cpu(),
                ):
                    category_id = int(label.item())
                    if category_id not in {1, 2}:
                        continue
                    x1, y1, x2, y2 = map(float, box.tolist())
                    if x2 <= x1 or y2 <= y1:
                        continue
                    predictions.append(
                        {
                            "image_id": image_id,
                            "category_id": category_id,
                            "bbox": [x1, y1, x2 - x1, y2 - y1],
                            "score": float(score.item()),
                        }
                    )
    return predictions


def train(config: dict[str, Any], pretrained: Path, resume: Path | None) -> None:
    if not torch.cuda.is_available():
        raise ProtocolError("CUDA is required for SSDLite training")
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda:0")
    model = build_ssdlite(
        int(config["num_classes"]), input_size=int(config["input_size"][0])
    )
    output_dir = resolve_repo_path(config["output_dir"])
    optimizer_spec = config["optimizer"]
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(optimizer_spec["lr"]),
        momentum=float(optimizer_spec["momentum"]),
        weight_decay=float(optimizer_spec["weight_decay"]),
    )
    scheduler_spec = config["scheduler"]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[int(value) for value in scheduler_spec["milestones"]],
        gamma=float(scheduler_spec["gamma"]),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=bool(config["amp"]))
    start_epoch = 0
    best_map = -math.inf
    if resume:
        state = checkpoint_state(resume)
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch = int(state["epoch"]) + 1
        best_map = float(state["best_map"])
    else:
        report = load_matching_pretrained(model, pretrained)
        print(json.dumps({"pretrained_transfer": report}, indent=2))
    model.to(device)

    generator = torch.Generator().manual_seed(seed)
    augmentation = config.get("augmentation", {})
    train_dataset = CocoDetectionDataset(
        config["images"],
        config["train_annotations"],
        horizontal_flip_probability=float(
            augmentation.get("horizontal_flip_probability", 0.0)
        ),
        category_id_offset=int(config["train_category_id_offset"]),
    )
    val_dataset = CocoDetectionDataset(config["images"], config["val_annotations"])
    workers = int(config["num_workers"])
    loader_options = {
        "num_workers": workers,
        "pin_memory": True,
        "collate_fn": collate_detection,
        "worker_init_fn": _seed_worker,
        "persistent_workers": workers > 0,
    }
    batch_size = int(config["physical_batch_size"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        generator=generator,
        **loader_options,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        **loader_options,
    )
    epochs = int(config["epochs"])
    accumulation = int(config["gradient_accumulation_steps"])
    log_path = output_dir / "log.txt"
    for epoch in range(start_epoch, epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_total = 0.0
        for batch_index, (images, targets) in enumerate(train_loader):
            images = [image.to(device, non_blocking=True) for image in images]
            targets = [
                {key: value.to(device, non_blocking=True) for key, value in target.items()}
                for target in targets
            ]
            scale = accumulation_loss_scale(
                batch_index=batch_index,
                total_batches=len(train_loader),
                accumulation_steps=accumulation,
                current_batch_size=len(images),
                total_samples=len(train_dataset),
                physical_batch_size=batch_size,
                batch_loss_reduction="mean",
            )
            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=bool(config["amp"])
            ):
                loss_dict = model(images, targets)
                loss = sum(loss_dict.values())
            scaler.scale(loss * scale).backward()
            should_step = (
                (batch_index + 1) % accumulation == 0
                or (batch_index + 1) == len(train_loader)
            )
            if should_step:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            loss_total += float(loss.detach().item()) * len(images)
        scheduler.step()
        predictions = _predictions(model, val_loader, device)
        metrics = coco_metrics(val_dataset.coco, predictions)
        map_value = float(metrics["map_50_95"])
        record = {
            "epoch": epoch,
            "train_loss": loss_total / len(train_dataset),
            "validation": metrics,
            "lr": optimizer.param_groups[0]["lr"],
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        state = {
            "schema_version": 1,
            "model_id": config["model"],
            "epoch": epoch,
            "best_map": max(best_map, map_value),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "config": config,
        }
        _atomic_torch_save(state, output_dir / "last.pt")
        if map_value > best_map:
            best_map = map_value
            _atomic_torch_save(state, output_dir / "best.pt")
        if (epoch + 1) % int(
            config["checkpoint_retention"]["periodic_every_epochs"]
        ) == 0:
            _atomic_torch_save(state, output_dir / f"epoch_{epoch + 1}.pt")
        print(json.dumps(record, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--pretrained", required=True, type=Path)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    train(load_config(args.config), args.pretrained.resolve(), args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
