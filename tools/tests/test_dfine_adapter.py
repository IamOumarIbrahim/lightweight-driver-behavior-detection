from __future__ import annotations

import torch

from tools.benchmark.adapters.dfine import DFineAdapter


class _FixedPostprocessor:
    def __call__(self, raw_outputs, sizes):
        del raw_outputs, sizes
        return (
            [torch.tensor([0, 1, 2])],
            [
                torch.tensor(
                    [
                        [10.0, 20.0, 10.0, 30.0],
                        [40.0, 50.0, 60.0, 50.0],
                        [70.0, 80.0, 90.0, 100.0],
                    ]
                )
            ],
            [torch.tensor([0.1, 0.2, 0.9])],
        )


def test_dfine_normalization_discards_collapsed_topk_boxes() -> None:
    adapter = object.__new__(DFineAdapter)
    adapter.device = torch.device("cpu")
    adapter.class_count = 4
    adapter.allow_pretrained_head_mismatch = False
    adapter.postprocessor = _FixedPostprocessor()

    predictions = adapter.normalize(raw_outputs=None, image_ids=[123])

    assert predictions == [
        {
            "image_id": 123,
            "category_id": 3,
            "bbox": [70.0, 80.0, 20.0, 20.0],
            "score": torch.tensor(0.9).item(),
        }
    ]
