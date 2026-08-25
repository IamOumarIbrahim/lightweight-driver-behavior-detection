import pytest

from tools.benchmark.protocol import validate_protocol
from tools.benchmark.evaluation import snippet_operating_point_metrics
from tools.data.freeze_nir_ratios import nested_subset
from tools.data.prepare_nir import (
    interpolate_box,
    load_ratio_tasks,
    result_value,
    signature,
    validate_counts,
)
from tools.setup.label_studio import to_one_fps


def test_interpolation():
    sequence = [
        {"frame": 3, "time": 0.1, "x": 10, "y": 20, "width": 30, "height": 40},
        {"frame": 30, "time": 1.0, "x": 20, "y": 30, "width": 40, "height": 50},
    ]
    assert interpolate_box(sequence, 0.55) == (15.0, 25.0, 35.0, 45.0)


def test_ratios_change_training_only():
    protocol = validate_protocol("NIR")
    tasks = load_ratio_tasks(protocol)
    validate_counts(tasks, protocol)
    evaluation = [
        [
            signature(task)
            for task in tasks[ratio]
            if task["data"]["split"] in {"val", "test"}
        ]
        for ratio in ("1to2", "1to6")
    ]
    assert evaluation[0] == evaluation[1]
    positives = {
        ratio: {
            signature(task)
            for task in tasks[ratio]
            if task["data"]["split"] == "train" and result_value(task) is not None
        }
        for ratio in ("1to2", "1to6")
    }
    negatives = {
        ratio: {
            signature(task)
            for task in tasks[ratio]
            if task["data"]["split"] == "train" and result_value(task) is None
        }
        for ratio in ("1to2", "1to6")
    }
    assert positives["1to2"] == positives["1to6"]
    assert negatives["1to2"] < negatives["1to6"]
    assert [signature(task) for task in tasks["1to2"]] == [
        signature(task) for task in nested_subset(tasks["1to6"])
    ]


def test_label_studio_tracklet_is_retimed_to_one_fps():
    task = {
        "id": 1,
        "data": {},
        "annotations": [
            {
                "result": [
                    {
                        "type": "videorectangle",
                        "value": {
                            "labels": ["drinking"],
                            "sequence": [
                                {
                                    "frame": 3,
                                    "time": 0.1,
                                    "x": 10,
                                    "y": 20,
                                    "width": 30,
                                    "height": 40,
                                },
                                {
                                    "frame": 30,
                                    "time": 1.0,
                                    "x": 20,
                                    "y": 30,
                                    "width": 40,
                                    "height": 50,
                                },
                            ],
                        },
                    }
                ]
            }
        ],
    }
    to_one_fps(task)
    value = task["annotations"][0]["result"][0]["value"]
    assert value["framesCount"] == 1
    assert [point["frame"] for point in value["sequence"]] == [1]
    assert value["sequence"][0]["time"] == 0.5
    assert value["sequence"][0]["x"] == pytest.approx(14.444444444444445)


def test_one_fps_snippet_metrics_match_single_frame_decisions():
    images = [
        {
            "id": task_id * 10 + 1,
            "file_name": f"task_{task_id:05d}_frame_01.jpg",
            "width": 640,
            "height": 640,
        }
        for task_id in (1, 2)
    ]
    annotations = [
        {
            "id": 1,
            "image_id": 11,
            "category_id": 1,
            "bbox": [100, 100, 100, 100],
            "area": 10000,
            "iscrowd": 0,
        }
    ]
    ground_truth = {
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": 1, "name": "drinking"},
            {"id": 2, "name": "phone_use"},
        ],
    }
    predictions = [
        {
            "image_id": 11,
            "category_id": 1,
            "bbox": [100, 100, 100, 100],
            "score": 0.9,
        }
    ] + [
        {
            "image_id": 21,
            "category_id": 2,
            "bbox": [50, 50, 50, 50],
            "score": 0.8,
        }
    ]
    metrics = snippet_operating_point_metrics(ground_truth, predictions, 0.5)
    assert metrics["snippets"] == 2
    assert metrics["frames_per_snippet"] == 1
    assert metrics["tp"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 0
    assert metrics["false_positive_snippets"] == 1
    assert metrics["false_positive_snippets_per_100_negative_snippets"] == 100.0
