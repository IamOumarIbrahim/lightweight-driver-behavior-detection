from tools.benchmark.protocol import validate_protocol
from tools.data.freeze_nir_ratios import nested_subset
from tools.data.prepare_nir import (
    interpolate_box,
    load_ratio_tasks,
    result_value,
    signature,
    validate_counts,
)
from tools.setup.label_studio import to_ten_fps


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


def test_label_studio_tracklet_is_retimed_to_ten_fps():
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
    to_ten_fps(task)
    value = task["annotations"][0]["result"][0]["value"]
    assert value["framesCount"] == 10
    assert [point["frame"] for point in value["sequence"]] == list(range(1, 11))
    assert value["sequence"][-1]["time"] == 1.0
