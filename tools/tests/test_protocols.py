from tools.benchmark.protocol import load_yaml, validate_protocol


def test_rgb_protocol_and_fingerprints():
    training = validate_protocol("RGB")["training"]
    assert training["epochs"] == 220
    assert training["run_seeds"] == [13, 37, 73]


def test_nir_protocol_and_fingerprints():
    protocol = validate_protocol("NIR")
    assert [item["id"] for item in protocol["models"]] == [
        "yolo11n",
        "yolo26n",
        "rtmdet_tiny",
        "efficientdet_d1",
        "dfine_n",
    ]
    assert protocol["training"]["epochs"] == 100
    assert protocol["training"]["seed"] == 13
    assert protocol["training"]["ratios"] == ["1to2", "1to6"]
    assert protocol["dataset"]["snippet_fps"] == 1
    assert protocol["dataset"]["frames_per_snippet"] == 1
    assert protocol["dataset"]["sample_time_seconds"] == 0.5
    assert protocol["dataset"]["source_frame_offset"] == 14
    assert protocol["training"]["checkpoint_retention"]["periodic_every_epochs"] == 100

    dfine = load_yaml("configs/NIR/dfine/base.yml")
    assert dfine["epochs"] == 100
    policy = dfine["train_dataloader"]["dataset"]["transforms"]["policy"]
    assert policy["epoch"] == 67
    assert dfine["train_dataloader"]["collate_fn"]["stop_epoch"] == 67
    assert protocol["training"]["optimization"]["rtmdet"]["augmentation_stop_epoch"] == 93
    assert protocol["training"]["optimization"]["efficientdet"]["warmup_epochs"] == 5
