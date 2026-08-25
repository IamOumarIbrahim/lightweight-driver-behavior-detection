from tools.benchmark.protocol import validate_protocol


def test_rgb_protocol_and_fingerprints():
    assert validate_protocol("RGB")["training"]["run_seeds"] == [13, 37, 73]


def test_nir_protocol_and_fingerprints():
    protocol = validate_protocol("NIR")
    assert protocol["training"]["seed"] == 13
    assert protocol["training"]["ratios"] == ["1to2", "1to6"]
    assert protocol["dataset"]["snippet_fps"] == 1
    assert protocol["dataset"]["frames_per_snippet"] == 1
    assert protocol["dataset"]["sample_time_seconds"] == 0.5
    assert protocol["dataset"]["source_frame_offset"] == 14
    assert protocol["training"]["checkpoint_retention"]["periodic_every_epochs"] == 100
