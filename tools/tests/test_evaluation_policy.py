from tools.benchmark.paths import is_authoritative_rgb_yolo, result_dir


def test_six_published_rgb_yolo_runs_are_frozen():
    for model in ("yolo11n", "yolo26n"):
        for seed in (13, 37, 73):
            path = result_dir("RGB", model, seed=seed)
            assert is_authoritative_rgb_yolo("RGB", model, path)
