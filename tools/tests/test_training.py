import pytest

from tools.benchmark.training import accumulation_loss_scale


def test_short_final_window_is_sample_correct():
    values = [
        accumulation_loss_scale(
            batch_index=index,
            total_batches=3,
            accumulation_steps=4,
            current_batch_size=size,
            total_samples=18,
            physical_batch_size=8,
            batch_loss_reduction="mean",
        )
        for index, size in enumerate((8, 8, 2))
    ]
    assert values == pytest.approx([8 / 18, 8 / 18, 2 / 18])
