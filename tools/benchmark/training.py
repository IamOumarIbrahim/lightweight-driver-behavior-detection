"""Shared training-budget helpers for the frozen DMS-Eval protocol."""

from __future__ import annotations


def accumulation_loss_scale(
    *,
    batch_index: int,
    total_batches: int,
    accumulation_steps: int,
    current_batch_size: int,
    total_samples: int,
    physical_batch_size: int,
    batch_loss_reduction: str,
) -> float:
    """Return a sample-correct accumulation multiplier for one mini-batch.

    Mean-reduced losses are weighted by the mini-batch's share of the actual
    window. Sum-reduced losses use one constant multiplier across the window
    so every image has equal weight while a short final window retains the
    nominal effective-batch magnitude.
    """

    values = {
        "total_batches": total_batches,
        "accumulation_steps": accumulation_steps,
        "current_batch_size": current_batch_size,
        "total_samples": total_samples,
        "physical_batch_size": physical_batch_size,
    }
    if batch_index < 0 or batch_index >= total_batches:
        raise ValueError("batch_index is outside the epoch")
    if any(value < 1 for value in values.values()):
        raise ValueError(f"Training dimensions must be positive: {values}")
    if current_batch_size > physical_batch_size:
        raise ValueError("current_batch_size exceeds the frozen physical batch")
    if batch_loss_reduction not in {"mean", "sum"}:
        raise ValueError("batch_loss_reduction must be 'mean' or 'sum'")

    window_start = (batch_index // accumulation_steps) * accumulation_steps
    window_end = min(window_start + accumulation_steps, total_batches)
    first_sample = window_start * physical_batch_size
    final_sample = min(total_samples, window_end * physical_batch_size)
    actual_window_samples = final_sample - first_sample
    if actual_window_samples < current_batch_size:
        raise ValueError(
            "Accumulation window contains fewer samples than the current batch"
        )
    if batch_loss_reduction == "mean":
        return float(current_batch_size / actual_window_samples)
    nominal_window_samples = accumulation_steps * physical_batch_size
    return float(nominal_window_samples / actual_window_samples)
