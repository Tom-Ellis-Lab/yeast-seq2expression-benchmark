"""Coverage readouts in model-output coordinates, independent of task annotations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def sum_binned_bases(
    coverage: torch.Tensor,
    base_positions: torch.Tensor,
    bin_width: int,
) -> torch.Tensor:
    """Sum exact bases without materializing uniformly unbinned coverage.

    ``coverage[..., bin]`` contains decoded bin totals. Positions are unique,
    zero-based offsets into the cropped output (e.g. a union of exons).
    A partial bin contributes its overlapping fraction of the total.
    """
    import torch

    if bin_width <= 0:
        raise ValueError("bin_width must be positive")
    positions = torch.as_tensor(
        base_positions, device=coverage.device, dtype=torch.long
    )
    if positions.ndim != 1:
        raise ValueError("base_positions must be one-dimensional")
    if positions.numel() and (
        positions.min() < 0 or positions.max() >= coverage.shape[-1] * bin_width
    ):
        raise ValueError("base_positions outside the cropped output")
    counts = torch.bincount(positions // bin_width, minlength=coverage.shape[-1])
    weights = counts.to(coverage.dtype) / bin_width
    return (coverage * weights).sum(dim=-1)


def select_strand(coverage: torch.Tensor, strand: str) -> torch.Tensor:
    """Select from ``(..., 2, positions)`` with strand order (+, -)."""
    if strand not in ("+", "-"):
        raise ValueError(f"Expected '+' or '-', got {strand!r}")
    return coverage[..., 0 if strand == "+" else 1, :]
