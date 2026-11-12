"""Shared decoded-coverage readouts for a student or a student ensemble."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

from yeastbench.models.shared.shorkie.encoding import reverse_complement
from yeastbench.models.readouts import select_strand, sum_binned_bases


class CoverageReadouts:
    """Readouts built on forward_coverage_binned and model output geometry."""

    def sum_region(self, coverage: torch.Tensor, base_positions, strand: str):
        """Sum exact exon/CDS bases from decoded ``(..., 2, bins)`` coverage.

        Coordinates are relative to the cropped output. Task adapters supply
        positions and strand; bin geometry and aggregation stay in the model.
        """
        return sum_binned_bases(
            select_strand(coverage, strand), base_positions, self.bin_width,
        )

    def predict_region_sums(self, x: torch.Tensor, regions, strands) -> torch.Tensor:
        """Predict one exon/CDS sum per input, evaluating only its needed strand.

        Regions contain cropped-output base offsets in input coordinates.
        Reverse-strand inputs are reverse-complemented for prediction and their
        bins realigned before aggregation. This avoids a redundant opposite-
        strand pass for large single-strand reporter libraries.
        """
        import torch

        if len(regions) != len(x) or len(strands) != len(x):
            raise ValueError("Each input needs one region and one strand")
        if any(strand not in ("+", "-") for strand in strands):
            raise ValueError("Strands must be '+' or '-'")
        result = torch.empty(len(x), device=self.device, dtype=torch.float32)
        for strand in ("+", "-"):
            rows = [i for i, value in enumerate(strands) if value == strand]
            if not rows:
                continue
            indices = torch.tensor(rows, device=x.device)
            inputs = x.index_select(0, indices)
            if strand == "-":
                inputs = reverse_complement(inputs)
            coverage = self.forward_coverage_binned(inputs)
            if strand == "-":
                coverage = coverage.flip(dims=[1])
            for j, row in enumerate(rows):
                result[row] = sum_binned_bases(coverage[j], regions[row], self.bin_width)
        return result

    def forward_strands_binned(self, x: torch.Tensor) -> torch.Tensor:
        """Coverage ``(B, 2, bins)`` in input coordinates, strands ``(+,-)``.

        Reverse-complement DNA channels before adding species encoding;
        the explicit N channel, if supplied, only reverses positions.
        Convert each orientation to coverage before any strand reduction.
        """
        import torch

        plus = self.forward_coverage_binned(x)
        x_rc = reverse_complement(x)
        minus = self.forward_coverage_binned(x_rc).flip(dims=[1])
        return torch.stack((plus, minus), dim=1)

    def forward_strands_perbase(self, x: torch.Tensor) -> torch.Tensor:
        """Stranded coverage ``(B, 2, bins * bin_width)`` over the crop."""
        binned = self.forward_strands_binned(x)
        return binned.repeat_interleave(self.bin_width, dim=-1) / self.bin_width

    def forward_unstranded_perbase(self, x: torch.Tensor) -> torch.Tensor:
        """Sum both strands and spread each bin total uniformly over bases."""
        binned = self.forward_strands_binned(x).sum(dim=1)
        return binned.repeat_interleave(self.bin_width, dim=-1) / self.bin_width
