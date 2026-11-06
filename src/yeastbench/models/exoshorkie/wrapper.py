"""Inference and coverage conversion for one distilled ExoShorkie model.

Forward and reverse-complement inputs predict opposite genomic strands.
They remain separate until an explicit unstranded sum. Normalization uses
the authors' whole-dataset notebook statistics, preserving their
``expm1(z * sigma + mu)`` convention without clipping negative values.
"""

from __future__ import annotations

import json
from contextlib import nullcontext
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from yeastbench.models.exoshorkie.nn import ExoShorkieModule

from yeastbench.models.exoshorkie.checkpoints import get_checkpoint
from yeastbench.models.shared.shorkie.encoding import reverse_complement


class ExoShorkie:
    """Single-student inference wrapper; ``variant`` selects normalization."""

    def __init__(
        self,
        model: ExoShorkieModule,
        variant: str,
        device: str | torch.device = "cuda",
        autocast: bool = False,
    ) -> None:
        import torch

        self.checkpoint = get_checkpoint(variant)
        self.variant = self.checkpoint.name
        self.device = torch.device(device)
        self.autocast = autocast
        self.model = model.to(self.device).eval()
        self.seq_len = model.backbone.seq_len
        self.output_bins = model.backbone.output_bins
        self.bin_width = model.backbone.bin_width
        self.crop_bp_each_side = model.backbone.crop_bp_each_side

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        variant: str,
        params_path: str | Path | None = None,
        device: str | torch.device = "cuda",
        autocast: bool = False,
    ) -> ExoShorkie:
        """Load a local student using its explicit variant and architecture.

        The bundled parameters reproduce the architecture in the authors'
        loader. A custom JSON file may contain the model config directly or
        under ``model``. The variant must match the checkpoint's training
        dataset; H5 weight files do not identify their normalization.
        """
        from yeastbench.models.exoshorkie.nn import ExoShorkieModule

        get_checkpoint(variant)
        if params_path is None:
            config = json.loads(
                files("yeastbench.models.exoshorkie")
                .joinpath("params.json")
                .read_text()
            )
        else:
            config = json.loads(Path(params_path).read_text())
        model = ExoShorkieModule.from_tf_checkpoint(
            config.get("model", config),
            checkpoint_path,
        )
        return cls(model, variant=variant, device=device, autocast=autocast)

    def _autocast_ctx(self):
        import torch

        if self.autocast and self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    def forward_normalized_binned(self, x: torch.Tensor) -> torch.Tensor:
        """Predict the input orientation in normalized units: ``(B, bins)``.

        Accepts four ACGT channels or five ACGTN channels, as documented on
        ``ExoShorkieModule``. Inference returns float32 on the model device.
        Use the underlying module directly when gradients are required.
        """
        import torch

        with torch.inference_mode(), self._autocast_ctx():
            return self.model(x.to(device=self.device, dtype=torch.float32)).float()

    def to_coverage(self, normalized: torch.Tensor) -> torch.Tensor:
        """Invert normalization to summed 16 bp coverage in float32.

        This follows the authors' notebook convention. These whole-dataset
        statistics are distinct from the teachers' per-CV-fold statistics.
        Predictions below the zero-coverage threshold remain negative.
        """
        import torch

        return torch.expm1(
            normalized.float() * self.checkpoint.sigma + self.checkpoint.mu
        )

    def forward_coverage_binned(self, x: torch.Tensor) -> torch.Tensor:
        """Input-orientation coverage bin totals: ``(B, bins)``."""
        return self.to_coverage(self.forward_normalized_binned(x))

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
