"""Shorkie's native multi-track head on the shared PyTorch backbone."""
from __future__ import annotations

from pathlib import Path

import h5py
import torch
from torch import nn
from torch.nn import functional as F

from yeastbench.models.shared.shorkie.backbone import ShorkieBackbone
from yeastbench.models.shared.shorkie.encoding import expand_species_encoding
from yeastbench.models.shared.shorkie.keras_weights import (
    get_tf_layer,
    load_linear,
    load_trunk_weights,
)


class ShorkieModule(nn.Module):
    """Shorkie backbone with its dense/softplus count head.

    Accepts ``(B, 4, L)`` DNA and returns ``(B, output_bins, tracks)``.
    The public constructor and H5 loader retain the original interface.
    PyTorch state dictionaries now group trunk tensors under ``backbone.``.
    """

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = config
        self.backbone = ShorkieBackbone(config)
        self.head = nn.Linear(self.backbone.out_channels, config["head"]["units"])

    def _expand_species_encoding(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != 4:
            raise ValueError("Expected Shorkie input with shape (B, 4, L)")
        return expand_species_encoding(x, encode_n=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(self._expand_species_encoding(x))
        return F.softplus(self.head(features))

    @classmethod
    def from_tf_checkpoint(
        cls, config: dict, h5_path: str | Path,
    ) -> ShorkieModule:
        """Load all trunk and head weights from an original Keras H5 file."""
        model = cls(config)
        with h5py.File(h5_path, "r") as root:
            head_index = load_trunk_weights(model.backbone, root)
            head_name = "dense" if head_index == 0 else f"dense_{head_index}"
            load_linear(model.head, get_tf_layer(root, head_name))
        return model
