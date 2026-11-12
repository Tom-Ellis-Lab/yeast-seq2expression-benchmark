"""Shorkie's native multi-track head on the shared PyTorch backbone."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import h5py
import torch
from torch import nn
from torch.nn import functional as F

from yeastbench.models.shared.shorkie.backbone import ShorkieBackbone
from yeastbench.models.shared.shorkie.encoding import (
    DNA_CHANNELS,
    NUM_SPECIES,
    SPECIES_OFFSET,
    expand_species_encoding,
)
from yeastbench.models.shared.shorkie.keras_weights import (
    get_tf_layer,
    load_linear,
    load_trunk_weights,
)


KNOWN_IN_CHANNELS = (DNA_CHANNELS, SPECIES_OFFSET + NUM_SPECIES)


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
        in_channels = self.backbone.conv_dna.conv.in_channels
        if in_channels not in KNOWN_IN_CHANNELS:
            raise ValueError(f"conv_dna input channels must be one of {KNOWN_IN_CHANNELS}")
        self._expects_species_encoding = in_channels != DNA_CHANNELS
        self.head = nn.Linear(self.backbone.out_channels, config["head"]["units"])

    def _expand_species_encoding(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != 4:
            raise ValueError("Expected Shorkie input with shape (B, 4, L)")
        return expand_species_encoding(x, encode_n=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != DNA_CHANNELS:
            raise ValueError("Expected Shorkie input with shape (B, 4, L)")
        # Supervised-from-scratch checkpoints take DNA without species channels.
        if self._expects_species_encoding:
            x = self._expand_species_encoding(x)
        features = self.backbone(x)
        return F.softplus(self.head(features))

    @classmethod
    def from_tf_checkpoint(
        cls, config: dict, h5_path: str | Path,
    ) -> ShorkieModule:
        """Load a published DNA-only (4) or species-encoded (170) checkpoint.

        Infer input width from its first kernel without modifying the caller's
        configuration, which the ensemble wrapper reuses across folds.
        """
        in_channels = _tf_conv_dna_in_channels(h5_path)
        model = cls(_with_conv_dna_in_channels(config, in_channels))
        with h5py.File(h5_path, "r") as root:
            head_index = load_trunk_weights(model.backbone, root)
            head_name = "dense" if head_index == 0 else f"dense_{head_index}"
            load_linear(model.head, get_tf_layer(root, head_name))
        return model


def _tf_conv_dna_in_channels(h5_path: str | Path) -> int:
    with h5py.File(h5_path, "r") as root:
        shape = get_tf_layer(root, "conv1d")["kernel:0"].shape
    if len(shape) != 3 or shape[1] not in KNOWN_IN_CHANNELS:
        raise ValueError(
            f"conv_dna kernel has shape {shape}; expected input channels "
            f"{KNOWN_IN_CHANNELS}. Use an original published TF checkpoint."
        )
    return shape[1]


def _with_conv_dna_in_channels(config: dict, in_channels: int) -> dict:
    config = deepcopy(config)
    config["trunk"][0]["in_channels"] = in_channels
    return config
