"""PyTorch port of the distilled ExoShorkie student.

The shared trunk includes its final GELU, as in Baskerville's
``model_trunk``. The student adds a linear one-output head; its predictions
are normalized log coverage, with no softplus or inverse transform here.

Reference: https://github.com/OrensteinLab/ExoShorkie/blob/v1.0.1/scripts/ISM_student.py
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import h5py
import torch
from torch import nn

from yeastbench.models.shared.shorkie.backbone import ShorkieBackbone
from yeastbench.models.shared.shorkie.encoding import expand_species_encoding
from yeastbench.models.shared.shorkie.keras_weights import (
    get_tf_layer,
    load_linear,
    load_trunk_weights,
)


class ExoShorkieModule(nn.Module):
    """One distilled student, returning ``(batch, output_bins)``.

    Input is channels-first DNA: ``(batch, 4, sequence_length)`` in ACGT
    order, with all-zero positions interpreted as N. An explicit fifth N
    channel is also accepted; use it to distinguish N from other ambiguous
    bases, which the upstream encoder leaves all-zero. Species encoding is
    added internally with the S. cerevisiae index used by the authors.
    """

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = deepcopy(config)
        self.backbone = ShorkieBackbone(self.config)
        self.distill_head = nn.Linear(self.backbone.out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = expand_species_encoding(x, encode_n=True)
        return self.distill_head(self.backbone(x)).squeeze(-1)

    @classmethod
    def from_tf_checkpoint(
        cls,
        config: dict,
        h5_path: str | Path,
    ) -> ExoShorkieModule:
        """Load all trunk and ``distill_head`` weights from a local H5 file.

        ``config`` is the model subsection of the upstream parameters.
        The original Shorkie head configuration, if present, is ignored.
        """
        model = cls(config)
        with h5py.File(h5_path, "r") as root:
            load_trunk_weights(model.backbone, root, head_name="distill_head")
            load_linear(model.distill_head, get_tf_layer(root, "distill_head"))
        return model
