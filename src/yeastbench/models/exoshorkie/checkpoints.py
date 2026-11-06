"""Metadata for Jonathan Mandl's six released distilled students.

The files are Keras HDF5 weights, reconstructed by the upstream loader:
https://github.com/OrensteinLab/ExoShorkie/blob/v1.0.1/scripts/ISM_student.py

Normalization follows the authors' whole-dataset notebook convention in
``constants.py``. It is not the per-CV-fold training normalization.
This module describes downloads; loading a model never downloads files.
"""

from dataclasses import dataclass
from types import MappingProxyType

from yeastbench.models.exoshorkie.constants import NORMALIZATION_STATS


@dataclass(frozen=True)
class CheckpointInfo:
    name: str
    filename: str
    url: str
    mu: float
    sigma: float


_BUCKET = "https://storage.googleapis.com/exoshhorkie_distilled"
CHECKPOINTS = MappingProxyType(
    {
        name: CheckpointInfo(
            name=name,
            filename=f"student_{name}_distilled.h5",
            url=f"{_BUCKET}/student_{name}_distilled.h5",
            mu=NORMALIZATION_STATS[name][0],
            sigma=NORMALIZATION_STATS[name][1],
        )
        for name in ("HPRT1", "HPRT1R", "dChr", "chr7", "Mpneumo", "Mmmyco")
    }
)


def get_checkpoint(variant: str) -> CheckpointInfo:
    """Resolve an exact upstream variant name, including capitalization."""
    try:
        return CHECKPOINTS[variant]
    except KeyError:
        raise ValueError(
            f"Unknown ExoShorkie variant {variant!r}; choose from "
            f"{', '.join(CHECKPOINTS)}"
        ) from None
