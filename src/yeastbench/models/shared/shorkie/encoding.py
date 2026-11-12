"""Species encoding for Shorkie-family models."""
from __future__ import annotations

import torch

DNA_CHANNELS = 4
NUM_SPECIES = 165
SPECIES_OFFSET = 5
R64_SPECIES_INDEX = 109


def expand_species_encoding(
    x: torch.Tensor, *, encode_n: bool = False,
    species_index: int = R64_SPECIES_INDEX,
) -> torch.Tensor:
    """Expand channels-first DNA to the 170 input features used by the trunk.

    Four input channels are A/C/G/T. With ``encode_n=True``, an all-zero
    position represents N and sets channel 4; otherwise channel 4 is zero,
    preserving the original Shorkie encoding. Five-channel A/C/G/T/N input
    supplies the N indicator explicitly, including when other ambiguous
    bases should remain all-zero. The yeast species channel is always set.
    """
    if x.ndim != 3 or x.shape[1] not in (DNA_CHANNELS, SPECIES_OFFSET):
        raise ValueError("Expected DNA input with shape (B, 4, L) or (B, 5, L)")
    if not 0 <= species_index < NUM_SPECIES:
        raise ValueError(f"Species index must be in [0, {NUM_SPECIES})")
    batch, channels, length = x.shape
    expanded = x.new_zeros(batch, SPECIES_OFFSET + NUM_SPECIES, length)
    expanded[:, :channels, :] = x
    if channels == DNA_CHANNELS and encode_n:
        expanded[:, DNA_CHANNELS, :] = (x == 0).all(dim=1).to(x.dtype)
    expanded[:, SPECIES_OFFSET + species_index, :] = 1
    return expanded


def reverse_complement(x: torch.Tensor) -> torch.Tensor:
    """Reverse DNA positions and complement A/C/G/T, preserving an N channel."""
    if x.ndim != 3 or x.shape[1] not in (DNA_CHANNELS, SPECIES_OFFSET):
        raise ValueError("Expected DNA input with shape (B, 4, L) or (B, 5, L)")
    dna = x[:, :DNA_CHANNELS, :].flip(dims=(1, 2))
    if x.shape[1] == SPECIES_OFFSET:
        return torch.cat((dna, x[:, DNA_CHANNELS:, :].flip(dims=(2,))), dim=1)
    return dna
