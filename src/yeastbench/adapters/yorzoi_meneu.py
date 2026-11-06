"""Yorzoi adapter for the Meneu foreign-DNA tiled-coverage benchmark.

Same model wrapper and same inverse-transform / unbinning path as the
Brooks adapter, with two task-specific differences:

1. **Track subset.** A new ``track_mode = "illumina_exo"`` averages over
   the 10 exogenous-human Illumina RNA-seq tracks
   (``YORZOI_HUMAN_ILLUMINA_PLUS_TRACK_IDS``, plus-axis idx 0..9) — the
   only foreign-DNA-in-yeast expression tracks Yorzoi was trained on, so
   the most in-distribution proxy for transcription of the bacterial
   chromosomes Meneu introduces into yeast.

2. **Unstranded output.** Meneu's truth is unstranded coverage
   (``fwd + rev``). The base Brooks adapter returns a *single* strand
   (plus tracks for ``strand == "+"``, minus otherwise). Here we override
   ``predict_coverage_batch`` to return, per sample, the mean over the
   chosen *plus* tracks PLUS the mean over the corresponding *minus*
   tracks (``plus_idx + N_PLUS_TRACKS``) — i.e. fwd + rev summed — so the
   returned per-base vector is unstranded, matching the truth.

``varies_by_strain = False`` (Meneu has no per-strain tracks; the human
RNA-seq subset is fixed). The model wrapper is untouched.
"""
from __future__ import annotations

import logging
from typing import ClassVar, Literal, Sequence

import numpy as np

from yeastbench.adapters._genome import one_hot_encode_channels_first
from yeastbench.adapters._yorzoi_constants import (
    BIN_WIDTH,
    OUTPUT_BINS,
    SEQ_LEN,
    YORZOI_HUMAN_ILLUMINA_PLUS_TRACK_IDS,
)
from yeastbench.adapters.yorzoi_brooks import N_PLUS_TRACKS, YorzoiBrooksPredictor
from yeastbench.models.yorzoi import Yorzoi

log = logging.getLogger(__name__)

# Extends the Brooks TrackMode with the Meneu-specific human-RNA-seq mode.
MeneuTrackMode = Literal["all", "nanopore_all", "matched", "illumina_exo"]


class YorzoiMeneuPredictor(YorzoiBrooksPredictor):
    """Unstranded Yorzoi coverage over the 10 exogenous-human Illumina
    tracks, for the Meneu foreign-DNA tiled-coverage benchmark."""

    # Meneu has no per-strain tracks — one fixed human-RNA-seq subset.
    varies_by_strain: ClassVar[bool] = False

    def __init__(
        self,
        model: Yorzoi,
        track_mode: MeneuTrackMode = "illumina_exo",
        batch_size: int = 16,
    ) -> None:
        super().__init__(model, track_mode=track_mode, batch_size=batch_size)

    @classmethod
    def from_pretrained(
        cls, hf_repo: str, device: str = "cuda",
        use_rc: bool = True, autocast: bool = True,
        track_mode: MeneuTrackMode = "illumina_exo",
        batch_size: int = 16,
        hf_revision: str | None = None,
    ) -> "YorzoiMeneuPredictor":
        return cls(
            Yorzoi.from_pretrained(
                hf_repo, device=device, use_rc=use_rc, autocast=autocast,
                hf_revision=hf_revision,
            ),
            track_mode=track_mode,
            batch_size=batch_size,
        )

    def _plus_axis_indices(self, strain: str | None) -> list[int]:
        """Indices on the 81-track plus axis for this prediction. The
        ``illumina_exo`` mode returns the 10 exogenous-human Illumina
        plus-tracks (idx 0..9); other modes defer to the Brooks logic.
        Minus-axis indices are derived by adding ``N_PLUS_TRACKS``."""
        if self.track_mode == "illumina_exo":
            return YORZOI_HUMAN_ILLUMINA_PLUS_TRACK_IDS
        return super()._plus_axis_indices(strain)

    def predict_coverage_batch(
        self,
        seqs: Sequence[str],
        strands: Sequence[str],
        strains: Sequence[str | None] | None = None,
    ) -> np.ndarray:
        """Batched per-base **unstranded** predicted coverage over the
        central ``seq_len - 2*crop`` = 3000 bp of each window, already
        untransformed and unbinned (raw predicted-count scale).

        Unlike the Brooks adapter (which picks a single strand), Meneu's
        truth is unstranded (``fwd + rev``), so for each sample this
        returns the mean over the chosen plus tracks PLUS the mean over
        the matching minus tracks (``plus_idx + N_PLUS_TRACKS``).

        ``strands`` and ``strains`` are accepted to satisfy the protocol
        but ignored — the output is unstranded and the human-RNA-seq
        subset is strain-independent. Returns shape ``(B, 3000)``."""
        import torch as _torch

        B = len(seqs)
        for s in seqs:
            assert len(s) == SEQ_LEN, (
                f"Meneu/Yorzoi window must be {SEQ_LEN} bp; got {len(s)}"
            )

        # One-hot stack: (B, SEQ_LEN, 4) channels-last as the model expects.
        arrs = np.stack(
            [one_hot_encode_channels_first(s).T for s in seqs], axis=0
        )
        x = _torch.from_numpy(arrs).to(self.model.device)
        with _torch.no_grad():
            # Per-base raw counts per track (all 162); inverse Borzoi
            # transform applied per (track, bin) before RC-averaging inside
            # the wrapper, so the track-subset mean below is on raw counts.
            perbase = self.model.forward_tracks_perbase(x)  # (B, 162, 3000)

        out = np.empty((B, OUTPUT_BINS * BIN_WIDTH), dtype=np.float64)
        for i in range(B):
            plus_axis = self._plus_axis_indices(
                strains[i] if strains is not None else None
            )
            minus_axis = [c + N_PLUS_TRACKS for c in plus_axis]
            plus_idx = _torch.tensor(
                plus_axis, device=perbase.device, dtype=_torch.long
            )
            minus_idx = _torch.tensor(
                minus_axis, device=perbase.device, dtype=_torch.long
            )
            fwd = perbase[i].index_select(0, plus_idx).mean(dim=0)
            rev = perbase[i].index_select(0, minus_idx).mean(dim=0)
            out[i] = (fwd + rev).cpu().numpy()  # unstranded
        return out
