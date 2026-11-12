"""Yorzoi adapter for the Brooks SCRaMBLE benchmark.

Sequence-in / **per-base, untransformed** coverage-out. Yorzoi was
trained with the Borzoi-style target transform
``y = transform(bin_4bp(x))`` where
``transform(x) = min(x^0.75, 384 + sqrt(x^0.75 - 384))`` is a piecewise
power+sqrt squash (`yorzoi/yorzoi/utils.py`). Predictions therefore live
in a transformed binned space — log-ratios / shape metrics computed
directly on them do not match what you'd compute on raw pileups. This
adapter inverts the transform and unbins back to a per-base predicted
count vector so the benchmark can compare apples-to-apples with the
raw Nanopore per-base truth in the distribution.

Geometry exposed: ``seq_len`` (input length, 4992 bp) and
``crop_bp_each_side`` (996 bp). The returned vector covers the central
``seq_len - 2 * crop_bp_each_side`` = 3000 bp, base-by-base.
"""
from __future__ import annotations

import logging
from typing import ClassVar, Literal, Sequence

import numpy as np

from yeastbench.adapters._genome import one_hot_encode_channels_first
from yeastbench.adapters._yorzoi_constants import (
    BIN_WIDTH,
    BROOKS_NANOPORE_TRACK_IDS_PLUS_ALL,
    BROOKS_NANOPORE_TRACKS_BY_STRAIN,
    CROP_BP_EACH_SIDE,
    OUTPUT_BINS,
    SEQ_LEN,
    YORZOI_PLUS_TRACK_IDS,
)
from yeastbench.adapters.protocols import CoverageTrackPredictor
from yeastbench.models.yorzoi import Yorzoi

log = logging.getLogger(__name__)

N_PLUS_TRACKS = 81  # minus-track index = plus-track index + N_PLUS_TRACKS

TrackMode = Literal["all", "nanopore_all", "matched"]


class YorzoiBrooksPredictor(CoverageTrackPredictor):
    # Geometry exposed to the benchmark
    seq_len: ClassVar[int] = SEQ_LEN
    crop_bp_each_side: ClassVar[int] = CROP_BP_EACH_SIDE
    # Yorzoi is a small single-fold model; relatively large batches fit.
    batch_size: ClassVar[int] = 16
    varies_by_strain: ClassVar[bool] = True

    def __init__(
        self,
        model: Yorzoi,
        track_mode: TrackMode = "matched",
        batch_size: int = 16,
    ) -> None:
        """``track_mode`` controls which output tracks the prediction
        averages across.

        - ``"all"`` (v1): mean over all 81 strand-matched Yorzoi tracks,
          mixing protocols (Nanopore / Illumina / clone alignments).
        - ``"nanopore_all"``: mean over the 63 Brooks Nanopore direct-RNA
          tracks only (drops 11 non-Nanopore tracks; keeps every strain).
        - ``"matched"`` (default): per call, average only over the
          Nanopore tracks belonging to the supplied ``strain`` — i.e.
          the model's most in-distribution prediction for that strain.
          Requires the benchmark to pass ``strain=`` to
          ``predict_coverage_batch``. If ``strain`` is missing or absent
          from ``BROOKS_NANOPORE_TRACKS_BY_STRAIN`` the call falls back
          to ``"nanopore_all"`` and emits a debug log line.
        """
        self.model = model
        self.track_mode = track_mode
        # Per-call max batch; benchmark chunks larger inputs.
        self.batch_size = int(batch_size)

    @classmethod
    def from_pretrained(
        cls, hf_repo: str, device: str = "cuda",
        use_rc: bool = True, autocast: bool = True,
        track_mode: TrackMode = "matched",
        batch_size: int = 16,
        hf_revision: str | None = None,
    ) -> "YorzoiBrooksPredictor":
        return cls(
            Yorzoi.from_pretrained(
                hf_repo, device=device, use_rc=use_rc, autocast=autocast,
                hf_revision=hf_revision,
            ),
            track_mode=track_mode,
            batch_size=batch_size,
        )

    def _plus_axis_indices(self, strain: str | None) -> list[int]:
        """Indices on the 81-track plus axis for this prediction.
        Minus-axis indices are derived by adding N_PLUS_TRACKS."""
        if self.track_mode == "all":
            return YORZOI_PLUS_TRACK_IDS
        if self.track_mode == "nanopore_all":
            return BROOKS_NANOPORE_TRACK_IDS_PLUS_ALL
        # "matched"
        if strain is None or strain not in BROOKS_NANOPORE_TRACKS_BY_STRAIN:
            log.debug("track_mode='matched' but strain=%r not in map; "
                      "falling back to nanopore_all", strain)
            return BROOKS_NANOPORE_TRACK_IDS_PLUS_ALL
        return BROOKS_NANOPORE_TRACKS_BY_STRAIN[strain]

    def predict_coverage_batch(
        self,
        seqs: Sequence[str],
        strands: Sequence[str],
        strains: Sequence[str | None] | None = None,
    ) -> np.ndarray:
        """Batched per-base predicted Nanopore-like coverage over the
        central ``seq_len - 2*crop`` = 3000 bp of each input window,
        **already untransformed and unbinned** so the benchmark can
        compute LFC, Pearson and JSD directly against raw per-base
        pileups in the same units (predicted-count scale).

        Per-sample ``strains`` route each prediction to per-strain
        Nanopore tracks when ``track_mode == "matched"`` (the default).
        Returns shape ``(B, 3000)``."""
        import torch as _torch

        B = len(seqs)
        assert len(strands) == B
        if strains is None:
            strains = [None] * B
        else:
            assert len(strains) == B
        for s in seqs:
            assert len(s) == SEQ_LEN, (
                f"Brooks construct must be {SEQ_LEN} bp; got {len(s)}"
            )

        # One-hot stack: (B, SEQ_LEN, 4) channels-last as the model expects.
        arrs = np.stack(
            [one_hot_encode_channels_first(s).T for s in seqs], axis=0
        )
        x = _torch.from_numpy(arrs).to(self.model.device)
        with _torch.no_grad():
            # Per-base raw counts per track. The inverse Borzoi transform is
            # applied per (track, bin) BEFORE RC-averaging inside the wrapper,
            # so the track-subset mean below happens on raw counts.
            perbase = self.model.forward_tracks_perbase(x)  # (B, 162, 3000)

        # Per-sample track-subset averaging happens post-forward on the raw
        # per-base predictions — the model always emits all 162 tracks;
        # ``strain`` and ``strand`` only pick which tracks to mean across.
        out = np.empty((B, OUTPUT_BINS * BIN_WIDTH), dtype=np.float64)
        for i in range(B):
            plus_axis = self._plus_axis_indices(strains[i])
            channels = (
                plus_axis
                if strands[i] == "+"
                else [c + N_PLUS_TRACKS for c in plus_axis]
            )
            idx = _torch.tensor(channels, device=perbase.device, dtype=_torch.long)
            out[i] = perbase[i].index_select(0, idx).mean(dim=0).cpu().numpy()
        return out
