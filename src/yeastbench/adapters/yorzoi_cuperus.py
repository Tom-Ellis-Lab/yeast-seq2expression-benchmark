"""Yorzoi adapter for the Cuperus 5'-UTR reporter task.

Same construct as the Shorkie Cuperus adapter (``CYC1``pr + UTR + ``HIS3`` +
``CYC1``term, embedded at a background locus, ``HIS3``-ORF readout, no REF
baseline) but with Yorzoi's 4992 bp input, 300 output bins, and strand-matched
track aggregation. The construct is always built on the + strand, so the readout
uses the plus-strand tracks (0–80); RC averaging swaps strand tracks inside the
wrapper. Contexts are streamed in batches (the random library is ~489k UTRs).
The construct is scored in the single fixed ``HIS3`` context (no marginalization).
See ``docs/benchmarks/cuperus_mpra_5utr.md``.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
from tqdm import tqdm

from yeastbench.adapters._cuperus_scaffold import (
    HIS3_BACKGROUND,
    CuperusConstruct,
    build_context,
)
from yeastbench.adapters._genome import one_hot_encode_channels_first
from yeastbench.adapters._yorzoi_constants import (
    BIN_WIDTH,
    CROP_BP_EACH_SIDE,
    OUTPUT_BINS,
    SEQ_LEN,
)
from yeastbench.adapters.protocols import FivePrimeUtrReporterExpressionPredictor
from yeastbench.models.yorzoi import Yorzoi

log = logging.getLogger(__name__)

# The construct is built on the + strand, so read out the plus-strand tracks.
PLUS_TRACKS = (0, 81)


class YorzoiCuperusPredictor(FivePrimeUtrReporterExpressionPredictor):
    def __init__(
        self,
        model: Yorzoi,
        fasta_path: str | Path,
        construct_json: str | Path | None = None,
        batch_size: int = 32,
    ) -> None:
        import pysam

        self.model = model
        self.fasta = pysam.FastaFile(str(fasta_path))
        self.construct = CuperusConstruct.from_json(construct_json) if construct_json else CuperusConstruct.from_json()
        self.batch_size = batch_size

    @classmethod
    def from_pretrained(
        cls,
        hf_repo: str,
        fasta_path: str | Path,
        construct_json: str | Path | None = None,
        device: str = "cuda",
        batch_size: int = 32,
        use_rc: bool = True,
        autocast: bool = True,
        hf_revision: str | None = None,
    ) -> "YorzoiCuperusPredictor":
        return cls(
            Yorzoi.from_pretrained(
                hf_repo, device=device, use_rc=use_rc, autocast=autocast,
                hf_revision=hf_revision,
            ),
            fasta_path=fasta_path,
            construct_json=construct_json,
            batch_size=batch_size,
        )

    def _contexts(self, utrs: Sequence[str]) -> Iterator[tuple[int, object]]:
        for ui, utr in enumerate(utrs):
            ctx = build_context(
                self.construct, utr, HIS3_BACKGROUND, self.fasta,
                seq_len=SEQ_LEN, crop_bp_each_side=CROP_BP_EACH_SIDE,
                bin_width=BIN_WIDTH, output_bins=OUTPUT_BINS,
            )
            if ctx is not None and ctx.readout_base_positions.size:
                yield ui, ctx

    def predict_utr_expressions(self, utrs: Sequence[str]) -> np.ndarray:
        import torch as _torch

        n = len(utrs)
        out = np.full(n, np.nan, dtype=np.float64)  # UTRs with no usable context stay NaN
        ts, te = PLUS_TRACKS

        def flush(buf: list) -> None:
            x = _torch.from_numpy(
                np.stack([one_hot_encode_channels_first(c.window_seq).T for _, c in buf])
            ).to(self.model.device)
            with _torch.no_grad():
                perbase = self.model.forward_tracks_perbase(x)  # (B, 162, OUT_LEN)
            for j, (ui, ctx) in enumerate(buf):
                base_t = _torch.as_tensor(
                    ctx.readout_base_positions, device=self.model.device, dtype=_torch.long
                )
                per_track = perbase[j].index_select(1, base_t).sum(dim=1)  # (162,)
                out[ui] = float(per_track[ts:te].mean().item())

        buf: list = []
        for item in tqdm(self._contexts(utrs), total=n, desc="Yorzoi Cuperus"):
            buf.append(item)
            if len(buf) >= self.batch_size:
                flush(buf)
                buf = []
        if buf:
            flush(buf)
        return out
