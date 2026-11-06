"""Yorzoi adapter for the Hong et al. IGR-insertion position-effect task.

Same construct as the Shorkie Hong adapter (native R64-5-1 spliced
with the constant ``TDH3p-mCherry-ADH1t`` cassette at the experimental
cut site, mCherry-CDS readout, no REF baseline) but with Yorzoi's
4992 bp input and 300 output bins. Cassette is always integrated on
the chromosome's ``+`` strand (donor PCR-primer orientation), so we
always use the ``+``-strand track subset (tracks 0–80).

IntTrain-fitted IntProp ρ: exposes
``predict_diagnostic_readouts(loci)`` returning per-locus signed
scores for multiple (RNA-seq track subset × readout region)
combinations. Yorzoi has no histone-mark or nucleosome-density
tracks (RNA-seq only), so the selection space is restricted to
RNA-seq track subsets: all + tracks (baseline), JS94 only (WT yeast
deep replicates), SCRaMBLE strains only, and the Brooks Nanopore
yeast block. All RNA-seq groups get biological sign +1.

See ``docs/benchmarks/hong_igr.md`` for the full Primary +
IntTrain-fitted design.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Sequence

import numpy as np
from tqdm import tqdm

from yeastbench.adapters._genome import one_hot_encode_channels_first
from yeastbench.adapters._hong_scaffold import (
    DEFAULT_CASSETTE_FASTA,
    HongInsertionContext,
    HongLocus,
    aggregate_diagnostic_base_readouts,
    build_insertion_context,
    load_cassette_payload,
)
from yeastbench.adapters.protocols import IGRInsertionExpressionPredictor
from yeastbench.adapters._yorzoi_constants import (
    BIN_WIDTH,
    CROP_BP_EACH_SIDE,
    OUTPUT_BINS,
    SEQ_LEN,
)
from yeastbench.models.yorzoi import Yorzoi

log = logging.getLogger(__name__)

# Cassette is always + strand → always use the + tracks (0..80).
YORZOI_PLUS_TRACK_START = 0
YORZOI_PLUS_TRACK_END = 81


def _yorzoi_diagnostic_track_groups(
    track_annotation_path: Path,
) -> list[tuple[str, list[int], int]]:
    """Parse Yorzoi's track_annotation.json and build the per-track-group
    index lists used by the IntTrain-fitted IntProp ρ candidate space.
    Index into the + strand only (positions 0..80 within the 162-track
    output)."""
    ann = json.loads(Path(track_annotation_path).read_text())
    plus_tracks = ann["+"]
    assert len(plus_tracks) == 81, (
        f"expected 81 + strand tracks, got {len(plus_tracks)}"
    )
    js94_re = re.compile(r"^JS94_")
    js_re = re.compile(r"^JS\d+_")
    js94_idx = [i for i, n in enumerate(plus_tracks) if js94_re.match(n)]
    scramble_idx = [
        i for i, n in enumerate(plus_tracks)
        if js_re.match(n) and not js94_re.match(n)
    ]
    brooks_idx = [i for i, n in enumerate(plus_tracks) if js_re.match(n)]
    return [
        ("All + tracks (baseline)", list(range(81)),  +1),
        ("JS94 (WT yeast)",         js94_idx,         +1),
        ("SCRaMBLE strains",        scramble_idx,     +1),
        ("Brooks Nanopore yeast",   brooks_idx,       +1),
    ]


# The Primary readout name (used by the Hong benchmark to identify the
# Primary among the diagnostic dict's keys).
PRIMARY_READOUT_NAME = "All + tracks (baseline) × cassette CDS"


class YorzoiHongPredictor(IGRInsertionExpressionPredictor):
    def __init__(
        self,
        model: Yorzoi,
        fasta_path: str | Path,
        cassette_fasta: str | Path | None = None,
        batch_size: int = 32,
        track_annotation_path: str | Path | None = None,
    ) -> None:
        import pysam

        self.model = model
        self.fasta = pysam.FastaFile(str(fasta_path))
        self.payload = load_cassette_payload(
            cassette_fasta if cassette_fasta is not None else DEFAULT_CASSETTE_FASTA
        )
        self.batch_size = batch_size
        if track_annotation_path is None:
            track_annotation_path = (
                Path(__file__).resolve().parents[3]
                / "yorzoi" / "track_annotation.json"
            )
        self._track_annotation_path = Path(track_annotation_path)
        self._diagnostic_track_groups = (
            _yorzoi_diagnostic_track_groups(self._track_annotation_path)
            if self._track_annotation_path.exists() else []
        )

    @classmethod
    def from_pretrained(
        cls,
        hf_repo: str,
        fasta_path: str | Path,
        cassette_fasta: str | Path | None = None,
        device: str = "cuda",
        batch_size: int = 32,
        use_rc: bool = True,
        autocast: bool = True,
        track_annotation_path: str | Path | None = None,
        hf_revision: str | None = None,
    ) -> "YorzoiHongPredictor":
        return cls(
            Yorzoi.from_pretrained(
                hf_repo, device=device, use_rc=use_rc, autocast=autocast,
                hf_revision=hf_revision,
            ),
            fasta_path=fasta_path,
            cassette_fasta=cassette_fasta,
            batch_size=batch_size,
            track_annotation_path=track_annotation_path,
        )

    # ── Window-building helpers ──────────────────────────────────────

    def _build_contexts(
        self, loci: Sequence[HongLocus]
    ) -> tuple[list[tuple[int, HongInsertionContext]], np.ndarray]:
        contexts: list[tuple[int, HongInsertionContext]] = []
        for i, locus in enumerate(loci):
            ctx = build_insertion_context(
                locus, self.payload, self.fasta,
                SEQ_LEN, CROP_BP_EACH_SIDE, BIN_WIDTH, OUTPUT_BINS,
            )
            if ctx is not None:
                contexts.append((i, ctx))
        valid_idx = np.array([i for i, _ in contexts], dtype=np.int64)
        return contexts, valid_idx

    # ── Primary readout (fast path) ──────────────────────────────────

    def predict_expressions(self, loci: Sequence[HongLocus]) -> np.ndarray:
        import torch as _torch

        scores = np.full(len(loci), np.nan, dtype=np.float64)
        contexts, _ = self._build_contexts(loci)

        for bs in tqdm(
            range(0, len(contexts), self.batch_size), desc="Yorzoi Hong (primary)"
        ):
            batch = contexts[bs : bs + self.batch_size]
            x = _torch.from_numpy(
                np.stack([
                    one_hot_encode_channels_first(c.window_seq).T
                    for _, c in batch
                ])
            ).to(self.model.device)
            with _torch.no_grad():
                # Per-base raw counts: the Borzoi inverse is applied per pass
                # before RC-averaging inside the wrapper, then unbinned.
                perbase = self.model.forward_tracks_perbase(x)  # (B, 162, 3000)
            for j, (row_idx, ctx) in enumerate(batch):
                base_idx = ctx.mcherry_base_positions
                if base_idx.size == 0:
                    continue  # readout outside the crop → leave score NaN
                base_t = _torch.as_tensor(
                    base_idx, device=self.model.device, dtype=_torch.long
                )
                per_track = perbase[j].index_select(1, base_t).sum(dim=1)  # (162,)
                scores[row_idx] = float(
                    per_track[YORZOI_PLUS_TRACK_START:YORZOI_PLUS_TRACK_END].mean().item()
                )

        return scores

    # ── Diagnostic readouts (IntTrain-fitted candidate space) ────────

    def predict_diagnostic_readouts(
        self, loci: Sequence[HongLocus]
    ) -> dict[str, np.ndarray]:
        """Score every (RNA-seq track subset × readout region) combination
        + the Primary, with biological signs applied (all +1 for
        Yorzoi). Returns a flat dict keyed by descriptive readout names,
        plus a ``'primary'`` alias."""
        import torch as _torch

        if not self._diagnostic_track_groups:
            return {"primary": self.predict_expressions(loci)}

        contexts, valid_idx = self._build_contexts(loci)
        n = len(loci)

        # Run inference once with all + tracks, retain full per-base
        # (n, OUTPUT_BINS*BIN_WIDTH, 81) raw counts.
        out_len = OUTPUT_BINS * BIN_WIDTH
        full_cov = np.full((n, out_len, 81), np.nan, dtype=np.float32)
        for bs in tqdm(
            range(0, len(contexts), self.batch_size),
            desc="Yorzoi Hong (diagnostic)",
        ):
            batch = contexts[bs : bs + self.batch_size]
            x = _torch.from_numpy(
                np.stack([
                    one_hot_encode_channels_first(c.window_seq).T
                    for _, c in batch
                ])
            ).to(self.model.device)
            with _torch.no_grad():
                perbase = self.model.forward_tracks_perbase(x)  # (B, 162, 3000)
            plus_only = perbase[:, :81, :].permute(0, 2, 1).cpu().numpy()  # (B, 3000, 81)
            for j, (row_idx, _ctx) in enumerate(batch):
                full_cov[row_idx] = plus_only[j]

        group_cov: dict[str, np.ndarray] = {}
        for group_name, idx, _sign in self._diagnostic_track_groups:
            if not idx:
                continue
            # Cross-track mean on raw per-base counts.
            group_cov[group_name] = full_cov[:, :, idx].mean(axis=2)
        readouts = aggregate_diagnostic_base_readouts(
            group_cov, contexts, self._diagnostic_track_groups,
            n, CROP_BP_EACH_SIDE, out_len,
        )
        readouts["primary"] = readouts[PRIMARY_READOUT_NAME]
        return readouts
