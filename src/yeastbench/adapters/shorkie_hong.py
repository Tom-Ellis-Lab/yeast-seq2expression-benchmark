"""Shorkie adapter for the Hong et al. IGR-insertion position-effect task.

For each locus, build a 16 kb input = native R64-5-1 with the constant
``TDH3p-mCherry-ADH1t`` cassette spliced in at the experimental cut
site (always + strand), forward-pass through the 8-fold ensemble with
optional RC averaging, take the cross-track mean over the T0 RNA-seq
tracks, and sum over the mCherry-CDS output bins. Absolute readout
(no REF baseline; the signal of interest is the absolute mCherry level
as a function of intergenic position).

IntTrain-fitted IntProp ρ: additionally exposes
``predict_diagnostic_readouts(loci)`` returning per-locus signed
scores for **all** (track group × readout region) combinations the
adapter supports. The Hong benchmark uses this to select the best
combination on IntTrain and evaluate on IntProp. Track groups include
T0 RNA-seq plus several Chip-MNase histone-mark tracks; readout
regions span the cassette and immediate native flanks. Biological
signs are applied: active marks and RNA-seq get sign +1; nucleosome
density (H3) gets sign −1.

See ``docs/benchmarks/hong_igr.md`` for the full Primary +
IntTrain-fitted design.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
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
from yeastbench.adapters._shorkie_constants import (
    BIN_WIDTH,
    CROP_BP_EACH_SIDE,
    OUTPUT_BINS,
    SEQ_LEN,
    SHORKIE_T0_RNA_SEQ_TRACK_IDS,
)
from yeastbench.models.shorkie import Shorkie

log = logging.getLogger(__name__)


# IntTrain-fitted track groups: each entry is (group_name,
# track_indices, biological_sign). Sign is applied to the signed
# score so that higher signed score is biologically expected to
# correlate with higher fluorescence (positive ρ). Active marks and
# RNA-seq use +1; nucleosome density (H3 total) uses −1 (more
# nucleosomes → less expression).
def _shorkie_diagnostic_track_groups(
    targets_path: Path,
) -> list[tuple[str, list[int], int]]:
    """Load Shorkie's targets.txt and build the per-track-group index
    lists used by the IntTrain-fitted IntProp ρ candidate space.
    Called once at adapter construction so the index lookups happen
    on the CPU side, not in the inference loop."""
    df = pd.read_csv(targets_path, sep="\t")

    def _ids(prefix: str, group: str) -> list[int]:
        mask = (
            df["identifier"].astype(str).str.upper().str.startswith(prefix.upper())
            & (df["group"] == group)
        )
        return df.loc[mask, "index"].astype(int).tolist()

    return [
        ("RNA-seq T0",              list(SHORKIE_T0_RNA_SEQ_TRACK_IDS), +1),
        ("H3K27ac",                 _ids("H3K27AC_", "Chip-MNase"),     +1),
        ("H3K4me3",                 _ids("H3K4ME3_", "Chip-MNase"),     +1),
        ("H3K9ac",                  _ids("H3K9AC_",  "Chip-MNase"),     +1),
        ("H3K36me3",                _ids("H3K36ME3_", "Chip-MNase"),    +1),
        ("H3 (nucleosome density)", _ids("H3_S",     "Chip-MNase"),     -1),
    ]


# The Primary readout name (used by the Hong benchmark to identify the
# Primary among the diagnostic dict's keys).
PRIMARY_READOUT_NAME = "RNA-seq T0 × cassette CDS"


def _default_targets_path(params_path: str | Path) -> Path:
    adjacent = Path(params_path).with_name("targets.txt")
    if adjacent.is_file():
        return adjacent
    return Path(__file__).resolve().parents[3] / "data/models/shorkie/targets.txt"


class ShorkieHongPredictor(IGRInsertionExpressionPredictor):
    def __init__(
        self,
        model: Shorkie,
        fasta_path: str | Path,
        cassette_fasta: str | Path | None = None,
        track_subset: list[int] = SHORKIE_T0_RNA_SEQ_TRACK_IDS,
        batch_size: int = 16,
        targets_path: str | Path | None = None,
    ) -> None:
        import pysam
        import torch as _torch

        self.model = model
        self.fasta = pysam.FastaFile(str(fasta_path))
        self.payload = load_cassette_payload(
            cassette_fasta if cassette_fasta is not None else DEFAULT_CASSETTE_FASTA
        )
        self.track_subset = list(track_subset)
        self.batch_size = batch_size
        self._track_idx_t = _torch.tensor(
            self.track_subset, device=self.model.device, dtype=_torch.long
        )
        # Resolve targets.txt path for IntTrain-fitted track groups
        if targets_path is None:
            targets_path = (
                Path(__file__).resolve().parents[3]
                / "data" / "models" / "shorkie" / "targets.txt"
            )
        self._targets_path = Path(targets_path)
        self._diagnostic_track_groups = (
            _shorkie_diagnostic_track_groups(self._targets_path)
            if self._targets_path.exists() else []
        )

    @classmethod
    def from_checkpoints(
        cls,
        params_path: str | Path,
        checkpoint_paths: Sequence[str | Path],
        fasta_path: str | Path,
        cassette_fasta: str | Path | None = None,
        track_subset: list[int] = SHORKIE_T0_RNA_SEQ_TRACK_IDS,
        device: str = "cuda",
        batch_size: int = 16,
        use_rc: bool = True,
        targets_path: str | Path | None = None,
    ) -> "ShorkieHongPredictor":
        return cls(
            Shorkie.from_checkpoints(
                params_path, checkpoint_paths, device=device, use_rc=use_rc,
            ),
            fasta_path=fasta_path,
            cassette_fasta=cassette_fasta,
            track_subset=list(track_subset),
            batch_size=batch_size,
            targets_path=(
                _default_targets_path(params_path) if targets_path is None else targets_path
            ),
        )

    # ── Window-building helpers (shared between primary + diagnostic) ─

    def _build_contexts(
        self, loci: Sequence[HongLocus]
    ) -> tuple[list[tuple[int, HongInsertionContext]], np.ndarray]:
        """Build per-locus insertion contexts + the index of loci that
        produced a valid window."""
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

    # ── Primary readout (fast path; no diagnostic groups) ─────────────

    def predict_expressions(self, loci: Sequence[HongLocus]) -> np.ndarray:
        import torch as _torch

        scores = np.full(len(loci), np.nan, dtype=np.float64)
        contexts, _ = self._build_contexts(loci)

        for bs in tqdm(
            range(0, len(contexts), self.batch_size), desc="Shorkie Hong (primary)"
        ):
            batch = contexts[bs : bs + self.batch_size]
            x = _torch.from_numpy(
                np.stack([
                    one_hot_encode_channels_first(c.window_seq) for _, c in batch
                ])
            ).to(self.model.device)
            with _torch.no_grad():
                # (B, OUTPUT_BINS*BIN_WIDTH) per-base raw counts (unbin-only;
                # softplus head is already raw counts).
                cov = self.model.forward_track_mean_perbase(x, self._track_idx_t)
            for j, (row_idx, ctx) in enumerate(batch):
                base_idx = ctx.mcherry_base_positions
                if base_idx.size == 0:
                    continue  # readout outside the crop → leave score NaN
                base_t = _torch.as_tensor(
                    base_idx, device=self.model.device, dtype=_torch.long
                )
                scores[row_idx] = float(cov[j].index_select(0, base_t).sum().item())

        return scores

    # ── Diagnostic readouts (all track groups × all regions) ──────────

    def predict_diagnostic_readouts(
        self, loci: Sequence[HongLocus]
    ) -> dict[str, np.ndarray]:
        """Score every (track group × readout region) combination + the
        Primary, with biological signs applied. Returns a flat dict
        keyed by descriptive readout names. Includes a ``'primary'`` key
        (same scores as the entry named ``PRIMARY_READOUT_NAME``)."""
        import torch as _torch

        if not self._diagnostic_track_groups:
            # No targets.txt loaded — degrade to primary-only.
            return {"primary": self.predict_expressions(loci)}

        contexts, valid_idx = self._build_contexts(loci)
        n = len(loci)
        out_len = OUTPUT_BINS * BIN_WIDTH
        # Per-locus per-base coverage, per track group.
        # cov_per_group[group_name] is (n, OUTPUT_BINS*BIN_WIDTH); rows not
        # in valid_idx are filled with NaN later.
        cov_per_group: dict[str, np.ndarray] = {}
        for group_name, track_ids, _sign in self._diagnostic_track_groups:
            cov_per_group[group_name] = np.full(
                (n, out_len), np.nan, dtype=np.float64
            )

        # Inference loop: for each batch, run forward_track_mean_perbase
        # once per track group. Encoding is shared.
        for bs in tqdm(
            range(0, len(contexts), self.batch_size),
            desc="Shorkie Hong (diagnostic)",
        ):
            batch = contexts[bs : bs + self.batch_size]
            x = _torch.from_numpy(
                np.stack([
                    one_hot_encode_channels_first(c.window_seq) for _, c in batch
                ])
            ).to(self.model.device)
            for group_name, track_ids, _sign in self._diagnostic_track_groups:
                track_idx_t = _torch.as_tensor(
                    track_ids, device=self.model.device, dtype=_torch.long,
                )
                with _torch.no_grad():
                    cov = self.model.forward_track_mean_perbase(x, track_idx_t)
                arr = cov.cpu().numpy()
                for j, (row_idx, _ctx) in enumerate(batch):
                    cov_per_group[group_name][row_idx] = arr[j]

        readouts = aggregate_diagnostic_base_readouts(
            cov_per_group, contexts, self._diagnostic_track_groups,
            n, CROP_BP_EACH_SIDE, out_len,
        )
        # Primary alias: shares the same per-locus bins as
        # predict_expressions by construction.
        readouts["primary"] = readouts[PRIMARY_READOUT_NAME]
        return readouts
