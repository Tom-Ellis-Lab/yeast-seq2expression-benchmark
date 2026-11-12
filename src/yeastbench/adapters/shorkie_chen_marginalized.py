"""Shorkie readout for the shared Chen synonymous-MPRA pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from yeastbench.adapters._chen_marginalized import (
    ChenHostContext,
    load_hosts,
)
from yeastbench.adapters._chen_predictor import ChenMarginalizedBase, _Bundle
from yeastbench.adapters._shorkie_constants import (
    BIN_WIDTH,
    CROP_BP_EACH_SIDE,
    OUTPUT_BINS,
    SEQ_LEN,
    SHORKIE_T0_RNA_SEQ_TRACK_IDS,
)
from yeastbench.models.shorkie import Shorkie

if TYPE_CHECKING:
    import torch

log = logging.getLogger(__name__)


class ShorkieChenPredictor(ChenMarginalizedBase):
    description = "Shorkie"
    seq_len = SEQ_LEN
    crop_bp_each_side = CROP_BP_EACH_SIDE
    bin_width = BIN_WIDTH
    output_bins = OUTPUT_BINS

    def __init__(
        self,
        model: Shorkie,
        fasta_path: str | Path,
        hosts_path: str | Path,
        data_dir: str | Path,
        track_subset: list[int] = SHORKIE_T0_RNA_SEQ_TRACK_IDS,
        batch_size: int = 4,
    ) -> None:
        """``batch_size`` is in *variants*; each variant fans out across all
        host contexts in one forward pass (B * n_hosts windows)."""
        import pysam
        import torch as _torch

        self.model = model
        self.track_subset = list(track_subset)
        self.batch_size = batch_size

        self.fasta = pysam.FastaFile(str(fasta_path))
        self.hosts = load_hosts(hosts_path)
        self.data_dir = Path(data_dir)
        self._track_idx_gpu = _torch.tensor(
            self.track_subset,
            device=self.model.device,
            dtype=_torch.long,
        )
        self._bundles: dict[str, _Bundle] = {}

    @classmethod
    def from_checkpoints(
        cls,
        params_path: str | Path,
        checkpoint_paths: Sequence[str | Path],
        fasta_path: str | Path,
        hosts_path: str | Path,
        data_dir: str | Path,
        track_subset: list[int] = SHORKIE_T0_RNA_SEQ_TRACK_IDS,
        device: str = "cuda",
        batch_size: int = 4,
        use_rc: bool = True,
    ) -> "ShorkieChenPredictor":
        return cls(
            Shorkie.from_checkpoints(
                params_path,
                checkpoint_paths,
                device=device,
                use_rc=use_rc,
            ),
            fasta_path=fasta_path,
            hosts_path=hosts_path,
            data_dir=data_dir,
            track_subset=list(track_subset),
            batch_size=batch_size,
        )

    def _predict_exon_sums(
        self,
        batch_oh: "torch.Tensor",
        contexts: list[ChenHostContext],
    ) -> "torch.Tensor":
        """``batch_oh`` shape (B, 4, SEQ_LEN). Returns (B,) — CDS-bin sum
        per row, with row i interpreted as host index i % n_hosts.

        This means the caller must arrange ``batch_oh`` so the *i*-th
        row maps to host ``i % n_hosts``."""
        import torch as _torch

        n = len(contexts)
        B = batch_oh.shape[0]
        if B % n != 0:
            raise ValueError(f"batch size {B} must be a multiple of n_hosts={n}")

        with _torch.no_grad():
            # (B, OUT_BINS*BIN_WIDTH) per-base raw counts (unbin-only;
            # softplus head is already raw counts).
            cov = self.model.forward_track_mean_perbase(batch_oh, self._track_idx_gpu)

        out = _torch.zeros(B, device=cov.device, dtype=cov.dtype)
        # Could be vectorised but n_hosts is small; per-host base slice is clearest.
        for h, ctx in enumerate(contexts):
            mask_rows = _torch.arange(h, B, n, device=cov.device)
            out[mask_rows] = cov[mask_rows, ctx.cds_base_lo : ctx.cds_base_hi].sum(
                dim=1
            )
        return out
