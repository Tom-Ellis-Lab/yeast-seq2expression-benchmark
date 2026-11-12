"""Shorkie eQTL readout using shared allele/window preparation."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from yeastbench.adapters._genome import (
    parse_gene_annotations,
)
from yeastbench.adapters._shorkie_constants import (
    BIN_WIDTH,
    CROP_BP_EACH_SIDE,
    OUTPUT_BINS,
    SEQ_LEN,
    SHORKIE_1011_RNA_SEQ_TRACK_IDS,
)
from yeastbench.adapters._variant_scorer import VariantScorerBase
from yeastbench.adapters.protocols import Variant, VariantEffectScorer
from yeastbench.models.shorkie import Shorkie


class ShorkieVariantScorer(VariantScorerBase, VariantEffectScorer):
    seq_len = SEQ_LEN
    crop_bp_each_side = CROP_BP_EACH_SIDE
    bin_width = BIN_WIDTH
    output_bins = OUTPUT_BINS

    def __init__(
        self,
        model: Shorkie,
        fasta_path: str | Path,
        gtf_path: str | Path,
        track_subset: list[int] = SHORKIE_1011_RNA_SEQ_TRACK_IDS,
        batch_size: int = 8,
    ) -> None:
        import pysam

        self.model = model
        self.fasta = pysam.FastaFile(str(fasta_path))
        self.genes = parse_gene_annotations(gtf_path)
        self.track_subset = list(track_subset)
        self.batch_size = batch_size

    @classmethod
    def from_checkpoints(
        cls,
        params_path: str | Path,
        checkpoint_paths: Sequence[str | Path],
        fasta_path: str | Path,
        gtf_path: str | Path,
        track_subset: list[int] = SHORKIE_1011_RNA_SEQ_TRACK_IDS,
        device: str = "cuda",
        batch_size: int = 8,
        use_rc: bool = True,
    ) -> "ShorkieVariantScorer":
        return cls(
            Shorkie.from_checkpoints(
                params_path,
                checkpoint_paths,
                device=device,
                use_rc=use_rc,
            ),
            fasta_path=fasta_path,
            gtf_path=gtf_path,
            track_subset=list(track_subset),
            batch_size=batch_size,
        )

    def score_variants(self, variants: Sequence[Variant]) -> np.ndarray:
        import torch as _torch

        jobs = self._prepare_jobs(variants)
        n = len(jobs)
        scores = np.empty(n, dtype=np.float64)
        track_idx_t = _torch.tensor(
            self.track_subset, device=self.model.device, dtype=_torch.long
        )

        for batch_start in range(0, n, self.batch_size):
            batch_end = min(batch_start + self.batch_size, n)
            batch_jobs = jobs[batch_start:batch_end]
            B = len(batch_jobs)

            ref_arrs, alt_arrs = [], []
            for job in batch_jobs:
                r, a = self._build_one_hot_pair(job)
                ref_arrs.append(r)
                alt_arrs.append(a)
            x = _torch.from_numpy(
                np.concatenate(
                    [np.stack(ref_arrs, axis=0), np.stack(alt_arrs, axis=0)], axis=0
                )
            ).to(self.model.device)

            with _torch.no_grad():
                # (2B, out_len) — ensemble + RC + track-mean, unbinned to
                # per-base raw counts (softplus head; no inverse transform).
                cov = self.model.forward_track_mean_perbase(x, track_idx_t)

            for i, job in enumerate(batch_jobs):
                base_idx = job.base_idx
                if base_idx.size == 0:
                    scores[batch_start + i] = 0.0
                    continue
                base_t = _torch.from_numpy(base_idx).to(self.model.device)
                ref_sum = cov[i].index_select(0, base_t).sum().item()
                alt_sum = cov[i + B].index_select(0, base_t).sum().item()
                scores[batch_start + i] = float(
                    np.log2(alt_sum + 1.0) - np.log2(ref_sum + 1.0)
                )

        return scores
