"""Yorzoi variant-effect scorer for the Caudal eQTL benchmark.

Implements the logSED-agg scoring procedure documented in
``docs/benchmarks/caudal_eqtl.md``'s Yorzoi section:
- 4,992 bp input window (channels-last), 300 output bins × 10 bp/bin,
  covering the central 3,000 bp.
- ``+`` (forward) strand tracks only (indices 0..80 of the 162-track
  output), regardless of the target gene's strand. Locked by the spec.
- Aggregation (raw-count scale): inverse Borzoi transform + per-base
  unbin → strand-matched cross-track mean → exon-base sum → log2 FC.
- Optional RC averaging — handled in `yeastbench.models.yorzoi.Yorzoi`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from yeastbench.adapters._genome import (
    ARABIC_TO_ROMAN,
    gene_exon_base_positions,
    one_hot_encode_channels_first,
    parse_gene_annotations,
    place_window,
)
from yeastbench.adapters.protocols import Variant, VariantEffectScorer

from yeastbench.adapters._yorzoi_constants import (
    BIN_WIDTH,
    CROP_BP_EACH_SIDE,
    OUTPUT_BINS,
    SEQ_LEN,
    YORZOI_PLUS_TRACK_IDS,
)
from yeastbench.models.yorzoi import Yorzoi


@dataclass(frozen=True)
class _ScoringJob:
    chrom_roman: str
    window_start: int  # 0-based
    var_idx_in_window: int
    ref: str
    alt: str
    base_idx: np.ndarray  # exon base positions in the cropped output


class YorzoiVariantScorer(VariantEffectScorer):
    def __init__(
        self,
        model: Yorzoi,
        fasta_path: str | Path,
        gtf_path: str | Path,
        track_subset: list[int] = YORZOI_PLUS_TRACK_IDS,
        batch_size: int = 16,
    ) -> None:
        import pysam

        self.model = model
        self.fasta = pysam.FastaFile(str(fasta_path))
        self.genes = parse_gene_annotations(gtf_path)
        self.track_subset = list(track_subset)
        self.batch_size = batch_size

        assert self.track_subset == YORZOI_PLUS_TRACK_IDS, (
            "YorzoiVariantScorer currently only supports the default "
            "plus-strand track subset; RC swap for arbitrary subsets is "
            "not implemented (and wasn't in the pre-refactor version)."
        )

    @classmethod
    def from_pretrained(
        cls,
        hf_repo: str,
        fasta_path: str | Path,
        gtf_path: str | Path,
        track_subset: list[int] = YORZOI_PLUS_TRACK_IDS,
        device: str = "cuda",
        batch_size: int = 16,
        use_rc: bool = True,
        autocast: bool = True,
        hf_revision: str | None = None,
    ) -> "YorzoiVariantScorer":
        return cls(
            Yorzoi.from_pretrained(
                hf_repo, device=device, use_rc=use_rc, autocast=autocast,
                hf_revision=hf_revision,
            ),
            fasta_path=fasta_path,
            gtf_path=gtf_path,
            track_subset=list(track_subset),
            batch_size=batch_size,
        )

    def _prepare_jobs(self, variants: Sequence[Variant]) -> list[_ScoringJob]:
        jobs: list[_ScoringJob] = []
        for v in variants:
            gene = self.genes[v.gene_id]
            chrom_roman = ARABIC_TO_ROMAN[v.chrom]
            if chrom_roman != gene.chrom_roman:
                raise ValueError(
                    f"variant chrom {v.chrom} ({chrom_roman}) does not match "
                    f"gene {v.gene_id} chrom {gene.chrom_roman}"
                )
            chrom_len = self.fasta.get_reference_length(chrom_roman)
            start0 = place_window(
                v.pos, gene.gene_center, chrom_len, SEQ_LEN, CROP_BP_EACH_SIDE
            )
            var_idx = v.pos - 1 - start0
            if 0 <= var_idx < SEQ_LEN:
                seq = self.fasta.fetch(
                    chrom_roman, start0, start0 + SEQ_LEN
                ).upper()
                ref_in_fasta = seq[var_idx]
                if ref_in_fasta != v.ref.upper():
                    raise ValueError(
                        f"REF mismatch at {v.chrom}:{v.pos}: "
                        f"FASTA={ref_in_fasta}, claimed={v.ref}"
                    )
            jobs.append(
                _ScoringJob(
                    chrom_roman=chrom_roman,
                    window_start=start0,
                    var_idx_in_window=var_idx,
                    ref=v.ref.upper(),
                    alt=v.alt.upper(),
                    base_idx=gene_exon_base_positions(
                        gene, start0, CROP_BP_EACH_SIDE, OUTPUT_BINS * BIN_WIDTH
                    ),
                )
            )
        return jobs

    def _build_one_hot_pair(self, job: _ScoringJob) -> tuple[np.ndarray, np.ndarray]:
        """Returns (ref, alt) one-hot arrays in channels-LAST layout ``(L, 4)``,
        which is what Yorzoi expects as input."""
        seq = self.fasta.fetch(
            job.chrom_roman, job.window_start, job.window_start + SEQ_LEN
        ).upper()
        ref_oh = one_hot_encode_channels_first(seq).T  # (L, 4)
        if 0 <= job.var_idx_in_window < SEQ_LEN:
            i = job.var_idx_in_window
            alt_seq = seq[:i] + job.alt + seq[i + len(job.ref):]
            if len(alt_seq) != SEQ_LEN:
                alt_seq = (alt_seq + "N" * SEQ_LEN)[:SEQ_LEN]
            alt_oh = one_hot_encode_channels_first(alt_seq).T
        else:
            alt_oh = ref_oh  # variant outside window → no mutation
        return ref_oh, alt_oh

    def score_variants(self, variants: Sequence[Variant]) -> np.ndarray:
        import torch as _torch

        jobs = self._prepare_jobs(variants)
        n = len(jobs)
        scores = np.empty(n, dtype=np.float64)
        plus_idx_t = _torch.tensor(
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
                # Per-base raw counts per track — the inverse Borzoi
                # transform is applied per pass before RC-averaging inside
                # the wrapper, so the track-mean below is on raw counts.
                perbase = self.model.forward_tracks_perbase(x)  # (2B, 162, out_len)
            # Strand-matched track mean on raw per-base counts → (2B, out_len)
            cov = perbase.index_select(1, plus_idx_t).mean(dim=1)

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
