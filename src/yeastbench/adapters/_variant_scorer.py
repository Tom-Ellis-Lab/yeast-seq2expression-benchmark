"""Shared eQTL window placement, allele validation and sequence construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from yeastbench.adapters._genome import (
    ARABIC_TO_ROMAN,
    gene_exon_base_positions,
    one_hot_encode_channels_first,
    parse_gene_annotations,
    place_window,
)
from yeastbench.adapters.protocols import Variant


@dataclass(frozen=True)
class _ScoringJob:
    """Per-variant pre-computed metadata for batched scoring."""

    chrom_roman: str
    window_start: int  # 0-based
    var_idx_in_window: int  # 0-based
    strand: str
    ref: str
    alt: str
    base_idx: np.ndarray  # exon base positions in the cropped output window


class VariantScorerBase:
    def __init__(self, model, fasta_path, gtf_path, batch_size=8):
        import pysam

        self.model = model
        self.fasta = pysam.FastaFile(str(fasta_path))
        self.genes = parse_gene_annotations(gtf_path)
        self.batch_size = batch_size

    def _prepare_jobs(self, variants: Sequence[Variant]) -> list[_ScoringJob]:
        """Resolve windows, verify ref base when the variant is inside the
        window, and pre-compute the exon-bin index per variant.

        When the variant is too far from the gene's center to fit alongside
        it inside one ``self.seq_len`` input window, the window falls back to
        gene-centered (matches the canonical Shorkie scoring script) and
        the variant ends up outside the input. In that case we keep the
        ref-pass prediction but skip the alt mutation, mirroring the
        upstream behavior — the resulting variant effect score is ≈ 0,
        which is the correct behavior for a variant the architecture
        cannot in principle see.
        """
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
                v.pos, gene.gene_center, chrom_len, self.seq_len, self.crop_bp_each_side
            )
            var_idx = v.pos - 1 - start0
            if 0 <= var_idx < self.seq_len:
                seq = self.fasta.fetch(
                    chrom_roman, start0, start0 + self.seq_len
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
                    strand=gene.strand,
                    ref=v.ref.upper(),
                    alt=v.alt.upper(),
                    base_idx=gene_exon_base_positions(
                        gene,
                        start0,
                        self.crop_bp_each_side,
                        self.output_bins * self.bin_width,
                    ),
                )
            )
        return jobs

    def _build_one_hot_pair(self, job: _ScoringJob) -> tuple[np.ndarray, np.ndarray]:
        seq = self.fasta.fetch(
            job.chrom_roman, job.window_start, job.window_start + self.seq_len
        ).upper()
        ref_oh = one_hot_encode_channels_first(seq)
        if 0 <= job.var_idx_in_window < self.seq_len:
            i = job.var_idx_in_window
            alt_seq = seq[:i] + job.alt + seq[i + len(job.ref) :]
            if len(alt_seq) != self.seq_len:
                alt_seq = (alt_seq + "N" * self.seq_len)[: self.seq_len]
            alt_oh = one_hot_encode_channels_first(alt_seq)
        else:
            alt_oh = ref_oh  # variant outside window → no mutation, score ≈ 0
        return ref_oh, alt_oh
