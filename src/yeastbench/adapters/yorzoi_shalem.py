"""Yorzoi marginalized adapter for the Shalem MPRA terminator benchmark.

Same protocol as the Shorkie version but with Yorzoi's 4992 bp input,
300 output bins at 10 bp/bin, and strand-matched track aggregation
(+ strand host genes → tracks 0–80, − strand host genes → tracks 81–161).

The REF→ALT→logSED orchestration lives in ``ShalemMarginalizedBase`` /
``MarginalizedLogSED``; this class only wires up Yorzoi's input geometry
and the :class:`YorzoiCoverage` reduction strategy.
"""
from __future__ import annotations

from pathlib import Path

from yeastbench.adapters._marginalized_logsed import YorzoiCoverage
from yeastbench.adapters._shalem_scaffold import ShalemMarginalizedBase
from yeastbench.adapters._yorzoi_constants import (
    BIN_WIDTH,
    CROP_BP_EACH_SIDE,
    OUTPUT_BINS,
    SEQ_LEN,
)
from yeastbench.models.yorzoi import Yorzoi


class YorzoiShalemPredictor(ShalemMarginalizedBase):
    def __init__(
        self,
        model: Yorzoi,
        fasta_path: str | Path,
        gtf_path: str | Path,
        host_genes_json: str | Path | None = None,
        batch_size: int = 64,
        n_sample: int | None = None,
        seed: int = 42,
    ) -> None:
        self._setup(
            model=model,
            cov=YorzoiCoverage(model),
            fasta_path=fasta_path,
            gtf_path=gtf_path,
            host_genes_json=host_genes_json,
            seq_len=SEQ_LEN,
            crop_bp_each_side=CROP_BP_EACH_SIDE,
            bin_width=BIN_WIDTH,
            output_bins=OUTPUT_BINS,
            batch_size=batch_size,
            n_sample=n_sample,
            seed=seed,
            desc="Yorzoi Shalem",
        )

    @classmethod
    def from_pretrained(
        cls,
        hf_repo: str,
        fasta_path: str | Path,
        gtf_path: str | Path,
        host_genes_json: str | Path | None = None,
        device: str = "cuda",
        batch_size: int = 64,
        use_rc: bool = True,
        autocast: bool = True,
        n_sample: int | None = None,
        seed: int = 42,
        hf_revision: str | None = None,
    ) -> "YorzoiShalemPredictor":
        return cls(
            Yorzoi.from_pretrained(
                hf_repo, device=device, use_rc=use_rc, autocast=autocast,
                hf_revision=hf_revision,
            ),
            fasta_path=fasta_path,
            gtf_path=gtf_path,
            host_genes_json=host_genes_json,
            batch_size=batch_size,
            n_sample=n_sample,
            seed=seed,
        )
