"""Yorzoi marginalized / native-position adapter for Rafi MPRA.

Inserts the 110 bp MPRA sequence at native yeast genome positions upstream
of 22 host genes and computes logSED between the native REF and the edited
ALT. Yorzoi's 162 output tracks split into strand pairs (0-80 = ``+``,
81-161 = ``-``); a host gene aggregates over its strand-matched 81 tracks.

The REF→ALT→logSED orchestration lives in ``MPRAMarginalizedBase`` /
``MarginalizedLogSED``; this class only wires up Yorzoi's input geometry
and the :class:`YorzoiCoverage` reduction strategy.
"""
from __future__ import annotations

from pathlib import Path

from yeastbench.adapters._marginalized_logsed import YorzoiCoverage
from yeastbench.adapters._marginalized_mpra import MPRAMarginalizedBase
from yeastbench.adapters._yorzoi_constants import (
    BIN_WIDTH,
    CROP_BP_EACH_SIDE,
    OUTPUT_BINS,
    SEQ_LEN,
)
from yeastbench.models.yorzoi import Yorzoi


class YorzoiMPRAMarginalizedPredictor(MPRAMarginalizedBase):
    def __init__(
        self,
        model: Yorzoi,
        fasta_path: str | Path,
        gtf_path: str | Path,
        batch_size: int = 64,
        n_sample: int | None = None,
        seed: int = 42,
    ) -> None:
        self._setup(
            model=model,
            cov=YorzoiCoverage(model),
            gtf_path=gtf_path,
            fasta_path=fasta_path,
            seq_len=SEQ_LEN,
            crop_bp_each_side=CROP_BP_EACH_SIDE,
            bin_width=BIN_WIDTH,
            output_bins=OUTPUT_BINS,
            batch_size=batch_size,
            n_sample=n_sample,
            seed=seed,
            desc="Yorzoi marginalized",
        )

    @classmethod
    def from_pretrained(
        cls,
        hf_repo: str,
        fasta_path: str | Path,
        gtf_path: str | Path,
        device: str = "cuda",
        batch_size: int = 64,
        use_rc: bool = True,
        autocast: bool = True,
        n_sample: int | None = None,
        seed: int = 42,
        hf_revision: str | None = None,
    ) -> "YorzoiMPRAMarginalizedPredictor":
        return cls(
            Yorzoi.from_pretrained(
                hf_repo, device=device, use_rc=use_rc, autocast=autocast,
                hf_revision=hf_revision,
            ),
            fasta_path=fasta_path,
            gtf_path=gtf_path,
            batch_size=batch_size,
            n_sample=n_sample,
            seed=seed,
        )
