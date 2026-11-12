"""Cuperus 5'-UTR benchmark — construct assembly + window placement.

Unlike Wu/Hong (a *constant* cassette spliced at a *varying* locus), Cuperus
varies the **50 bp UTR** inside a *fixed* reporter construct
(``CYC1`` promoter + UTR + ``HIS3`` ORF + ``CYC1`` terminator) and reads out the
``HIS3`` ORF. The construct is a plasmid, so it has no genomic locus; to fill the
model's receptive window with in-distribution sequence we **embed the whole
construct at a background genomic locus** (``HIS3``'s own) and let the real
genomic flanks surround it. The construct is scored in this single fixed context:
a divergence check over diverse backgrounds (incl. a random-flank control)
confirmed the flanks shift the readout magnitude near-uniformly across UTRs, so
the rank-based metrics are unchanged (clean-bucket Spearman Δ ≤ 0.007).

This reuses ``_cassette_scaffold.build_insertion_context``: the *payload* is the
assembled construct and the *readout* is its ``HIS3`` sub-region. The background
is just the InsertionSite whose native span the construct replaces. Window anchor
is ``center_cassette`` (``HIS3`` sits mid-construct; no position-effect-rich side).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from yeastbench.adapters._cassette_scaffold import (
    CassetteGeometry,
    InsertionContext,
    InsertionSite,
    build_insertion_context,
)

if TYPE_CHECKING:
    import pysam

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONSTRUCT_JSON = (
    REPO_ROOT / "data" / "tasks" / "cuperus_mpra_5utr" / "construct.json"
)


@dataclass(frozen=True)
class CuperusConstruct:
    """The fixed reporter flanks; the UTR is spliced between promoter and ORF.
    Built by ``scripts/cuperus/build_construct.py`` (verified vs the genome)."""
    cyc1_promoter: str
    his3_orf: str
    cyc1_terminator: str

    @classmethod
    def from_reference(cls, fasta: "pysam.FastaFile") -> "CuperusConstruct":
        """Reproduce build_construct.py directly from the supplied R64-1-1 FASTA.

        The fixed reporter JSON is not part of the published task artifact.
        These are the same 1-based inclusive coordinates and junction checks
        used by the existing build script; no external plasmid lookup is needed.
        """
        promoter = fasta.fetch("X", 525980, 526278).upper()
        his3 = fasta.fetch("XV", 721945, 722608).upper()
        terminator = fasta.fetch("X", 526664, 526914).upper()
        if (len(promoter) != 298 or not promoter.endswith("ACATTAGGACCTTTGCAGC")
                or len(his3) != 663 or not his3.startswith("ATGACAGAGCAGAAAGCCCT")
                or his3[-3:] not in {"TAA", "TAG", "TGA"} or len(terminator) != 250):
            raise ValueError("Reference does not reproduce the Cuperus reporter junctions")
        return cls(promoter, his3, terminator)

    @classmethod
    def from_json(cls, path: str | Path = DEFAULT_CONSTRUCT_JSON) -> "CuperusConstruct":
        d = json.loads(Path(path).read_text())
        return cls(
            cyc1_promoter=d["cyc1_promoter"].upper(),
            his3_orf=d["his3_orf"].upper(),
            cyc1_terminator=d["cyc1_terminator"].upper(),
        )

    def assemble(self, utr: str) -> CassetteGeometry:
        """``promoter + UTR + HIS3 + terminator``, with the HIS3 ORF as the
        readout. ``readout_start`` shifts with the (variable-length) UTR."""
        utr = utr.upper()
        payload = self.cyc1_promoter + utr + self.his3_orf + self.cyc1_terminator
        readout_start = len(self.cyc1_promoter) + len(utr)
        return CassetteGeometry(
            payload=payload,
            readout_start=readout_start,
            readout_len=len(self.his3_orf),
        )


@dataclass(frozen=True)
class CuperusBackground:
    """A genomic locus whose native span the construct replaces, supplying the
    flanking sequence that fills the model window. ``HIS3``'s own locus
    (``HIS3_BACKGROUND``) is the single fixed context the benchmark scores in."""
    name: str
    chrom: str            # roman
    replace_start: int    # 1-based inclusive
    replace_end: int      # 1-based inclusive

    def site(self) -> InsertionSite:
        # The construct is always built/read on the + strand.
        return InsertionSite(
            chrom=self.chrom,
            deletion_start=self.replace_start,
            deletion_end=self.replace_end,
            cassette_strand="+",
        )


# Default single background: HIS3's own R64-1-1 locus (YOR202W), so the
# construct's HIS3 sits in roughly its native neighbourhood.
HIS3_BACKGROUND = CuperusBackground("HIS3", "XV", 721946, 722608)


def build_context(
    construct: CuperusConstruct,
    utr: str,
    background: CuperusBackground,
    fasta: "pysam.FastaFile",
    *,
    seq_len: int,
    crop_bp_each_side: int,
    bin_width: int,
    output_bins: int,
) -> InsertionContext | None:
    """Assemble the construct for ``utr``, embed it at ``background``, and
    return the model-input window + the ``HIS3`` readout positions. ``None`` if
    the background is too close to a chromosome end to form a full window."""
    cassette = construct.assemble(utr)
    return build_insertion_context(
        background.site(), cassette, fasta,
        seq_len, crop_bp_each_side, bin_width, output_bins,
        window_anchor="center_cassette",
    )


__all__ = [
    "CuperusConstruct",
    "CuperusBackground",
    "HIS3_BACKGROUND",
    "DEFAULT_CONSTRUCT_JSON",
    "build_context",
]
