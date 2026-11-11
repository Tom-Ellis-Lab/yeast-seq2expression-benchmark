"""Shared infrastructure for *cassette-insertion* benchmarks.

Benchmarks like Wu (RFP integrated at deleted-ORF loci) and Hong
(TDH3p-mCherry-ADH1t integrated at intergenic regions) are the same
conceptual operation: splice a **constant** foreign cassette into a
chromosome at a per-locus site, build a model-input window centred
on the cassette per a configurable anchor strategy, and report the
output bins overlapping the cassette's readout sub-feature
(e.g. mCherry CDS). This module owns that machinery; per-benchmark
modules (`_wu_scaffold.py`, `_hong_scaffold.py`) supply the
cassette geometry, the locus dataclass, and pick the anchor.

The anchor strategies currently in use:

- ``readout_at_downstream_edge`` — place the readout's downstream end
  (e.g. mCherry stop codon) at the downstream edge of the readable
  output crop. Maximises native flank visible **upstream** of the
  reporter at the cost of downstream context. Used by Wu, where the
  cassette replaces a deleted ORF and upstream context (where the
  ORF's natural promoter used to live) is the position-effect-rich
  side.
- ``center_cassette`` — put the cassette's midpoint at the window
  midpoint. Gives balanced up/downstream native flank. Used by Hong,
  where the cassette inserts between two intact native genes and
  signal can plausibly come from either side.
- ``center_readout`` — put the reporter CDS midpoint at the window
  midpoint. Used by Gui so marker length does not push GFP into a
  model's unreadable output crop.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    import pysam

_COMPLEMENT = str.maketrans("ACGTacgtNn", "TGCAtgcaNn")


def reverse_complement(seq: str) -> str:
    return seq.translate(_COMPLEMENT)[::-1]


@dataclass(frozen=True)
class CassetteGeometry:
    """A constant cassette plus the readout window inside it (CDS bins)."""
    payload: str             # uppercase DNA
    readout_start: int       # 0-based offset of readout (e.g. CDS 5')
    readout_len: int         # length of the readout (incl. stop if CDS)

    @property
    def payload_len(self) -> int:
        return len(self.payload)


@dataclass(frozen=True)
class InsertionSite:
    """Generic per-locus insertion descriptor.

    Two flavours, parameterised by the ``deletion_*`` pair:

    - *ORF-replacement* (Wu): cassette replaces the native bases at
      1-based inclusive span ``[deletion_start, deletion_end]``.
    - *Point insertion* (Hong): ``deletion_end < deletion_start``
      represents an empty span — no native bases removed, cassette
      goes between ``deletion_end`` and ``deletion_start``.
    """
    chrom: str
    deletion_start: int      # 1-based inclusive (or `deletion_end + 1` for point insertion)
    deletion_end: int        # 1-based inclusive; may be < start for empty span
    cassette_strand: str     # "+" or "-": cassette orientation in the chromosome


@dataclass(frozen=True)
class InsertionContext:
    """One model-input window + the readout's output bins, plus the per-base
    positions covering the same readout span (for per-base, untransformed
    scoring)."""
    window_seq: str
    readout_bins: np.ndarray   # output-bin indices overlapping the cassette readout
    readout_base_positions: np.ndarray  # per-base positions over the same span
    up_avail: int              # native bp available upstream of the insertion
    window_start_in_spliced: int
    payload_rc: bool
    payload_start_in_window: int
    readout_start_in_window: int


def span_to_bins(
    win_lo: int,
    win_hi: int,
    crop_bp_each_side: int,
    bin_width: int,
    output_bins: int,
) -> tuple[int, int] | None:
    """Window span → ``[b_lo, b_hi)`` output-bin range, or ``None`` if
    it does not overlap the readable crop."""
    b_lo = max(0, (win_lo - crop_bp_each_side) // bin_width)
    b_hi = min(
        output_bins, (win_hi - crop_bp_each_side + bin_width - 1) // bin_width
    )
    return (b_lo, b_hi) if b_hi > b_lo else None


def _readout_bins(
    readout_start_in_window: int,
    readout_len: int,
    crop_bp_each_side: int,
    bin_width: int,
    output_bins: int,
) -> np.ndarray:
    span = span_to_bins(
        readout_start_in_window,
        readout_start_in_window + readout_len,
        crop_bp_each_side, bin_width, output_bins,
    )
    if span is None:
        return np.array([], dtype=np.int64)
    return np.arange(span[0], span[1], dtype=np.int64)


def _readout_base_positions(
    readout_start_in_window: int,
    readout_len: int,
    crop_bp_each_side: int,
    out_len: int,
) -> np.ndarray:
    """Per-base analogue of :func:`_readout_bins`: cropped-output base
    positions in ``[0, out_len)`` covered by the readout span. The cassette
    readout is one contiguous feature, so no loop is needed. Unlike the
    1-based gene-exon helper, ``readout_start_in_window`` is already 0-based
    — no ``-1`` offset. Summing exact readout bases avoids the bin-boundary
    rounding ``_readout_bins`` incurs when the CDS edge lands mid-bin."""
    lo = max(0, readout_start_in_window - crop_bp_each_side)
    hi = min(out_len, readout_start_in_window + readout_len - crop_bp_each_side)
    if hi <= lo:
        return np.array([], dtype=np.int64)
    return np.arange(lo, hi, dtype=np.int64)


def build_insertion_context(
    site: InsertionSite,
    cassette: CassetteGeometry,
    fasta: "pysam.FastaFile",
    seq_len: int,
    crop_bp_each_side: int,
    bin_width: int,
    output_bins: int,
    window_anchor: Literal["readout_at_downstream_edge", "center_cassette", "center_readout"],
) -> InsertionContext | None:
    """Splice the cassette into the chromosome at the per-locus site,
    place the SEQ_LEN model input window per the requested anchor
    strategy, and return the spliced window + readout output-bin
    indices. Returns ``None`` if the locus cannot form a full window
    (too close to a chromosome end) or the readout falls outside the
    output crop.
    """
    chrom_len = fasta.get_reference_length(site.chrom)
    Lp = cassette.payload_len

    if site.cassette_strand == "+":
        oriented = cassette.payload
        readout_in_oriented = cassette.readout_start
    elif site.cassette_strand == "-":
        oriented = reverse_complement(cassette.payload)
        readout_in_oriented = Lp - cassette.readout_start - cassette.readout_len
    else:
        raise ValueError(f"cassette_strand must be '+' or '-', got {site.cassette_strand!r}")

    # ``deletion_start`` and ``deletion_end`` are 1-based inclusive.
    # For an empty deletion (point insertion), deletion_end < deletion_start.
    up_anchor_0 = site.deletion_start - 1   # 0-based: native bases through this index exclusive
    down_anchor_0 = site.deletion_end       # 0-based: native bases from this index inclusive
    if up_anchor_0 < 0 or down_anchor_0 > chrom_len:
        return None

    up_avail = min(seq_len, up_anchor_0)
    down_avail = min(seq_len, chrom_len - down_anchor_0)
    native_up = fasta.fetch(site.chrom, up_anchor_0 - up_avail, up_anchor_0).upper()
    native_down = fasta.fetch(site.chrom, down_anchor_0, down_anchor_0 + down_avail).upper()

    spliced = native_up + oriented + native_down
    if len(spliced) < seq_len:
        return None

    cassette_start_in_spliced = up_avail
    cassette_end_in_spliced = up_avail + Lp
    readout_start_in_spliced = up_avail + readout_in_oriented
    readout_end_in_spliced = readout_start_in_spliced + cassette.readout_len

    if window_anchor == "readout_at_downstream_edge":
        # Put the readout's *transcriptional* 3' end (stop codon for a
        # CDS readout) at the downstream edge of the readable crop.
        # For + strand cassette: transcription runs genomic-L→R, so the
        # 3' end is ``readout_end_in_spliced``; anchor it at window
        # position ``seq_len − crop``.
        # For − strand cassette (payload RC'd): transcription runs
        # genomic-R→L, so the 3' end is ``readout_start_in_spliced``
        # (genomic-low end of the RC'd readout); anchor it at window
        # position ``crop``.
        if site.cassette_strand == "+":
            window_start = readout_end_in_spliced - (seq_len - crop_bp_each_side)
        else:
            window_start = readout_start_in_spliced - crop_bp_each_side
    elif window_anchor == "center_cassette":
        cassette_center = (cassette_start_in_spliced + cassette_end_in_spliced) // 2
        window_start = cassette_center - seq_len // 2
    elif window_anchor == "center_readout":
        readout_center = (readout_start_in_spliced + readout_end_in_spliced) // 2
        window_start = readout_center - seq_len // 2
    else:
        raise ValueError(f"unknown window_anchor: {window_anchor!r}")
    window_start = max(0, min(window_start, len(spliced) - seq_len))

    window_seq = spliced[window_start : window_start + seq_len]
    readout_start_in_window = readout_start_in_spliced - window_start
    readout_bins = _readout_bins(
        readout_start_in_window, cassette.readout_len,
        crop_bp_each_side, bin_width, output_bins,
    )
    if readout_bins.size == 0:
        return None
    # readout_bins stays the window-validity gate, so locus selection is
    # unchanged; the per-base positions are the readout actually used by the
    # per-base adapters.
    readout_base_positions = _readout_base_positions(
        readout_start_in_window, cassette.readout_len,
        crop_bp_each_side, output_bins * bin_width,
    )

    return InsertionContext(
        window_seq=window_seq,
        readout_bins=readout_bins,
        readout_base_positions=readout_base_positions,
        up_avail=up_avail,
        window_start_in_spliced=window_start,
        payload_rc=(site.cassette_strand == "-"),
        payload_start_in_window=cassette_start_in_spliced - window_start,
        readout_start_in_window=readout_start_in_window,
    )


def load_cassette_payload(
    fasta_path: str | Path,
    expected_len: int | None = None,
    expected_readout_start: int | None = None,
    expected_readout_len: int | None = None,
) -> str:
    """Read a single-record cassette FASTA → uppercase payload. Asserts
    the geometry so a regenerated/edited FASTA that no longer matches the
    spec fails loudly instead of silently shifting the readout."""
    lines = Path(fasta_path).read_text().splitlines()
    seq = "".join(ln.strip() for ln in lines if ln and not ln.startswith(">")).upper()
    if expected_len is not None and len(seq) != expected_len:
        raise ValueError(
            f"cassette payload is {len(seq)} bp, expected {expected_len} ({fasta_path})"
        )
    if expected_readout_start is not None and expected_readout_len is not None:
        readout = seq[expected_readout_start : expected_readout_start + expected_readout_len]
        if not readout.startswith("ATG"):
            raise ValueError(
                f"readout CDS does not start with ATG at expected offset "
                f"{expected_readout_start} — cassette FASTA layout drift ({fasta_path})"
            )
    return seq


__all__ = [
    "CassetteGeometry",
    "InsertionSite",
    "InsertionContext",
    "build_insertion_context",
    "load_cassette_payload",
    "reverse_complement",
    "span_to_bins",
]
