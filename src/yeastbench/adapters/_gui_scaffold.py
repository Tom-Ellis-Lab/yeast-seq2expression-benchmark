"""Validated insertion geometry for the Gui GFP RNA benchmark.

Coordinates describe the actual replaced interval, not the annotated native
gene span. Each FASTA record is the complete sequence between the two genomic
junctions, in GFP transcriptional orientation. The scalar readout uses GFP
only; its single edited copy needs neither a WT baseline nor diploid mixing.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Mapping, Sequence

from yeastbench.adapters._cassette_scaffold import (
    CassetteGeometry,
    InsertionContext,
    InsertionSite,
    build_insertion_context,
)

if TYPE_CHECKING:
    import pysam


@dataclass(frozen=True)
class GuiLocus:
    """Sequence-only input: phenotype columns never enter the adapter."""

    locus_id: str  # e.g. "YNL075W"
    chrom: str  # e.g. "XIV", matching the configured reference FASTA
    deletion_start: int  # 1-based inclusive genomic coordinate
    deletion_end: int  # 1-based inclusive genomic coordinate
    cassette_strand: str  # "+" or "-" in reference chromosome coordinates
    construct_id: str  # record identifier in constructs.fasta
    gfp_cds_start: int  # 0-based offset within the forward FASTA payload
    gfp_cds_end: int  # exclusive offset, including the stop codon


class GuiContextError(ValueError):
    """A valid locus cannot be scored with this reference/window geometry."""


def load_gui_constructs(path: str | Path) -> dict[str, str]:
    """Read explicit complete payloads; ambiguous bases are not verified DNA."""
    constructs: dict[str, str] = {}
    construct_id: str | None = None
    fragments: list[str] = []
    for line_number, line in enumerate(Path(path).read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if construct_id is not None:
                constructs[construct_id] = "".join(fragments).upper()
            fields = line[1:].split()
            if not fields:
                raise ValueError(f"Empty construct identifier at line {line_number}")
            construct_id = fields[0]
            if construct_id in constructs:
                raise ValueError(f"Duplicate construct identifier: {construct_id}")
            fragments = []
        else:
            if construct_id is None:
                raise ValueError("Construct sequence precedes its FASTA header")
            fragments.append(line)
    if construct_id is not None:
        constructs[construct_id] = "".join(fragments).upper()
    if not constructs:
        raise ValueError("No verified construct sequences in FASTA")
    for identifier, payload in constructs.items():
        if not payload or set(payload) - set("ACGT"):
            raise ValueError(f"Construct {identifier} must contain nonempty, unambiguous A/C/G/T DNA")
    return constructs


def validate_gui_locus(locus: GuiLocus, payload: str) -> None:
    """Reject invalid metadata rather than silently correcting a construct."""
    if not locus.locus_id or not locus.chrom or not locus.construct_id:
        raise ValueError("Locus, chromosome and construct identifiers are required")
    if locus.cassette_strand not in ("+", "-"):
        raise ValueError(f"{locus.locus_id}: cassette_strand must be + or -")
    if not 1 <= locus.deletion_start <= locus.deletion_end:
        raise ValueError(f"{locus.locus_id}: invalid inclusive replacement interval")
    if not payload or set(payload) - set("ACGT"):
        raise ValueError(f"{locus.locus_id}: construct must contain only A/C/G/T")
    if not 0 <= locus.gfp_cds_start < locus.gfp_cds_end <= len(payload):
        raise ValueError(f"{locus.locus_id}: GFP CDS is outside the construct")
    cds = payload[locus.gfp_cds_start:locus.gfp_cds_end]
    if len(cds) < 6 or len(cds) % 3 or not cds.startswith("ATG"):
        raise ValueError(f"{locus.locus_id}: GFP CDS must start with ATG and have a whole-codon length")
    stop_codons = {"TAA", "TAG", "TGA"}
    if cds[-3:] not in stop_codons:
        raise ValueError(f"{locus.locus_id}: GFP CDS must include its terminal stop codon")
    if any(cds[offset:offset + 3] in stop_codons for offset in range(3, len(cds) - 3, 3)):
        raise ValueError(f"{locus.locus_id}: GFP CDS contains an internal stop codon")


def build_gui_context(
    locus: GuiLocus,
    payload: str,
    fasta: "pysam.FastaFile",
    seq_len: int,
    crop_bp_each_side: int,
    bin_width: int,
    output_bins: int,
) -> InsertionContext:
    """Center GFP and require its complete CDS inside the readable output."""
    validate_gui_locus(locus, payload)
    if seq_len <= 0 or crop_bp_each_side < 0 or bin_width <= 0 or output_bins <= 0:
        raise ValueError("Invalid model window dimensions")
    if seq_len - 2 * crop_bp_each_side != output_bins * bin_width:
        raise ValueError("Model output dimensions do not match the cropped window")
    if locus.chrom not in fasta.references:
        raise GuiContextError("chromosome_missing_from_reference")
    if locus.deletion_end > fasta.get_reference_length(locus.chrom):
        raise GuiContextError("replacement_outside_reference")
    context = build_insertion_context(
        InsertionSite(locus.chrom, locus.deletion_start, locus.deletion_end, locus.cassette_strand),
        CassetteGeometry(payload, locus.gfp_cds_start, locus.gfp_cds_end - locus.gfp_cds_start),
        fasta, seq_len, crop_bp_each_side, bin_width, output_bins,
        window_anchor="center_readout",
    )
    if context is None:
        raise GuiContextError("cannot_construct_readable_window")
    cds_length = locus.gfp_cds_end - locus.gfp_cds_start
    if len(context.readout_base_positions) != cds_length:
        raise GuiContextError("incomplete_gfp_readout")
    return context


def build_gui_contexts(
    loci: Sequence[GuiLocus],
    constructs: Mapping[str, str],
    fasta: "pysam.FastaFile",
    seq_len: int,
    crop_bp_each_side: int,
    bin_width: int,
    output_bins: int,
) -> tuple[list[tuple[int, GuiLocus, InsertionContext]], dict[str, str]]:
    """Preserve row indices and distinguish unavailable windows from bad data."""
    if len({locus.locus_id for locus in loci}) != len(loci):
        raise ValueError("Duplicate Gui locus identifiers")
    contexts: list[tuple[int, GuiLocus, InsertionContext]] = []
    exclusions: dict[str, str] = {}
    for row_index, locus in enumerate(loci):
        if locus.construct_id not in constructs:
            raise ValueError(f"{locus.locus_id}: missing construct {locus.construct_id}")
        try:
            context = build_gui_context(
                locus, constructs[locus.construct_id], fasta,
                seq_len, crop_bp_each_side, bin_width, output_bins,
            )
        except GuiContextError as error:
            exclusions[locus.locus_id] = str(error)
        else:
            contexts.append((row_index, locus, context))
    return contexts, exclusions


def compute_gui_context_metadata(
    locus: GuiLocus, context: InsertionContext, payload_length: int,
) -> dict[str, int | str]:
    """Describe the actual window, including boundary-induced anchor shifts."""
    window_length = len(context.window_seq)
    native_left = max(0, min(window_length, context.payload_start_in_window))
    native_right = max(0, min(window_length, window_length - context.payload_start_in_window - payload_length))
    upstream, downstream = (native_left, native_right) if locus.cassette_strand == "+" else (native_right, native_left)
    return {
        "window_sha256": hashlib.sha256(context.window_seq.encode()).hexdigest(),
        "window_length": window_length,
        "gfp_start_in_window": context.readout_start_in_window,
        "gfp_end_in_window": context.readout_start_in_window + locus.gfp_cds_end - locus.gfp_cds_start,
        "gfp_readout_bases": len(context.readout_base_positions),
        "native_upstream_bp": upstream,
        "native_downstream_bp": downstream,
    }
