"""Construct loading and genome splicing for the YTK promoter panel."""

from __future__ import annotations

import csv
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


@dataclass(frozen=True)
class YTKConstruct:
    construct_id: str
    promoter: str
    reporter: str
    payload: str
    reporter_cds_start: int
    reporter_cds_len: int
    chrom: str
    deletion_start: int
    deletion_end: int
    strand: str


def _read_fasta(path: str | Path) -> dict[str, str]:
    records: dict[str, list[str]] = {}
    name: str | None = None
    for line in Path(path).read_text().splitlines():
        if line.startswith(">"):
            name = line[1:].split()[0]
            if name in records:
                raise ValueError(f"duplicate FASTA record {name!r} in {path}")
            records[name] = []
        elif name is not None:
            records[name].append(line.strip())
    return {name: "".join(lines).upper() for name, lines in records.items()}


def load_constructs(
    metadata_path: str | Path,
    fasta_path: str | Path,
) -> list[YTKConstruct]:
    sequences = _read_fasta(fasta_path)
    constructs: list[YTKConstruct] = []
    with Path(metadata_path).open(newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            construct_id = row["construct_id"]
            if construct_id not in sequences:
                raise ValueError(f"missing FASTA record for {construct_id}")
            payload = sequences[construct_id]
            cds_start = int(row["reporter_cds_start"])
            cds_len = int(row["reporter_cds_length"])
            if int(row["reporter_cds_end"]) != cds_start + cds_len:
                raise ValueError(f"inconsistent reporter CDS span for {construct_id}")
            if int(row["payload_length"]) != len(payload):
                raise ValueError(f"payload length mismatch for {construct_id}")
            if payload[cds_start : cds_start + cds_len][:3] != "ATG":
                raise ValueError(
                    f"reporter CDS does not start with ATG for {construct_id}"
                )
            constructs.append(
                YTKConstruct(
                    construct_id=construct_id,
                    promoter=row["promoter"],
                    reporter=row["reporter"],
                    payload=payload,
                    reporter_cds_start=cds_start,
                    reporter_cds_len=cds_len,
                    chrom=row["chrom"],
                    deletion_start=int(row["deletion_start"]),
                    deletion_end=int(row["deletion_end"]),
                    strand=row["strand"],
                )
            )
    if set(sequences) != {c.construct_id for c in constructs}:
        raise ValueError("construct metadata and FASTA record sets differ")
    return constructs


def build_context(
    construct: YTKConstruct,
    fasta: "pysam.FastaFile",
    seq_len: int,
    crop_bp_each_side: int,
    bin_width: int,
    output_bins: int,
) -> InsertionContext | None:
    """Build one centred post-integration window with exact CDS positions."""
    return build_insertion_context(
        InsertionSite(
            chrom=construct.chrom,
            deletion_start=construct.deletion_start,
            deletion_end=construct.deletion_end,
            cassette_strand=construct.strand,
        ),
        CassetteGeometry(
            payload=construct.payload,
            readout_start=construct.reporter_cds_start,
            readout_len=construct.reporter_cds_len,
        ),
        fasta,
        seq_len,
        crop_bp_each_side,
        bin_width,
        output_bins,
        window_anchor="center_cassette",
    )


__all__ = ["YTKConstruct", "build_context", "load_constructs"]
