#!/usr/bin/env python3
"""Build the Lee et al. YTK promoter-panel v1 runtime files.

The raw inputs are the digitised Figure 3A CSV and the 96 GenBank records
from the paper's official sequence archive. The final promoter-test plasmids
were not published. This script reconstructs the integrated cassette from the
standard YTK parts described in the paper and supporting information.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path


PROMOTERS = (
    "pTDH3",
    "pCCW12",
    "pPGK1",
    "pHHF2",
    "pTEF1",
    "pTEF2",
    "pHHF1",
    "pHTB2",
    "pRPL18B",
    "pALD6",
    "pPAB1",
    "pRET2",
    "pRNR1",
    "pSAC6",
    "pRNR2",
    "pPOP6",
    "pRAD27",
    "pPSP2",
    "pREV1",
)
REPORTERS = {
    "Venus": "pYTK033.gb",
    "mRuby2": "pYTK034.gb",
}
TRUTH_COLUMNS = (
    "promoter",
    "mRuby2_min_fold",
    "mRuby2_median_fold",
    "mRuby2_max_fold",
    "Venus_min_fold",
    "Venus_median_fold",
    "Venus_max_fold",
    "highlight_color",
    "n_biological_replicates",
    "source_url",
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_genbank_sequence(path: Path) -> str:
    """Read the ORIGIN sequence without requiring Biopython."""
    sequence: list[str] = []
    in_origin = False
    for line in path.read_text().splitlines():
        if line.startswith("ORIGIN"):
            in_origin = True
            continue
        if in_origin and line.startswith("//"):
            break
        if in_origin:
            sequence.append(re.sub(r"[^A-Za-z]", "", line))
    seq = "".join(sequence).upper()
    if not seq or set(seq) - set("ACGTN"):
        raise ValueError(f"could not parse a DNA sequence from {path}")
    return seq


def _ytk_part(path: Path, left: str, right: str) -> str:
    """Return one BsaI-released YTK part, including both 4bp overhangs."""
    seq = _read_genbank_sequence(path)
    forward_sites = [m.start() for m in re.finditer("GGTCTC", seq)]
    reverse_sites = [m.start() for m in re.finditer("GAGACC", seq)]
    if len(forward_sites) != 1 or len(reverse_sites) != 1:
        raise ValueError(f"{path}: expected one outward-facing BsaI site pair")
    # BsaI cuts GGTCTC(N1)/(N5). The reverse site therefore has one spacer
    # base immediately before it which is not part of the released insert.
    part = seq[forward_sites[0] + 7 : reverse_sites[0] - 1]
    if not part.startswith(left) or not part.endswith(right):
        raise ValueError(
            f"{path}: expected {left}...{right}, got {part[:4]}...{part[-4:]}"
        )
    return part


def _assemble(parts: list[str]) -> str:
    assembled = parts[0]
    for part in parts[1:]:
        if assembled[-4:] != part[:4]:
            raise ValueError(
                f"Golden Gate junction mismatch: {assembled[-4:]} != {part[:4]}"
            )
        assembled += part[4:]
    return assembled


def _read_fasta(path: Path) -> dict[str, str]:
    records: dict[str, list[str]] = {}
    name: str | None = None
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            name = line[1:].split()[0]
            records[name] = []
        elif name is not None:
            records[name].append(line.strip())
    return {name: "".join(lines).upper() for name, lines in records.items()}


def _ura3_replacement(parts: dict[str, str], reference_fasta: Path) -> tuple[int, int]:
    """Resolve the inner URA3 homology-arm boundaries on chromosome V."""
    chrom = _read_fasta(reference_fasta)["V"]
    five_arm = parts["ura3_5hom"][4:504]
    three_arm = parts["ura3_3hom"][24:524]

    # The 5' arm differs from R64-1-1 at two bases. Its last 400 bases are
    # exact and fix the alignment without a general-purpose aligner.
    suffix_start = chrom.find(five_arm[100:])
    if suffix_start < 100:
        raise ValueError("could not align the URA3 5' homology arm")
    five_start = suffix_start - 100
    mismatches = sum(
        a != b for a, b in zip(five_arm, chrom[five_start : five_start + 500])
    )
    if mismatches > 5:
        raise ValueError(f"URA3 5' homology arm has {mismatches} reference mismatches")

    three_start = chrom.find(three_arm)
    if three_start < 0:
        raise ValueError("could not align the URA3 3' homology arm")
    if five_start + 500 >= three_start:
        raise ValueError("URA3 homology arms overlap or are reversed")

    # Generic scaffold coordinates are 1-based inclusive. This removes the
    # native interval [five-arm end, three-arm start) and keeps both arms.
    return five_start + 501, three_start


def _write_fasta(path: Path, records: list[tuple[str, str]]) -> None:
    with path.open("w") as fh:
        for name, seq in records:
            fh.write(f">{name}\n")
            for start in range(0, len(seq), 80):
                fh.write(seq[start : start + 80] + "\n")


def build(input_dir: Path, reference_fasta: Path, output_dir: Path) -> None:
    plasmids = input_dir / "plasmids"
    truth_source = input_dir / "figure3a_main_plot_digitized.csv"
    output_dir.mkdir(parents=True, exist_ok=True)

    with truth_source.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    if tuple(rows[0]) != TRUTH_COLUMNS:
        raise ValueError(f"unexpected digitised CSV columns: {tuple(rows[0])}")
    if tuple(row["promoter"] for row in rows) != PROMOTERS:
        raise ValueError("digitised promoter order does not match pYTK009-pYTK027")

    with (output_dir / "figure3a.tsv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=TRUTH_COLUMNS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    parts = {
        "ura3_5hom": _ytk_part(plasmids / "pYTK092.gb", "CAAT", "CCCT"),
        "connector_5": _ytk_part(plasmids / "pYTK002.gb", "CCCT", "AACG"),
        "terminator": _ytk_part(plasmids / "pYTK056.gb", "ATCC", "GCTG"),
        "connector_3": _ytk_part(plasmids / "pYTK072.gb", "GCTG", "TACA"),
        "zeocin": _ytk_part(plasmids / "pYTK080.gb", "TACA", "GAGT"),
        "ura3_3hom": _ytk_part(plasmids / "pYTK086.gb", "GAGT", "CCGA"),
    }
    deletion_start, deletion_end = _ura3_replacement(parts, reference_fasta)

    reporters = {
        name: _ytk_part(plasmids / filename, "TATG", "ATCC")
        for name, filename in REPORTERS.items()
    }
    promoter_parts = {
        promoter: _ytk_part(plasmids / f"pYTK{part_no:03}.gb", "AACG", "TATG")
        for part_no, promoter in enumerate(PROMOTERS, start=9)
    }

    fasta_records: list[tuple[str, str]] = []
    metadata: list[dict[str, object]] = []
    for promoter_no, promoter in enumerate(PROMOTERS, start=9):
        for reporter, reporter_part in reporters.items():
            prefix = _assemble(
                [
                    parts["ura3_5hom"],
                    parts["connector_5"],
                    promoter_parts[promoter],
                ]
            )
            cds_start_full = len(prefix) - 3  # ATG is the last 3bp of TATG
            full = _assemble(
                [
                    parts["ura3_5hom"],
                    parts["connector_5"],
                    promoter_parts[promoter],
                    reporter_part,
                    parts["terminator"],
                    parts["connector_3"],
                    parts["zeocin"],
                    parts["ura3_3hom"],
                ]
            )

            # Keep only the sequence between the two homology arms. This is
            # the payload that replaces native URA3 in the reference genome.
            left_trim = 4 + 500
            right_trim = 500 + 4
            payload = full[left_trim:-right_trim]
            cds_start = cds_start_full - left_trim
            # Type-3 bodies omit the start codon and add a GG linker before
            # the ATCC overhang. The exact reporter CDS excludes that linker.
            reporter_body = reporter_part[4:-6]
            reporter_cds = "ATG" + reporter_body
            cds_end = cds_start + len(reporter_cds)
            if payload[cds_start:cds_end] != reporter_cds:
                raise ValueError(f"{promoter}/{reporter}: reporter CDS shifted")

            construct_id = f"{promoter}__{reporter}"
            fasta_records.append((construct_id, payload))
            metadata.append(
                {
                    "construct_id": construct_id,
                    "promoter": promoter,
                    "promoter_part": f"pYTK{promoter_no:03}",
                    "reporter": reporter,
                    "reporter_part": REPORTERS[reporter].removesuffix(".gb"),
                    "reporter_cds_start": cds_start,
                    "reporter_cds_end": cds_end,
                    "reporter_cds_length": len(reporter_cds),
                    "payload_length": len(payload),
                    "chrom": "V",
                    "deletion_start": deletion_start,
                    "deletion_end": deletion_end,
                    "strand": "+",
                    "terminator_part": "pYTK056",
                    "marker_part": "pYTK080",
                }
            )

    _write_fasta(output_dir / "constructs.fasta", fasta_records)
    with (output_dir / "constructs.tsv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(metadata[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(metadata)

    build_metadata = {
        "version": "v1",
        "paper_doi": "10.1021/sb500366v",
        "reporter_cds_definition": (
            "ATG junction plus annotated Type-3 reporter body; excludes the "
            "GGATCC linker and Type-4 stop codon"
        ),
        "construct_inference": (
            "Type-1 ConLS + Type-2 promoter + Type-3 reporter + Type-4 tTDH1 + "
            "Type-5 ConRE + Type-6 ZeocinR, integrated between URA3 homology arms"
        ),
        "raw_sha256": {
            "digitised_figure3a_csv": _sha256(truth_source),
            "reference_fasta": _sha256(reference_fasta),
            **{path.name: _sha256(path) for path in sorted(plasmids.glob("pYTK*.gb"))},
        },
    }
    (output_dir / "build_metadata.json").write_text(
        json.dumps(build_metadata, indent=2, sort_keys=True) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--reference-fasta", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    build(args.input_dir, args.reference_fasta, args.output_dir)


if __name__ == "__main__":
    main()
