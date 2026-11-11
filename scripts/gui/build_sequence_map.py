"""Build a conservative, as-designed Gui replacement-sequence map.

The core PCR cassette already includes U2 and reverse-complemented D2 homology.
Only single-design loci with consistent canonical two-tag PCR records are
accepted here. Original oligos, rather than another benchmark's scaffold,
determine the retained strand orientation. No barcode is fabricated.

"Verified" means source-resolved reconstruction, not sequencing of each strain.
Older records whose outer U1 primer disagrees with the inner barcode primer
are excluded, as are ambiguous designs and unresolved genomic homology arms.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import textwrap
import zipfile

import pandas as pd
import pysam

from audit_data import load_xlsx_table
from yeastbench.adapters._gui_scaffold import GuiLocus, load_gui_constructs, validate_gui_locus


U1 = "GATGTCCACGAGGTCTCT"
U2 = "CGTACGCTGCAGGTCGAC"
D1 = "CGGTGTCGGTCTCGTAG"
D2 = "ATCGATGAATTCGAGCTCG"
CORE_CDS_START = 666
CORE_CDS_END = 1383


class SequenceEvidenceError(ValueError):
    """Source records do not identify one conservative insertion sequence."""


def get_reverse_complement(sequence: str) -> str:
    return sequence.translate(str.maketrans("ACGT", "TGCA"))[::-1]


def compute_retained_flanks(design: dict, primers: dict) -> tuple[str, str]:
    """Return sequence before/after the core, in GFP transcriptional direction.

    D1 and D2 are reverse-primer sequences. Accordingly the top strand is
    U1-UPTAG-[U2...core...RC(D2)]-RC(DNTAG)-RC(D1). Including literal D1/D2
    sequences after the core would orient those retained sites incorrectly.
    """
    if primers["UPstream45_primer_sequence"] != design["up_arm"]:
        raise SequenceEvidenceError("inconsistent_or_noncanonical_outer_upstream_primer")
    if primers["DNstream45_primer_sequence"] != design["down_arm"]:
        raise SequenceEvidenceError("noncanonical_outer_downstream_primer")
    uptag, dntag = design["uptag"], design["dntag"]
    if len(uptag) != 20 or len(dntag) != 20 or set(uptag + dntag) - set("ACGT"):
        raise SequenceEvidenceError("missing_or_ambiguous_designed_barcode")
    upstream_suffix, downstream_suffix = U1 + uptag + U2, D1 + dntag + D2
    upstream_primer, downstream_primer = primers["UPTAG_primer_sequence"], primers["DNTAG_primer_sequence"]
    if not upstream_primer.endswith(upstream_suffix) or not downstream_primer.endswith(downstream_suffix):
        raise SequenceEvidenceError("barcode_primer_structure_disagrees_with_tag_record")
    upstream_overlap = upstream_primer[:-len(upstream_suffix)]
    downstream_overlap = downstream_primer[:-len(downstream_suffix)]
    if (len(upstream_overlap) < 18 or len(downstream_overlap) < 18
            or not design["up_arm"].endswith(upstream_overlap)
            or not design["down_arm"].endswith(downstream_overlap)):
        raise SequenceEvidenceError("two_stage_pcr_genomic_overlap_not_exact")
    return U1 + uptag, get_reverse_complement(D1 + dntag)


def validate_genomic_arms(design: dict, chromosome: str, reference: pysam.FastaFile) -> None:
    start, end = int(design["deletion_start"]), int(design["deletion_end"])
    upstream, downstream = design["up_arm"], design["down_arm"]
    if design["strand"] == "+":
        left = reference.fetch(chromosome, start - 1 - len(upstream), start - 1).upper()
        right = reference.fetch(chromosome, end, end + len(downstream)).upper()
        matches = left == upstream and right == get_reverse_complement(downstream)
    else:
        left = reference.fetch(chromosome, start - 1 - len(downstream), start - 1).upper()
        right = reference.fetch(chromosome, end, end + len(upstream)).upper()
        matches = left == downstream and right == get_reverse_complement(upstream)
    if not matches:
        raise SequenceEvidenceError("genomic_arms_disagree_with_reference")


def compute_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def build_sequence_map(args: argparse.Namespace) -> None:
    cores = load_gui_constructs(args.core)
    if len(cores) != 1:
        raise ValueError("Expected one reconstructed core PCR cassette")
    core = next(iter(cores.values()))
    if len(core) != 3144 or not core.startswith(U2) or not core.endswith(get_reverse_complement(D2)):
        raise ValueError("Core must contain the complete 3144 bp PCR product, including U2/D2 homology")
    with zipfile.ZipFile(args.archive) as archive:
        labels = load_xlsx_table(archive.read("TableS2.xlsx")).set_index("Gene.ID")
    candidates = pd.read_csv(args.designs, sep="\t", keep_default_na=False)
    catalog = pd.read_csv(args.primers, sep="\t", dtype=str, keep_default_na=False)
    catalog.columns = catalog.columns.str.strip()
    catalog = catalog.map(str.strip).set_index("rec_num")
    if not catalog.index.is_unique:
        raise ValueError("Primer catalog design IDs are not unique")
    observations, constructs = [], {}
    with pysam.FastaFile(str(args.reference)) as reference:
        for locus_id in labels.columns[7:]:
            chromosome = labels.loc[locus_id, "Reference"].removeprefix("chr")
            observation = dict(
                locus_id=locus_id, chrom=chromosome, deletion_start="", deletion_end="",
                cassette_strand="", construct_id="", gfp_cds_start="", gfp_cds_end="",
                sequence_status="unresolved", exclusion_reason="", sequence_evidence="",
                source_design_id="", barcode_evidence="as-designed SGTC; not resequenced",
            )
            records = candidates.loc[candidates.strain == locus_id]
            try:
                if len(records) != 1:
                    raise SequenceEvidenceError("multiple_or_missing_deletion_design_records")
                design = records.iloc[0].to_dict()
                observation["source_design_id"] = design["design_id"]
                if design["status"] != "unique_exact_arms":
                    raise SequenceEvidenceError("unresolved_genomic_homology_arms")
                for coordinate_name in ("deletion_start", "deletion_end"):
                    coordinate = float(design[coordinate_name])
                    if not coordinate.is_integer():
                        raise ValueError(f"Noninteger audited coordinate for {locus_id}")
                    design[coordinate_name] = int(coordinate)
                if design["strand"] != labels.loc[locus_id, "Strand"]:
                    raise SequenceEvidenceError("replacement_orientation_disagrees_with_annotation")
                primer_record = catalog.loc[design["design_id"]].to_dict()
                upstream, downstream = compute_retained_flanks(design, primer_record)
                validate_genomic_arms(design, chromosome, reference)
                payload = upstream + core + downstream
                construct_id = f"gui_{locus_id}_{design['design_id']}"
                locus = GuiLocus(
                    locus_id=locus_id, chrom=chromosome,
                    deletion_start=int(design["deletion_start"]), deletion_end=int(design["deletion_end"]),
                    cassette_strand=design["strand"], construct_id=construct_id,
                    gfp_cds_start=len(upstream) + CORE_CDS_START,
                    gfp_cds_end=len(upstream) + CORE_CDS_END,
                )
                validate_gui_locus(locus, payload)
                constructs[construct_id] = payload
                observation.update(
                    deletion_start=locus.deletion_start, deletion_end=locus.deletion_end,
                    cassette_strand=locus.cassette_strand, construct_id=construct_id,
                    gfp_cds_start=locus.gfp_cds_start, gfp_cds_end=locus.gfp_cds_end,
                    sequence_status="verified",
                    sequence_evidence=(
                        "Source-reconstructed as-designed, not strain-DNA verified: Chen2016 TableS1 "
                        "PCR core from AJ002682.1 and Callegari2011 pGSKU AppendixD; "
                        f"single SGTC design {design['design_id']}; canonical two-tag oligos with exact "
                        "two-stage-PCR overlap and R64 reference homology arms; designed barcodes retained "
                        "with reverse-primer regions reverse-complemented."
                    ),
                )
            except SequenceEvidenceError as error:
                observation["exclusion_reason"] = str(error)
            observations.append(observation)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    geometry = pd.DataFrame(observations)
    geometry.to_csv(args.out_dir / "resolved_loci.tsv", sep="\t", index=False)
    fasta_path = args.out_dir / "resolved_constructs.fasta"
    fasta_path.write_text("".join(
        f">{identifier}\n{textwrap.fill(payload, 80)}\n"
        for identifier, payload in sorted(constructs.items())
    ))
    summary = {
        "sequence_evidence": "source-reconstructed as-designed; not strain DNA sequencing",
        "n_genotypes": len(geometry), "n_source_resolved": len(constructs),
        "exclusions": geometry.loc[geometry.sequence_status != "verified", "exclusion_reason"].value_counts().to_dict(),
        "payload_lengths": sorted({len(payload) for payload in constructs.values()}),
        "core_cds_start": CORE_CDS_START, "core_cds_end": CORE_CDS_END,
        "canonical_payload_cds_start": len(U1) + 20 + CORE_CDS_START,
        "canonical_payload_cds_end": len(U1) + 20 + CORE_CDS_END,
        "retained_architecture": "U1-UPTAG-[core includes U2 and RC(D2)]-RC(DNTAG)-RC(D1)",
        "policy": "Single exact design; canonical two-tag PCR layout; no phenotype-based exclusion",
        "inputs": {str(path): compute_sha256(path) for path in (
            args.core, args.archive, args.designs, args.primers, args.reference,
        )},
        "outputs": {name: compute_sha256(args.out_dir / name) for name in (
            "resolved_loci.tsv", "resolved_constructs.fasta",
        )},
    }
    (args.out_dir / "sequence_map_provenance.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", type=Path, default=Path("results/gui_audit/cassette/gui_cassette_3144bp.fasta"))
    parser.add_argument("--archive", type=Path, default=Path("archive/msab104_supplementary_data.zip"))
    parser.add_argument("--designs", type=Path, default=Path("results/gui_audit/deletion_designs.tsv"))
    parser.add_argument("--primers", type=Path, default=Path("data/tasks/gui_audit/source/Deletion_primers_PCR_sizes.txt"))
    parser.add_argument("--reference", type=Path, default=Path("data/tasks/R64-5-1.fa"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/tasks/gui_audit"))
    build_sequence_map(parser.parse_args())
