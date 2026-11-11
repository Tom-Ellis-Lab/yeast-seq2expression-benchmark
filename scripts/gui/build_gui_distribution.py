"""Build the Gui GFP RNA task from deposited abundances and resolved sequences.

Without --sequence-map/--constructs, export the verified expression labels with
unresolved sequence status. Such a distribution deliberately cannot be scored.
The sequence map is a TSV of source-resolved replacement geometry; it must
include sequence_evidence for every accepted locus. No cassette is inferred
from a generic vector name or a native gene's annotation.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile
import zipfile

import numpy as np
import pandas as pd
import pysam

from audit_data import load_soft_samples, load_xlsx_table


GEOMETRY_COLUMNS = (
    "chrom", "deletion_start", "deletion_end", "cassette_strand",
    "construct_id", "gfp_cds_start", "gfp_cds_end",
)


def compute_sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def load_gfp_libraries(abundance_path: Path, samples: pd.DataFrame) -> pd.DataFrame:
    """Extract reporter measurements by GSM accession, independent of row order."""
    accession_set = set(samples.accession)
    observations = []
    with tarfile.open(abundance_path) as archive:
        for member in archive.getmembers():
            accession = Path(member.name).name.split("_")[0]
            if not member.isfile() or accession not in accession_set:
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                raise ValueError(f"Cannot read abundance file for {accession}")
            abundance = pd.read_csv(io.BytesIO(gzip.decompress(extracted.read())), sep="\t")
            reporter = abundance.loc[abundance["Gene ID"] == "GFP", "TPM"]
            if len(reporter) != 1:
                raise ValueError(f"Expected exactly one GFP row for {accession}")
            observations.append({"accession": accession, "gfp_tpm": float(reporter.iloc[0])})
    measurements = pd.DataFrame(observations)
    if measurements.empty or measurements.accession.duplicated().any():
        raise ValueError("Missing or duplicated GFP library measurements")
    libraries = samples[["accession", "strain"]].merge(
        measurements, on="accession", validate="one_to_one", how="left",
    )
    if libraries.gfp_tpm.isna().any() or not np.isfinite(libraries.gfp_tpm).all():
        raise ValueError("Incomplete or nonfinite GFP library measurements")
    if (libraries.gfp_tpm < 0).any():
        raise ValueError("Negative deposited GFP TPM")
    return libraries


def build_labels(labels: pd.DataFrame, libraries: pd.DataFrame) -> pd.DataFrame:
    """Reconcile each engineered genotype's arithmetic mean with Table S2."""
    observations = []
    for locus_id in labels.columns[7:]:
        group = libraries.loc[libraries.strain == locus_id]
        if group.empty:
            raise ValueError(f"No RNA libraries for {locus_id}")
        mean_tpm = float(group.gfp_tpm.mean())
        published_tpm = float(labels.loc["GFP", locus_id])
        if mean_tpm <= 0 or not np.isclose(mean_tpm, published_tpm, atol=1e-6, rtol=0):
            raise ValueError(f"GFP abundance does not reconcile for {locus_id}")
        observations.append({
            "locus_id": locus_id,
            "gene_name": labels.loc[locus_id, "Gene.Name"],
            "gfp_tpm": mean_tpm,
            "n_libraries": len(group),
            "library_accessions": ";".join(sorted(group.accession)),
            "chrom": labels.loc[locus_id, "Reference"].removeprefix("chr"),
            "deletion_start": pd.NA,
            "deletion_end": pd.NA,
            "cassette_strand": "",
            "construct_id": "",
            "gfp_cds_start": pd.NA,
            "gfp_cds_end": pd.NA,
            "sequence_status": "unresolved",
            "exclusion_reason": "Exact insertion payload and replacement design not resolved",
            "sequence_evidence": "",
        })
    return pd.DataFrame(observations)


def apply_sequence_map(
    distribution: pd.DataFrame, sequence_map: Path, constructs_path: Path, reference_path: Path,
) -> tuple[pd.DataFrame, str]:
    # scripts/ may use the runtime validator; the dependency never goes backwards.
    from yeastbench.adapters._gui_scaffold import (
        GuiLocus, load_gui_constructs, validate_gui_locus,
    )

    geometry = pd.read_csv(sequence_map, sep="\t", keep_default_na=False)
    required = {"locus_id", "sequence_evidence", *GEOMETRY_COLUMNS}
    if not required.issubset(geometry.columns):
        raise ValueError(f"Sequence map is missing {sorted(required - set(geometry.columns))}")
    if geometry.locus_id.duplicated().any():
        raise ValueError("Sequence map has duplicate locus identifiers")
    unknown = set(geometry.locus_id) - set(distribution.locus_id)
    if unknown:
        raise ValueError(f"Unknown sequence-map loci: {sorted(unknown)}")
    constructs = load_gui_constructs(constructs_path)
    resolved = distribution.set_index("locus_id")
    with pysam.FastaFile(str(reference_path)) as reference:
        for record in geometry.to_dict("records"):
            locus_id = record["locus_id"]
            status = record.get("sequence_status", "verified")
            if status not in {"verified", "unresolved", "excluded"}:
                raise ValueError(f"Unknown sequence status for {locus_id}: {status}")
            if status != "verified":
                reason = str(record.get("exclusion_reason", "")).strip()
                if not reason:
                    raise ValueError(f"Missing sequence exclusion reason for {locus_id}")
                resolved.loc[locus_id, "sequence_status"] = status
                resolved.loc[locus_id, "exclusion_reason"] = reason
                resolved.loc[locus_id, "sequence_evidence"] = record["sequence_evidence"]
                continue
            if not str(record["sequence_evidence"]).strip():
                raise ValueError(f"Missing sequence evidence for {locus_id}")
            if record["chrom"] != resolved.loc[locus_id, "chrom"]:
                raise ValueError(f"Sequence-map chromosome disagrees with Table S2: {locus_id}")
            for column in ("deletion_start", "deletion_end", "gfp_cds_start", "gfp_cds_end"):
                coordinate = float(record[column])
                if not np.isfinite(coordinate) or not coordinate.is_integer():
                    raise ValueError(f"Noninteger {column} for {locus_id}")
                record[column] = int(coordinate)
            locus = GuiLocus(locus_id=locus_id, **{name: record[name] for name in GEOMETRY_COLUMNS})
            # The model adapters also validate each CDS against its actual payload.
            if locus.construct_id not in constructs:
                raise ValueError(f"Missing construct {locus.construct_id} for {locus_id}")
            payload = constructs[locus.construct_id]
            validate_gui_locus(locus, payload)
            if (locus.cassette_strand not in {"+", "-"}
                    or not 1 <= locus.deletion_start <= locus.deletion_end
                    <= reference.get_reference_length(locus.chrom)):
                raise ValueError(f"Invalid replacement geometry for {locus_id}")
            for column in GEOMETRY_COLUMNS:
                resolved.loc[locus_id, column] = record[column]
            resolved.loc[locus_id, "sequence_evidence"] = record["sequence_evidence"]
            resolved.loc[locus_id, "sequence_status"] = "verified"
            resolved.loc[locus_id, "exclusion_reason"] = ""
    used_constructs = set(resolved.loc[resolved.sequence_status == "verified", "construct_id"])
    fasta_text = "".join(f">{name}\n{constructs[name]}\n" for name in sorted(used_constructs))
    return resolved.reset_index(), fasta_text


def build_distribution(args: argparse.Namespace) -> None:
    if (args.sequence_map is None) != (args.constructs is None):
        raise ValueError("Supply --sequence-map and --constructs together")
    with zipfile.ZipFile(args.archive) as archive:
        labels = load_xlsx_table(archive.read("TableS2.xlsx")).set_index("Gene.ID")
    samples_path = args.source_dir / "GSE142839_family.soft.gz"
    abundance_path = args.source_dir / "GSE142839_RAW.tar"
    samples = load_soft_samples(samples_path, labels)
    libraries = load_gfp_libraries(abundance_path, samples)
    distribution = build_labels(labels, libraries)
    if len(distribution) != 240 or distribution.locus_id.duplicated().any():
        raise ValueError("Expected 240 unique engineered genotypes")
    constructs_text = ""
    inputs = [args.archive, samples_path, abundance_path, args.reference]
    if args.sequence_map is not None:
        distribution, constructs_text = apply_sequence_map(
            distribution, args.sequence_map, args.constructs, args.reference,
        )
        inputs.extend([args.sequence_map, args.constructs])
    for column in ("deletion_start", "deletion_end", "gfp_cds_start", "gfp_cds_end"):
        distribution[column] = distribution[column].astype("Int64")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    distribution.to_csv(args.out_dir / "gui_gfp_rna_v1.tsv", sep="\t", index=False)
    (args.out_dir / "constructs.fasta").write_text(constructs_text)
    libraries.to_csv(args.out_dir / "library_gfp.tsv", sep="\t", index=False)
    manifest = {
        "n_genotypes": len(distribution),
        "n_sequence_verified": int((distribution.sequence_status == "verified").sum()),
        "label": "arithmetic mean of deposited per-library GFP TPM",
        "source_doi": "10.1093/molbev/msab104",
        "inputs": {str(path): compute_sha256(path) for path in inputs},
        "outputs": {name: compute_sha256(args.out_dir / name)
                    for name in ("gui_gfp_rna_v1.tsv", "constructs.fasta")},
    }
    (args.out_dir / "build_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("archive/msab104_supplementary_data.zip"))
    parser.add_argument("--source-dir", type=Path, default=Path("data/tasks/gui_audit/source"))
    parser.add_argument("--reference", type=Path, default=Path("data/tasks/R64-5-1.fa"))
    parser.add_argument("--sequence-map", type=Path)
    parser.add_argument("--constructs", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("data/tasks/gui"))
    build_distribution(parser.parse_args())
