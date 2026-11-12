"""Source-label reconciliation and explicit sequence-evidence requirements."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pysam
import pytest


@pytest.fixture
def builder(monkeypatch):
    script_dir = Path(__file__).resolve().parents[1] / "scripts" / "gui"
    monkeypatch.syspath_prepend(str(script_dir))
    spec = importlib.util.spec_from_file_location("gui_distribution_builder", script_dir / "build_gui_distribution.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_source_labels():
    return pd.DataFrame({
        "Gene.Name": ["GFP", "TFC3"], "Reference": ["GFP", "chrI"],
        "Strand": ["+", "-"], "Start": [1, 100], "End": [9, 150],
        "Length": [9, 51], "WILD TYPE": [0, 10], "YAL001C": [20, 12],
    }, index=pd.Index(["GFP", "YAL001C"], name="Gene.ID"))


def test_labels_average_libraries_and_keep_sequences_unresolved(builder):
    libraries = pd.DataFrame({"strain": ["YAL001C"] * 2,
                              "accession": ["GSM2", "GSM1"], "gfp_tpm": [30., 10.]})
    table = builder.build_labels(build_source_labels(), libraries)
    assert table.iloc[0].gfp_tpm == 20
    assert table.iloc[0].n_libraries == 2
    assert table.iloc[0].library_accessions == "GSM1;GSM2"
    assert table.iloc[0].sequence_status == "unresolved"
    assert pd.isna(table.iloc[0].deletion_start)


def test_label_disagreement_stops_build(builder):
    libraries = pd.DataFrame({"strain": ["YAL001C"], "accession": ["GSM1"], "gfp_tpm": [21.]})
    with pytest.raises(ValueError, match="does not reconcile"):
        builder.build_labels(build_source_labels(), libraries)


@pytest.mark.parametrize("evidence,chrom,start,expected_error", [
    ("", "I", 100, "Missing sequence evidence"),
    ("synthetic test fixture", "II", 100, "chromosome disagrees"),
    ("synthetic test fixture", "I", 100.5, "Noninteger"),
])
def test_bad_resolution_records_rejected(builder, tmp_path, evidence, chrom, start, expected_error):
    libraries = pd.DataFrame({"strain": ["YAL001C"], "accession": ["GSM1"], "gfp_tpm": [20.]})
    table = builder.build_labels(build_source_labels(), libraries)
    reference = tmp_path / "reference.fa"
    reference.write_text(">I\n" + "A" * 1000 + "\n")
    pysam.faidx(str(reference))
    constructs = tmp_path / "constructs.fasta"
    constructs.write_text(">test_construct\nCCATGAAATAACC\n")
    sequence_map = tmp_path / "sequences.tsv"
    pd.DataFrame([dict(locus_id="YAL001C", chrom=chrom, deletion_start=start,
                       deletion_end=150, cassette_strand="-", construct_id="test_construct",
                       gfp_cds_start=2, gfp_cds_end=11, sequence_evidence=evidence)]).to_csv(
                           sequence_map, sep="\t", index=False)
    with pytest.raises(ValueError, match=expected_error):
        builder.apply_sequence_map(table, sequence_map, constructs, reference)


def test_registry_forwards_explicit_constructs_path(tmp_path, monkeypatch):
    from yeastbench.adapters.protocols import GuiReporterExpressionPredictor
    from yeastbench import registry

    class Task:
        adapter_protocol = GuiReporterExpressionPredictor
        fasta_path = tmp_path / "custom_reference.fa"
        constructs_path = tmp_path / "custom_constructs.fa"

    def build_adapter(**arguments):
        return arguments

    monkeypatch.setitem(registry.SHORKIE_ADAPTERS, GuiReporterExpressionPredictor,
                        (build_adapter, registry.GUI_FIELDS))
    arguments = registry.MODELS["shorkie"](Task(), device="cpu")
    assert arguments["constructs_path"] == Task.constructs_path
    assert arguments["fasta_path"] == Task.fasta_path


def test_resolved_map_preserves_geometry_and_omits_unused_payloads(builder, tmp_path):
    libraries = pd.DataFrame({"strain": ["YAL001C"], "accession": ["GSM1"], "gfp_tpm": [20.]})
    table = builder.build_labels(build_source_labels(), libraries)
    reference = tmp_path / "reference.fa"
    reference.write_text(">I\n" + "A" * 1000 + "\n")
    pysam.faidx(str(reference))
    constructs = tmp_path / "constructs.fasta"
    constructs.write_text(">test_construct\nCCATGAAATAACC\n>unused\nATGCCCTAA\n")
    sequence_map = tmp_path / "sequences.tsv"
    pd.DataFrame([dict(locus_id="YAL001C", chrom="I", deletion_start=100,
                       deletion_end=150, cassette_strand="-", construct_id="test_construct",
                       gfp_cds_start=2, gfp_cds_end=11,
                       sequence_evidence="synthetic test fixture")]).to_csv(sequence_map, sep="\t", index=False)
    resolved, fasta_text = builder.apply_sequence_map(table, sequence_map, constructs, reference)
    assert resolved.iloc[0].sequence_status == "verified"
    assert resolved.iloc[0].exclusion_reason == ""
    assert resolved.iloc[0].cassette_strand == "-"
    assert resolved.iloc[0].gfp_cds_end == 11
    assert resolved.iloc[0].gfp_tpm == 20
    assert fasta_text == ">test_construct\nCCATGAAATAACC\n"


def test_gui_runtime_manifest_uses_processed_data_mirrors():
    from yeastbench.data.artifacts import BackendKind, License
    from yeastbench.data.manifest import artifacts_for

    artifacts = {artifact.id: artifact for artifact in artifacts_for(tasks=["gui_gfp_rna"])}
    assert set(artifacts) == {"gui", "refs"}
    assert artifacts["gui"].include == ("gui_gfp_rna_v1.tsv", "constructs.fasta")
    artifact = artifacts["gui"]
    assert artifact.redistributable
    assert artifact.license == License.CC_BY_NC_4_0
    assert [(mirror.backend, mirror.base, mirror.prefix) for mirror in artifact.mirrors] == [
        (BackendKind.HF, "tom-ellis-lab/yeast-seq2expression-benchmark-data", "gui/"),
        (BackendKind.GCS, "gs://yeast-seq2expression-benchmark/", "gui_v1/"),
    ]
