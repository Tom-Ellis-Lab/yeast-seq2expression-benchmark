"""Check retained barcode orientation and source-primer reconstruction guards."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pysam
import pytest


@pytest.fixture
def sequence_builder(monkeypatch):
    script_dir = Path(__file__).resolve().parents[1] / "scripts" / "gui"
    monkeypatch.syspath_prepend(str(script_dir))
    spec = importlib.util.spec_from_file_location("gui_sequence_builder", script_dir / "build_sequence_map.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def imp4_design():
    # Literal SGTC n3001 primer record; oligos are written 5' to 3'.
    return {
        "up_arm": "TGATTCAAGTGTTCCCATAAATAATAAAACAGTTAAATCGAAATG",
        "down_arm": "CATCGGCCTTCTATTTTAACCTTTACAAATCTATAAAAAAATTCA",
        "uptag": "AGAGCACTACAACCCTTGGA",
        "dntag": "AGGATAATCTCACCGAAGCC",
    }


@pytest.fixture
def imp4_primers(imp4_design):
    return {
        "UPstream45_primer_sequence": imp4_design["up_arm"],
        "DNstream45_primer_sequence": imp4_design["down_arm"],
        "UPTAG_primer_sequence":
            "AACAGTTAAATCGAAATGGATGTCCACGAGGTCTCTAGAGCACTACAACCCTTGGACGTACGCTGCAGGTCGAC",
        "DNTAG_primer_sequence":
            "AATCTATAAAAAAATTCACGGTGTCGGTCTCGTAGAGGATAATCTCACCGAAGCCATCGATGAATTCGAGCTCG",
    }


def test_imp4_retained_sites_follow_forward_and_reverse_primer_orientation(
    sequence_builder, imp4_design, imp4_primers,
):
    upstream, downstream = sequence_builder.compute_retained_flanks(imp4_design, imp4_primers)
    assert upstream == "GATGTCCACGAGGTCTCTAGAGCACTACAACCCTTGGA"
    # RC(DNTAG), then RC(D1); U2 and RC(D2) are already present in the core.
    assert downstream == "GGCTTCGGTGAGATTATCCTCTACGAGACCGACACCG"
    assert len(upstream) == 38
    assert len(downstream) == 37


@pytest.mark.parametrize("change,reason", [
    ("older_outer_u1", "inconsistent_or_noncanonical_outer_upstream_primer"),
    ("missing_dntag", "missing_or_ambiguous_designed_barcode"),
    ("different_primer_tag", "barcode_primer_structure_disagrees_with_tag_record"),
    ("genomic_overlap_mismatch", "two_stage_pcr_genomic_overlap_not_exact"),
])
def test_uncertain_source_records_are_excluded_without_guessing(
    sequence_builder, imp4_design, imp4_primers, change, reason,
):
    if change == "older_outer_u1":
        imp4_primers["UPstream45_primer_sequence"] += "GATGTCCACGAGCTCTCT"
    elif change == "missing_dntag":
        imp4_design["dntag"] = ""
    elif change == "different_primer_tag":
        imp4_design["uptag"] = "A" * 20
    else:
        imp4_primers["DNTAG_primer_sequence"] = "C" + imp4_primers["DNTAG_primer_sequence"][1:]
    with pytest.raises(sequence_builder.SequenceEvidenceError, match=reason):
        sequence_builder.compute_retained_flanks(imp4_design, imp4_primers)


@pytest.mark.parametrize("strand,chromosome_sequence", [
    ("+", "AAA" + "AACGTTAGGC" + "GGGTTTAACCG" + "GACCGTTAGG" + "TTT"),
    ("-", "AAA" + "CCTAACGGTC" + "GGGTTTAACCG" + "GCCTAACGTT" + "TTT"),
])
def test_deletion_boundaries_rechecked_on_both_genomic_strands(
    sequence_builder, tmp_path, strand, chromosome_sequence,
):
    reference_path = tmp_path / "reference.fa"
    reference_path.write_text(f">I\n{chromosome_sequence}\n")
    pysam.faidx(str(reference_path))
    design = dict(deletion_start=14, deletion_end=24, strand=strand,
                  up_arm="AACGTTAGGC", down_arm="CCTAACGGTC")
    with pysam.FastaFile(str(reference_path)) as reference:
        sequence_builder.validate_genomic_arms(design, "I", reference)
        design["deletion_start"] += 1
        with pytest.raises(sequence_builder.SequenceEvidenceError, match="genomic_arms_disagree"):
            sequence_builder.validate_genomic_arms(design, "I", reference)
