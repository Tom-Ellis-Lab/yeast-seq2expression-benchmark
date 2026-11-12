"""Synthetic sequence and adapter checks; no downloaded Gui construct assumed."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pysam
import pytest

from yeastbench.adapters._cassette_scaffold import reverse_complement
from yeastbench.adapters._gui_scaffold import (
    GuiContextError,
    GuiLocus,
    build_gui_context,
    build_gui_contexts,
    compute_gui_context_metadata,
    load_gui_constructs,
    validate_gui_locus,
)

PAYLOAD = "CCCCATGAAACCCTAAGGGGGGGG"
LOCUS = GuiLocus("test-plus", "I", 301, 310, "+", "reporter", 4, 16)
WINDOW = dict(seq_len=64, crop_bp_each_side=8, bin_width=4, output_bins=12)


@pytest.fixture
def reference_path(tmp_path: Path) -> Path:
    path = tmp_path / "reference.fa"
    path.write_text(">I\n" + "ACGT" * 15000 + "\n")
    pysam.faidx(str(path))
    return path


@pytest.fixture
def genome(reference_path: Path):
    with pysam.FastaFile(str(reference_path)) as fasta:
        yield fasta


@pytest.fixture
def constructs_path(tmp_path: Path) -> Path:
    path = tmp_path / "constructs.fasta"
    path.write_text(f">reporter synthetic test fixture\n{PAYLOAD}\n")
    return path


@pytest.mark.parametrize("strand", ["+", "-"])
def test_replacement_and_exact_cds_in_both_orientations(genome, strand):
    locus = replace(LOCUS, cassette_strand=strand)
    context = build_gui_context(locus, PAYLOAD, genome, **WINDOW)
    oriented = PAYLOAD if strand == "+" else reverse_complement(PAYLOAD)
    cds_offset = locus.gfp_cds_start if strand == "+" else len(PAYLOAD) - locus.gfp_cds_end
    reference = genome.fetch("I")
    edited = reference[:300] + oriented + reference[310:]
    cds_start = 300 + cds_offset
    window_start = cds_start + 6 - WINDOW["seq_len"] // 2
    assert context.window_seq == edited[window_start:window_start + WINDOW["seq_len"]]
    assert context.readout_start_in_window == 26
    assert len(context.readout_base_positions) == 12
    np.testing.assert_array_equal(context.readout_base_positions, np.arange(18, 30))
    extracted = context.window_seq[26:38]
    assert extracted == (PAYLOAD[4:16] if strand == "+" else reverse_complement(PAYLOAD[4:16]))
    # Both junctions are present and no removed genomic bases leak in.
    payload_start = context.payload_start_in_window
    assert context.window_seq[payload_start - 8:payload_start] == reference[292:300]
    payload_end = payload_start + len(PAYLOAD)
    assert context.window_seq[payload_end:payload_end + 8] == reference[310:318]
    metadata = compute_gui_context_metadata(locus, context, len(PAYLOAD))
    assert metadata["gfp_readout_bases"] == 12
    assert metadata["native_upstream_bp"] == 22
    assert metadata["native_downstream_bp"] == 18


def test_partial_cds_overlap_is_rejected(genome):
    with pytest.raises(GuiContextError, match="incomplete_gfp_readout"):
        build_gui_context(
            LOCUS, PAYLOAD, genome, seq_len=48, crop_bp_each_side=20, bin_width=1, output_bins=8,
        )


def test_boundary_shift_cannot_truncate_reporter(genome):
    with pytest.raises(GuiContextError, match="incomplete_gfp_readout"):
        build_gui_context(replace(LOCUS, deletion_start=1, deletion_end=10), PAYLOAD, genome, **WINDOW)


@pytest.mark.parametrize(
    "change, expected",
    [
        ({"deletion_start": 0}, "replacement interval"),
        ({"deletion_end": 300}, "replacement interval"),
        ({"cassette_strand": "?"}, "cassette_strand"),
        ({"gfp_cds_end": 500}, "outside"),
        ({"gfp_cds_end": 15}, "whole-codon"),
        ({"gfp_cds_start": 1}, "ATG"),
    ],
)
def test_invalid_geometry_fails_instead_of_being_repaired(change, expected):
    with pytest.raises(ValueError, match=expected):
        validate_gui_locus(replace(LOCUS, **change), PAYLOAD)


@pytest.mark.parametrize(
    "contents, expected",
    [
        (">same\nACGT\n>same\nACGT\n", "Duplicate"),
        (">reporter\nACNT\n", "unambiguous"),
        (">reporter\n", "nonempty"),
        ("", "No verified"),
        ("ACGT\n", "precedes"),
    ],
)
def test_unverified_or_malformed_payloads_are_rejected(tmp_path, contents, expected):
    path = tmp_path / "bad.fasta"
    path.write_text(contents)
    with pytest.raises(ValueError, match=expected):
        load_gui_constructs(path)


def test_internal_stop_cannot_be_declared_gfp_cds():
    with pytest.raises(ValueError, match="internal stop"):
        validate_gui_locus(LOCUS, "CCCCATGTAACCCTAAGGGGGGGG")


def test_window_exclusions_preserve_original_row_positions(genome):
    loci = [replace(LOCUS, locus_id="missing", chrom="II"), LOCUS]
    contexts, exclusions = build_gui_contexts(loci, {"reporter": PAYLOAD}, genome, **WINDOW)
    assert exclusions == {"missing": "chromosome_missing_from_reference"}
    assert contexts[0][0] == 1
    assert contexts[0][1] == LOCUS
    with pytest.raises(ValueError, match="missing construct"):
        build_gui_contexts([LOCUS], {}, genome, **WINDOW)


@pytest.mark.parametrize("from_checkpoints", [False, True])
def test_shorkie_uses_exact_cds_and_preserves_nan_alignment(
    reference_path, constructs_path, monkeypatch, from_checkpoints,
):
    torch = pytest.importorskip("torch")
    from yeastbench.adapters.shorkie_gui import ShorkieGuiPredictor

    class FakeShorkie:
        device = "cpu"

        def forward_track_mean_perbase(self, inputs, track_indices):
            assert tuple(inputs.shape[1:]) == (4, 16384)
            assert track_indices.tolist() == list(range(4201, 5215))
            # Large T0/other-track signals must not affect the Gui score.
            track_coverage = torch.full((5215,), 50000.0)
            track_coverage[4201:5215] = 2.0
            selected_mean = track_coverage.index_select(0, track_indices).mean()
            return selected_mean.expand(len(inputs), 14336)

    if from_checkpoints:
        from yeastbench.models.shorkie import Shorkie

        monkeypatch.setattr(Shorkie, "from_checkpoints", lambda *args, **kwargs: FakeShorkie())
        adapter = ShorkieGuiPredictor.from_checkpoints(
            "params.json", ["f0.h5"], reference_path, constructs_path, batch_size=1,
        )
    else:
        adapter = ShorkieGuiPredictor(FakeShorkie(), reference_path, constructs_path, batch_size=1)
    locus = replace(LOCUS, deletion_start=30001, deletion_end=30010)
    scores = adapter.predict_expressions([replace(locus, locus_id="missing", chrom="II"), locus])
    assert np.isnan(scores[0])
    assert scores[1] == 24.0  # 12 CDS bases, including partial 16 bp bins
    assert adapter.prediction_exclusions == {"missing": "chromosome_missing_from_reference"}
    assert set(adapter.prediction_metadata) == {locus.locus_id}
    assert adapter.prediction_metadata[locus.locus_id]["gfp_readout_bases"] == 12
    assert adapter.readout_metadata["group"] == "1000-RNA-Seq"
    assert adapter.readout_metadata["n_tracks"] == 1014
    assert adapter.readout_metadata["track_ids"] == list(range(4201, 5215))
    assert adapter.predict_expressions([locus]).tolist() == [24.0]
    assert adapter.prediction_exclusions == {}


@pytest.mark.parametrize("nanopore_coverage", [1000.0, 100000.0])
def test_yorzoi_uses_all_illumina_samples_on_reporter_strand(
    reference_path, constructs_path, nanopore_coverage,
):
    torch = pytest.importorskip("torch")
    from yeastbench.adapters.yorzoi_gui import YorzoiGuiPredictor

    class FakeYorzoi:
        device = "cpu"

        def forward_tracks_perbase(self, inputs):
            assert tuple(inputs.shape[1:]) == (4992, 4)
            coverage = torch.full((len(inputs), 162, 3000), nanopore_coverage)
            coverage[:, :10, :] = 2.0
            coverage[:, 10, :] = 13.0  # The non-human Illumina sample must count.
            coverage[:, 81:91, :] = 7.0
            coverage[:, 91, :] = 18.0
            return coverage

    adapter = YorzoiGuiPredictor(FakeYorzoi(), reference_path, constructs_path, batch_size=2)
    plus = replace(LOCUS, deletion_start=30001, deletion_end=30010)
    minus = replace(plus, locus_id="test-minus", cassette_strand="-")
    scores = adapter.predict_expressions([minus, plus])
    np.testing.assert_array_equal(scores, [96.0, 36.0])  # 12 bp × means of 8 and 3.
    assert adapter.prediction_exclusions == {}
    assert adapter.readout_metadata["group"] == "Illumina"
    assert adapter.readout_metadata["n_tracks"] == 11
    assert adapter.readout_metadata["track_ids_by_strand"] == {
        "+": list(range(11)), "-": list(range(81, 92)),
    }


def test_shorkie_custom_readout_metadata(reference_path, constructs_path):
    pytest.importorskip("torch")
    from yeastbench.adapters.shorkie_gui import ShorkieGuiPredictor

    class FakeShorkie:
        device = "cpu"

    adapter = ShorkieGuiPredictor(FakeShorkie(), reference_path, constructs_path, track_subset=[1148, 1156])
    assert adapter.readout_metadata["group"] == "custom"
    assert adapter.readout_metadata["track_ids"] == [1148, 1156]
    assert adapter.readout_metadata["n_tracks"] == 2
