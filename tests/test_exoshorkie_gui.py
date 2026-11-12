"""Gui reporter readouts use exact CDS bases and the cassette's RNA strand."""
# ruff: noqa: E402
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pysam
import pytest

torch = pytest.importorskip("torch")

from test_exoshorkie import wrapper
from yeastbench.adapters._gui_scaffold import GuiLocus
from yeastbench.adapters.exoshorkie import ExoShorkieGuiPredictor
from yeastbench.adapters.protocols import GuiReporterExpressionPredictor
from yeastbench.models.exoshorkie import ExoShorkie
from yeastbench.models.exoshorkie.checkpoints import CHECKPOINTS
from yeastbench.models.exoshorkie.readouts import CoverageReadouts
from yeastbench.registry import MODELS


LOCUS = GuiLocus("plus", "I", 301, 310, "+", "reporter", 4, 13)


class BinnedReporterModel(CoverageReadouts):
    """Bin-dependent coverage distinguishes strand alignment and partial bins."""

    device = "cpu"
    seq_len = 64
    crop_bp_each_side = 8
    bin_width = 4
    output_bins = 12

    def __init__(self):
        self.forward_batch_sizes: list[int] = []

    def forward_coverage_binned(self, inputs):
        assert inputs.shape[1:] == (4, self.seq_len)
        self.forward_batch_sizes.append(len(inputs))
        bin_totals = torch.arange(1, self.output_bins + 1) * self.bin_width
        # Native flanks contain A; their reverse complements contain T.
        strand_scale = 1 + 2 * inputs[:, 3, 0]
        return strand_scale[:, None] * bin_totals


@pytest.fixture
def reference_path(tmp_path: Path) -> Path:
    path = tmp_path / "reference.fa"
    path.write_text(">I\n" + "A" * 2000 + "\n")
    pysam.faidx(str(path))
    return path


@pytest.fixture
def constructs_path(tmp_path: Path) -> Path:
    path = tmp_path / "constructs.fasta"
    path.write_text(">reporter\nCCCCATGAAATAAGGGGGGGG\n")
    return path


@pytest.mark.parametrize("batch_size", [1, 3])
def test_exact_cds_strands_exclusions_and_metadata(reference_path, constructs_path, batch_size):
    model = BinnedReporterModel()
    adapter = ExoShorkieGuiPredictor(model, reference_path, constructs_path, batch_size)
    minus = replace(LOCUS, locus_id="minus", cassette_strand="-")
    missing = replace(LOCUS, locus_id="missing", chrom="II")

    # The nine CDS bases cover four bases each in bins 5 and 6, and one in
    # bin 7. Minus-strand bins are reversed and have threefold coverage.
    scores = adapter.predict_expressions([minus, missing, LOCUS])
    np.testing.assert_allclose(scores, [3 * (4 * 7 + 4 * 6 + 5), np.nan, 4 * 6 + 4 * 7 + 8])
    assert sum(model.forward_batch_sizes) == 2
    assert adapter.prediction_exclusions == {"missing": "chromosome_missing_from_reference"}
    assert set(adapter.prediction_metadata) == {"minus", "plus"}
    for metadata in adapter.prediction_metadata.values():
        assert metadata["gfp_readout_bases"] == 9
        assert metadata["gfp_start_in_window"] == 28
        assert metadata["gfp_end_in_window"] == 37
        assert metadata["window_length"] == 64
        assert len(metadata["window_sha256"]) == 64
    assert adapter.readout_metadata["strand"] == "reporter strand"
    assert adapter.readout_metadata["coverage_units"] == "CPM"
    assert adapter.readout_metadata["n_tracks"] == 1

    adapter.predict_expressions([LOCUS])
    assert adapter.prediction_exclusions == {}
    assert set(adapter.prediction_metadata) == {"plus"}


@pytest.mark.parametrize("invalid_score", [np.nan, np.inf, -1.0])
def test_invalid_model_scores_preserve_rows_and_accept_zero(
    reference_path, constructs_path, monkeypatch, invalid_score,
):
    model = BinnedReporterModel()
    monkeypatch.setattr(
        model, "predict_region_sums",
        lambda inputs, regions, strands: torch.tensor([invalid_score, 0.0]),
    )
    adapter = ExoShorkieGuiPredictor(model, reference_path, constructs_path)
    zero_locus = replace(LOCUS, locus_id="zero")
    np.testing.assert_allclose(adapter.predict_expressions([LOCUS, zero_locus]), [np.nan, 0.0])
    assert adapter.prediction_exclusions == {"plus": "invalid_model_score"}
    assert set(adapter.prediction_metadata) == {"plus", "zero"}


def test_registry_builds_gui_ensemble_with_all_six_students(
    reference_path, constructs_path, tmp_path, monkeypatch,
):
    loaded = []

    def load_model(path, variant, **kwargs):
        loaded.append((path, variant))
        return wrapper(variant=variant, seq_len=64, crop=8, width=4)

    monkeypatch.setattr(ExoShorkie, "from_checkpoint", load_model)
    task = SimpleNamespace(
        info=SimpleNamespace(name="gui_gfp_rna"),
        adapter_protocol=GuiReporterExpressionPredictor,
        fasta_path=reference_path,
        constructs_path=constructs_path,
    )
    checkpoint_dir = tmp_path / "weights"
    adapter = MODELS["exoshorkie_ensemble"](
        task, device="cpu", checkpoint_dir=checkpoint_dir, batch_size=2,
    )
    assert isinstance(adapter, GuiReporterExpressionPredictor)
    assert loaded == [(checkpoint_dir / info.filename, variant) for variant, info in CHECKPOINTS.items()]
    metadata = adapter.benchmark_metadata
    assert metadata["ensemble_members"] == list(CHECKPOINTS)
    assert metadata["excluded_students"] == []
    assert metadata["ensemble_weights"] == {variant: 1 / 6 for variant in CHECKPOINTS}
    assert metadata["training_overlap_contigs"] == []
    assert np.isfinite(adapter.predict_expressions([LOCUS])).all()
