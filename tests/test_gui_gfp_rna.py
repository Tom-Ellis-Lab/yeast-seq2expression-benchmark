"""Gui GFP label alignment, scoring, provenance and shared-cohort reporting."""
import json
import numpy as np
import pandas as pd
import pytest

from yeastbench.benchmarks.base import BenchmarkInfo
from yeastbench.benchmarks.gui_gfp_rna import GuiGFPRNABenchmark, compute_gui_metrics


@pytest.fixture
def gui_distribution(tmp_path):
    distribution = tmp_path / "gui"
    distribution.mkdir()
    rows = []
    for locus_id, tpm, strand in [
        ("YNL075W", 1000, "+"),
        ("YAL024C", 30, "-"),
        ("YAL016W", 20, "+"),
        ("YAL012W", 10, "+"),
    ]:
        rows.append(dict(
            locus_id=locus_id, chrom="I", deletion_start=100, deletion_end=130,
            cassette_strand=strand, construct_id="gfp", gfp_cds_start=4, gfp_cds_end=16,
            gfp_tpm=tpm, n_libraries=1, library_accessions="GSM123456",
            sequence_status="verified", exclusion_reason="",
        ))
    rows[1].update(sequence_status="unresolved", exclusion_reason="ambiguous_deletion")
    for column in ("deletion_start", "deletion_end", "gfp_cds_start", "gfp_cds_end"):
        rows[1][column] = ""
    pd.DataFrame(rows).to_csv(distribution / "gui_gfp_rna_v1.tsv", sep="\t", index=False)
    (distribution / "constructs.fasta").write_text(">gfp\nCCCCATGAAACCCTAAGGGG\n")
    reference = tmp_path / "reference.fa"
    reference.write_text(">I\n" + "A" * 1000 + "\n")
    return distribution, reference


def build_benchmark(gui_distribution, **kwargs):
    distribution, reference = gui_distribution
    return GuiGFPRNABenchmark(
        distribution, reference,
        BenchmarkInfo("gui_gfp_rna", "1", "GFP RNA", "local"),
        bootstrap_samples=50, **kwargs,
    )


class KnownScorePredictor:
    def __init__(self, scores=None, reasons=None):
        self.scores = scores if scores is not None else [100, 2, 1]
        self.prediction_exclusions = reasons or {}
        self.seen_loci = None

    def predict_expressions(self, loci):
        self.seen_loci = loci
        return np.asarray(self.scores)


def test_only_verified_geometry_reaches_adapter_and_rows_keep_identity(gui_distribution):
    benchmark = build_benchmark(gui_distribution)
    adapter = KnownScorePredictor([100, np.nan, 1], {"YAL016W": "cds_outside_crop"})
    adapter.prediction_metadata = {"YNL075W": {"window_length": 4992, "native_upstream_bp": 1234}}
    evaluated = benchmark.evaluate(adapter)
    assert [locus.locus_id for locus in adapter.seen_loci] == ["YNL075W", "YAL016W", "YAL012W"]
    assert not hasattr(adapter.seen_loci[0], "gfp_tpm")
    table = evaluated.predictions.set_index("locus_id")
    assert table.index.tolist() == ["YNL075W", "YAL024C", "YAL016W", "YAL012W"]
    assert table.loc["YAL024C", "exclusion_reason"] == "ambiguous_deletion"
    assert table.loc["YAL016W", "exclusion_reason"] == "cds_outside_crop"
    assert table.loc["YAL012W", "prediction"] == 1
    assert table.loc["YNL075W", "window_length"] == 4992
    assert table.loc["YNL075W", "native_upstream_bp"] == 1234
    assert pd.isna(table.loc["YAL024C", "window_length"])
    assert evaluated.summary["n_rows_total"] == 4
    assert evaluated.summary["n_sequence_verified"] == 3
    assert evaluated.summary["n_scored"] == 2
    assert evaluated.summary["n_excluded"] == 2


def test_imp4_is_retained_in_primary_with_separate_sensitivity(gui_distribution):
    benchmark = build_benchmark(gui_distribution)
    evaluated = benchmark.evaluate(KnownScorePredictor([.1, 2, 1]))
    assert evaluated.summary["n_scored"] == 3
    assert evaluated.summary["spearman_rho"] < 0
    assert evaluated.summary["without_imp4"]["n_scored"] == 2
    assert evaluated.summary["without_imp4"]["spearman_rho"] == pytest.approx(1)


@pytest.mark.parametrize("scores", [[[1], [2], [3]], [1, 2]])
def test_adapter_shape_is_checked(gui_distribution, scores):
    with pytest.raises(ValueError, match="Adapter returned shape"):
        build_benchmark(gui_distribution).evaluate(KnownScorePredictor(scores))


def test_negative_readouts_are_rejected(gui_distribution):
    with pytest.raises(ValueError, match="nonnegative linear RNA"):
        build_benchmark(gui_distribution).evaluate(KnownScorePredictor([-1, 2, 3]))


def test_unresolved_dataset_cannot_be_benchmarked(gui_distribution):
    distribution, _ = gui_distribution
    path = distribution / "gui_gfp_rna_v1.tsv"
    labels = pd.read_csv(path, sep="\t", keep_default_na=False)
    labels["sequence_status"] = "unresolved"
    labels.to_csv(path, sep="\t", index=False)
    with pytest.raises(ValueError, match="No sequence-verified"):
        build_benchmark(gui_distribution)


@pytest.mark.parametrize("column, replacement, message", [
    ("locus_id", ["duplicate"] * 4, "unique"),
    ("gfp_tpm", [0, 1, 2, 3], "finite and positive"),
    ("n_libraries", [1.5, 1, 1, 1], "positive integers"),
    ("deletion_start", [100.5, "", 100, 100], "integer coordinates"),
    ("sequence_status", ["verfied", "unresolved", "verified", "verified"], "Unknown sequence_status"),
])
def test_invalid_label_manifest_is_rejected(gui_distribution, column, replacement, message):
    distribution, _ = gui_distribution
    path = distribution / "gui_gfp_rna_v1.tsv"
    labels = pd.read_csv(path, sep="\t", keep_default_na=False)
    labels[column] = replacement
    labels.to_csv(path, sep="\t", index=False)
    with pytest.raises(ValueError, match=message):
        build_benchmark(gui_distribution)


def test_bootstrap_is_deterministic_and_resamples_ties():
    scores = np.array([1., 1., 2., 4., 5., 8.])
    labels = np.array([5., 3., 1., 2., 6., 9.])
    first = compute_gui_metrics(scores, labels, bootstrap_samples=100, bootstrap_seed=12)
    second = compute_gui_metrics(scores, labels, bootstrap_samples=100, bootstrap_seed=12)
    assert first == second
    assert first["bootstrap_valid_draws"]["spearman_rho"] > 0
    assert len(first["confidence_intervals_95"]["spearman_rho"]) == 2
    perfect = compute_gui_metrics(scores, scores * 3, bootstrap_samples=100)
    assert perfect["confidence_intervals_95"]["spearman_rho"] == pytest.approx([1, 1])


def test_undefined_metrics_are_json_null_and_log_uses_its_own_cohort():
    constant = compute_gui_metrics(np.ones(4), np.arange(1, 5), bootstrap_samples=50)
    for metric in ["pearson_r", "spearman_rho", "pearson_log2_r"]:
        assert constant[metric] is None
        assert constant["confidence_intervals_95"][metric] is None
        assert constant["bootstrap_valid_draws"][metric] == 0
    json.dumps(constant, allow_nan=False)
    partial = compute_gui_metrics(np.array([0, 2, 4, np.nan]), np.array([1, 2, 4, 8]))
    assert partial["n_scored"] == 3
    assert partial["n_log2"] == 2
    assert partial["pearson_log2_r"] == pytest.approx(1)


def test_roundtrip_preserves_scores_exclusions_and_provenance(gui_distribution, tmp_path):
    benchmark = build_benchmark(gui_distribution)
    adapter = KnownScorePredictor([100, np.nan, 1])
    adapter.readout_metadata = {
        "group": "example", "n_tracks": 2, "track_ids": [3, 7],
        "aggregation": "arithmetic mean of linear CDS sums",
    }
    evaluated = benchmark.evaluate(adapter)
    destination = tmp_path / "scores"
    benchmark.save_results(evaluated, destination)
    restored = benchmark.load_results(destination)
    assert restored.summary == evaluated.summary
    assert restored.summary["readout_metadata"] == adapter.readout_metadata
    assert restored.predictions.exclusion_reason.tolist() == evaluated.predictions.exclusion_reason.tolist()
    np.testing.assert_allclose(restored.predictions.prediction, evaluated.predictions.prediction, equal_nan=True)
    assert set(restored.summary["input_sha256"]) == {"labels", "constructs", "reference_fasta"}
    assert all(len(digest) == 64 for digest in restored.summary["input_sha256"].values())
    assert "NaN" not in (destination / "metrics.json").read_text()


@pytest.mark.parametrize("legacy", [False, True])
def test_reload_keeps_cli_run_identity_out_of_benchmark_metrics(
    gui_distribution, tmp_path, legacy,
):
    benchmark = build_benchmark(gui_distribution)
    evaluated = benchmark.evaluate(KnownScorePredictor())
    destination = tmp_path / "scores"
    benchmark.save_results(evaluated, destination)
    (destination / "summary.json").write_text(json.dumps({
        "model": "example", "task": "gui_gfp_rna", "task_version": "1",
        **benchmark.summary_dict(evaluated),
    }))
    if legacy:
        (destination / "metrics.json").unlink()
    restored = benchmark.load_results(destination)
    assert benchmark.summary_dict(restored) == benchmark.summary_dict(evaluated)


def test_model_comparison_uses_shared_loci(gui_distribution, tmp_path):
    benchmark = build_benchmark(gui_distribution)
    directories = {"model_a": tmp_path / "model_a", "model_b": tmp_path / "model_b"}
    benchmark.save_results(benchmark.evaluate(KnownScorePredictor([100, 2, 1])), directories["model_a"])
    benchmark.save_results(benchmark.evaluate(KnownScorePredictor([1, np.nan, 2])), directories["model_b"])
    comparison = tmp_path / "comparison"
    comparison_path = benchmark.compare_plot(directories, comparison)
    assert comparison_path == comparison / "plot.svg"
    assert comparison_path.is_file()
    summary = json.loads((comparison / "common_cohort_metrics.json").read_text())
    assert summary["locus_ids"] == ["YAL012W", "YNL075W"]
    assert summary["original_n_scored"] == {"model_a": 3, "model_b": 2}
    assert summary["metrics"]["model_a"]["spearman_rho"] == pytest.approx(1)
    assert summary["metrics"]["model_b"]["spearman_rho"] == pytest.approx(-1)
    assert set(summary["input_sha256"]) == {"labels", "constructs", "reference_fasta"}


@pytest.mark.parametrize("changed_input", ["labels", "constructs", "reference_fasta"])
def test_model_comparison_rejects_stale_input_sequences_even_with_same_labels(
    gui_distribution, tmp_path, changed_input,
):
    benchmark = build_benchmark(gui_distribution)
    directories = {"model_a": tmp_path / "model_a", "model_b": tmp_path / "model_b"}
    for destination in directories.values():
        benchmark.save_results(benchmark.evaluate(KnownScorePredictor()), destination)
    summary_path = directories["model_b"] / "metrics.json"
    summary = json.loads(summary_path.read_text())
    summary["input_sha256"][changed_input] = "0" * 64
    summary_path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="incompatible Gui input hashes"):
        benchmark.compare_plot(directories, tmp_path / "comparison")
    assert not (tmp_path / "comparison").exists()


def test_model_comparison_requires_input_provenance(gui_distribution, tmp_path):
    benchmark = build_benchmark(gui_distribution)
    directories = {"model_a": tmp_path / "model_a", "model_b": tmp_path / "model_b"}
    for destination in directories.values():
        benchmark.save_results(benchmark.evaluate(KnownScorePredictor()), destination)
    summary_path = directories["model_b"] / "metrics.json"
    summary = json.loads(summary_path.read_text())
    del summary["input_sha256"]
    summary_path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="missing Gui input provenance"):
        benchmark.compare_plot(directories, tmp_path / "comparison")


def test_plot_exports_separate_single_panel_images(gui_distribution, tmp_path):
    benchmark = build_benchmark(gui_distribution)
    destination = tmp_path / "plots"
    benchmark.plot(benchmark.evaluate(KnownScorePredictor()), destination)
    assert sorted(path.name for path in destination.iterdir()) == [
        "scatter_linear.png", "scatter_linear.svg", "scatter_log.png", "scatter_log.svg",
    ]
