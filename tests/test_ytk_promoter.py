from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from yeastbench.adapters._ytk_scaffold import YTKConstruct, build_context
from yeastbench.benchmarks.base import BenchmarkInfo
from yeastbench.benchmarks.ytk_promoter import (
    YTKPromoterBenchmark,
    _range_metrics,
)


def _write_panel(tmp_path: Path) -> tuple[Path, Path, Path]:
    promoters = ["pStrong", "pMiddle", "pWeak"]
    truth = [
        {
            "promoter": promoter,
            "mRuby2_min_fold": ruby * 0.9,
            "mRuby2_median_fold": ruby,
            "mRuby2_max_fold": ruby * 1.1,
            "Venus_min_fold": venus * 0.9,
            "Venus_median_fold": venus,
            "Venus_max_fold": venus * 1.1,
        }
        for promoter, ruby, venus in zip(
            promoters, [100.0, 10.0, 1.0], [10000.0, 100.0, 1.0]
        )
    ]
    labels_path = tmp_path / "figure3a.tsv"
    with labels_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(truth[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(truth)

    fasta_path = tmp_path / "constructs.fasta"
    metadata_path = tmp_path / "constructs.tsv"
    metadata = []
    with fasta_path.open("w") as fasta:
        for promoter in promoters:
            for reporter, cds_len in (("Venus", 9), ("mRuby2", 6)):
                construct_id = f"{promoter}__{reporter}"
                payload = "C" * 10 + "ATG" + "A" * (cds_len - 3) + "G" * 20
                fasta.write(f">{construct_id}\n{payload}\n")
                metadata.append(
                    {
                        "construct_id": construct_id,
                        "promoter": promoter,
                        "reporter": reporter,
                        "reporter_cds_start": 10,
                        "reporter_cds_end": 10 + cds_len,
                        "reporter_cds_length": cds_len,
                        "payload_length": len(payload),
                        "chrom": "V",
                        "deletion_start": 100,
                        "deletion_end": 110,
                        "strand": "+",
                    }
                )
    with metadata_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(metadata[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(metadata)
    return labels_path, metadata_path, fasta_path


def _benchmark(tmp_path: Path) -> YTKPromoterBenchmark:
    labels, metadata, constructs = _write_panel(tmp_path)
    return YTKPromoterBenchmark(
        labels_path=labels,
        constructs_path=metadata,
        constructs_fasta=constructs,
        fasta_path=tmp_path / "reference.fa",
        info=BenchmarkInfo("ytk_promoter", "v1", "test", ""),
    )


class _HalfRangeAdapter:
    def __init__(self, benchmark: YTKPromoterBenchmark) -> None:
        self.benchmark = benchmark

    def predict_reporter_expressions(self, constructs):
        scales = {"mRuby2": 10.0, "Venus": 1000.0}
        return np.asarray(
            [
                np.sqrt(label) * scales[construct.reporter]
                for construct, label in zip(constructs, self.benchmark.labels)
            ]
        )


def test_panel_metrics_and_reporter_scale_invariance(tmp_path):
    benchmark = _benchmark(tmp_path)
    results = benchmark.evaluate(_HalfRangeAdapter(benchmark))

    assert len(results.scores) == 6
    assert results.metrics["mRuby2"].dynamic_range_recovery == pytest.approx(0.5)
    assert results.metrics["Venus"].dynamic_range_recovery == pytest.approx(0.5)
    assert results.metrics["consensus"].dynamic_range_recovery == pytest.approx(0.5)
    assert results.metrics["consensus"].dynamic_range_fidelity == pytest.approx(0.5)
    assert results.metrics["consensus"].spearman_rho == pytest.approx(1.0)
    assert results.metrics["consensus"].observed_fold_range == pytest.approx(1000.0)


def test_range_fidelity_penalises_exaggeration_and_collapse():
    exaggerated = _range_metrics(np.array([1.0, 10000.0]), np.array([1.0, 100.0]))
    assert exaggerated.dynamic_range_recovery == pytest.approx(2.0)
    assert exaggerated.dynamic_range_fidelity == pytest.approx(0.5)

    collapsed = _range_metrics(np.ones(3), np.array([1.0, 10.0, 100.0]))
    assert collapsed.dynamic_range_recovery == 0.0
    assert collapsed.dynamic_range_fidelity == 0.0
    assert np.isnan(collapsed.pearson_log10)


def test_results_round_trip(tmp_path):
    benchmark = _benchmark(tmp_path)
    results = benchmark.evaluate(_HalfRangeAdapter(benchmark))
    out_dir = tmp_path / "results"
    benchmark.save_results(results, out_dir)
    loaded = benchmark.load_results(out_dir)

    np.testing.assert_allclose(loaded.scores, results.scores)
    assert loaded.construct_ids == results.construct_ids
    assert loaded.metrics["consensus"] == results.metrics["consensus"]
    summary = benchmark.summary_dict(loaded)
    assert summary["consensus_dynamic_range_recovery"] == pytest.approx(0.5)


def test_compare_plot_has_model_by_reporter_scatter_panels(tmp_path):
    pytest.importorskip("matplotlib")
    benchmark = _benchmark(tmp_path)
    results = benchmark.evaluate(_HalfRangeAdapter(benchmark))
    model_dirs = {}
    for model in ("shorkie", "yorzoi"):
        model_dir = tmp_path / f"{model}__ytk_promoter"
        benchmark.save_results(results, model_dir)
        model_dirs[model] = model_dir

    output = benchmark.compare_plot(model_dirs, tmp_path / "compare")

    assert output == tmp_path / "compare/plot.svg"
    assert output.exists()
    assert output.stat().st_size > 0


def test_context_returns_only_exact_cds_bases(tmp_path):
    pysam = pytest.importorskip("pysam")
    reference = tmp_path / "reference.fa"
    reference.write_text(">V\n" + "A" * 12000 + "\n")
    pysam.faidx(str(reference))
    fasta = pysam.FastaFile(str(reference))
    construct = YTKConstruct(
        construct_id="pX__Venus",
        promoter="pX",
        reporter="Venus",
        payload="C" * 300 + "ATGAAAA" + "G" * 693,
        reporter_cds_start=300,
        reporter_cds_len=7,
        chrom="V",
        deletion_start=6001,
        deletion_end=6100,
        strand="+",
    )
    context = build_context(
        construct,
        fasta,
        seq_len=2000,
        crop_bp_each_side=100,
        bin_width=4,
        output_bins=450,
    )
    assert context is not None
    assert context.readout_base_positions.size == 7
    assert (
        context.window_seq[
            context.readout_start_in_window : context.readout_start_in_window + 7
        ]
        == "ATGAAAA"
    )
    assert context.readout_base_positions.tolist() == list(
        range(
            context.readout_start_in_window - 100,
            context.readout_start_in_window - 100 + 7,
        )
    )


def test_processed_v1_geometry_if_data_are_present():
    root = Path(__file__).resolve().parents[1]
    metadata = root / "data/tasks/ytk_promoter/constructs.tsv"
    fasta = root / "data/tasks/ytk_promoter/constructs.fasta"
    if not metadata.exists() or not fasta.exists():
        pytest.skip("processed YTK distribution is not installed")
    benchmark = YTKPromoterBenchmark(
        labels_path=root / "data/tasks/ytk_promoter/figure3a.tsv",
        constructs_path=metadata,
        constructs_fasta=fasta,
        fasta_path=root / "data/tasks/R64-1-1.fa",
        info=BenchmarkInfo("ytk_promoter", "v1", "test", ""),
    )
    assert len(benchmark.constructs) == 38
    lengths = {(c.reporter, c.reporter_cds_len) for c in benchmark.constructs}
    assert lengths == {("Venus", 714), ("mRuby2", 711)}
    assert {(c.deletion_start, c.deletion_end) for c in benchmark.constructs} == {
        (115946, 117045)
    }
    metadata_json = json.loads((metadata.parent / "build_metadata.json").read_text())
    assert metadata_json["version"] == "v1"


def test_shorkie_adapter_sums_selected_raw_bases(monkeypatch):
    torch = pytest.importorskip("torch")
    from yeastbench.adapters import shorkie_ytk
    from yeastbench.adapters.shorkie_ytk import ShorkieYTKPredictor
    from yeastbench.adapters._shorkie_constants import SEQ_LEN

    positions = np.array([1, 3, 5], dtype=np.int64)
    context = SimpleNamespace(
        window_seq="A" * SEQ_LEN,
        readout_base_positions=positions,
    )
    monkeypatch.setattr(shorkie_ytk, "build_context", lambda *args: context)

    class Model:
        device = torch.device("cpu")

        def forward_track_mean_perbase(self, x, track_subset):
            return torch.arange(14336, dtype=torch.float32)[None, :]

    predictor = ShorkieYTKPredictor.__new__(ShorkieYTKPredictor)
    predictor.model = Model()
    predictor.fasta = object()
    predictor.batch_size = 1
    predictor._track_idx_t = torch.tensor([0])
    scores = predictor.predict_reporter_expressions([object()])
    assert scores[0] == pytest.approx(1 + 3 + 5)


def test_yorzoi_adapter_sums_selected_raw_bases(monkeypatch):
    torch = pytest.importorskip("torch")
    from yeastbench.adapters import yorzoi_ytk
    from yeastbench.adapters.yorzoi_ytk import YorzoiYTKPredictor
    from yeastbench.adapters._yorzoi_constants import SEQ_LEN

    positions = np.array([2, 4, 7], dtype=np.int64)
    context = SimpleNamespace(
        window_seq="A" * SEQ_LEN,
        readout_base_positions=positions,
    )
    monkeypatch.setattr(yorzoi_ytk, "build_context", lambda *args: context)

    class Model:
        device = torch.device("cpu")

        def forward_tracks_perbase(self, x):
            base = torch.arange(3000, dtype=torch.float32)
            return base[None, None, :].expand(1, 162, 3000)

    predictor = YorzoiYTKPredictor.__new__(YorzoiYTKPredictor)
    predictor.model = Model()
    predictor.fasta = object()
    predictor.batch_size = 1
    scores = predictor.predict_reporter_expressions([object()])
    assert scores[0] == pytest.approx(2 + 4 + 7)
