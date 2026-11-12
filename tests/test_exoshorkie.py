"""Decoding order, strand semantics, partial-bin readouts and task integration."""

# ruff: noqa: E402
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from yeastbench.adapters.exoshorkie import (
    ExoShorkieBrooksPredictor,
    ExoShorkieChenPredictor,
    ExoShorkieCoverage,
    ExoShorkieMeneuPredictor,
    ExoShorkieVariantScorer,
    ExoShorkieYTKPredictor,
)
from yeastbench.models.exoshorkie import ExoShorkie
from yeastbench.models.exoshorkie.checkpoints import CHECKPOINTS, STUDENT_MODELS
from yeastbench.models.readouts import sum_binned_bases


class SignalModel(torch.nn.Module):
    """Orientation-dependent analytic coverage, emitted in training units."""

    def __init__(self, variant="dChr", seq_len=16384, crop=1024, width=16):
        super().__init__()
        self.backbone = SimpleNamespace(
            seq_len=seq_len,
            crop_bp_each_side=crop,
            bin_width=width,
            output_bins=(seq_len - 2 * crop) // width,
        )
        self.stats = CHECKPOINTS[variant]
        self.calls = []

    def forward(self, x):
        self.calls.append(tuple(x.shape))
        # C on forward DNA and G on its RC have different effects. A vs T
        # at the first position distinguishes the unmutated orientations.
        coverage = 1 + x[:, 1].sum(1) + 2 * x[:, 2].sum(1) + 2 * x[:, 3, 0]
        bins = (
            coverage[:, None].expand(-1, self.backbone.output_bins)
            * self.backbone.bin_width
        )
        return (torch.log1p(bins) - self.stats.mu) / self.stats.sigma


def wrapper(**kwargs):
    return ExoShorkie(
        SignalModel(**kwargs), kwargs.get("variant", "dChr"), device="cpu"
    )


def test_released_student_species_encoding_preserves_shorkie_default():
    from yeastbench.models.exoshorkie.nn import ExoShorkieModule
    from yeastbench.models.shared.shorkie.encoding import expand_species_encoding

    dna = torch.zeros(1, 4, 3)
    dna[0, 0, 0] = 1
    # Capture the actual expanded input reaching the student's backbone.
    stub = SimpleNamespace(backbone=lambda x: x, distill_head=lambda x: x)
    encoded = ExoShorkieModule.forward(stub, dna)
    assert encoded[:, 5 + 114].eq(1).all()
    assert encoded[:, 5 + 109].eq(0).all()
    torch.testing.assert_close(encoded[:, 4], torch.tensor([[0., 1., 1.]]))
    original = expand_species_encoding(dna)
    assert original[:, 5 + 109].eq(1).all()
    assert original[:, 5 + 114].eq(0).all()
    assert original[:, 4].eq(0).all()


@pytest.mark.parametrize("variant", CHECKPOINTS)
def test_inverse_is_natural_log_and_variant_specific(variant):
    model = wrapper(variant=variant)
    values = torch.tensor([-0.5, 0.0, 1.0, 1000.0])
    z = (torch.log1p(values) - model.checkpoint.mu) / model.checkpoint.sigma
    torch.testing.assert_close(model.to_coverage(z), values, atol=1e-4, rtol=1e-5)


def test_clamp_each_orientation_before_strand_sum(monkeypatch):
    model = wrapper(seq_len=12, crop=2, width=4)
    raw = [torch.tensor([[-0.5, 8.0]]), torch.tensor([[4.0, 12.0]])]
    seen = []

    def normalized(x):
        seen.append(x.clone())
        values = raw[len(seen) - 1]
        return (torch.log1p(values) - model.checkpoint.mu) / model.checkpoint.sigma

    monkeypatch.setattr(model, "forward_normalized_binned", normalized)
    x = torch.zeros(1, 5, 12)
    x[0, 0, 0] = 1
    x[0, 4, 2] = 1
    result = model.forward_unstranded_perbase(x)
    # minus bins flip to [12, 4]; clamped plus bins are [0, 8].
    torch.testing.assert_close(result, torch.full((1, 8), 3.0))
    assert seen[1][0, 3, -1] == 1  # A -> T at reversed position
    assert seen[1][0, 4, 9] == 1  # N reverses without channel swapping
    assert model.coverage_statistics() == {
        "coverage_bins_processed": 4,
        "negative_bins_clipped": 1,
        "negative_bin_fraction": 0.25,
    }


@pytest.mark.parametrize("strand,index", [("+", 0), ("-", 1)])
def test_partial_exons_match_unbin_then_sum(strand, index):
    model = wrapper(seq_len=12, crop=2, width=4)
    coverage = torch.tensor([[[8.0, 20.0], [12.0, 32.0]], [[4.0, 16.0], [40.0, 8.0]]])
    bases = torch.tensor([1, 2, 5, 6, 7])
    expected = (coverage[:, index].repeat_interleave(4, -1) / 4)[:, bases].sum(-1)
    torch.testing.assert_close(model.sum_region(coverage, bases, strand), expected)
    torch.testing.assert_close(model.sum_region(coverage, [], strand), torch.zeros(2))
    with pytest.raises(ValueError, match="outside"):
        model.sum_region(coverage, [8], strand)


def test_binning_inverse_cannot_follow_aggregation():
    model = wrapper(seq_len=12, crop=2, width=4)
    raw = torch.tensor([[1.0, 100.0]])
    z = (torch.log1p(raw) - model.checkpoint.mu) / model.checkpoint.sigma
    recovered = sum_binned_bases(model.to_coverage(z), torch.arange(8), 4)
    torch.testing.assert_close(recovered, torch.tensor([101.0]))
    wrong = model.to_coverage(z.repeat_interleave(4, -1) / 4).sum(-1)
    assert not torch.allclose(recovered, wrong)


def test_selected_strand_region_forward_matches_full_output(monkeypatch):
    model = wrapper(seq_len=12, crop=2, width=4)

    def normalized(x):
        bins = torch.tensor([4., 12.])[None, :] * (1 + x[:, 3, 0, None])
        return (torch.log1p(bins) - model.checkpoint.mu) / model.checkpoint.sigma

    monkeypatch.setattr(model, "forward_normalized_binned", normalized)
    x = torch.zeros(3, 4, 12)
    x[:, 0] = 1
    regions = [[0, 1, 4], [1, 5, 6, 7], [2]]
    strands = ["+", "-", "+"]
    full = model.forward_strands_binned(x)
    expected = torch.stack([model.sum_region(full[i], bases, strand)
                            for i, (bases, strand) in enumerate(zip(regions, strands))])
    torch.testing.assert_close(model.predict_region_sums(x, regions, strands), expected)
    assert model.coverage_statistics()["coverage_bins_processed"] == 18  # 12 + 6


def test_fast_encoder_matches_shared_encoder():
    from yeastbench.adapters.exoshorkie import _one_hot
    from yeastbench.adapters._genome import one_hot_encode_channels_first

    seq = "ACGTNRY-acgtn" * 11
    np.testing.assert_array_equal(_one_hot(seq), one_hot_encode_channels_first(seq))


def test_brooks_selects_strand_meneu_sums_strands():
    model = wrapper(seq_len=12, crop=2, width=4)
    seqs = ["A" * 12, "A" * 12]
    brooks = ExoShorkieBrooksPredictor(model)
    cov = brooks.predict_coverage_batch(seqs, ["+", "-"])
    np.testing.assert_allclose(cov[0], 1, rtol=1e-5)
    np.testing.assert_allclose(cov[1], 3, rtol=1e-5)
    summed = ExoShorkieMeneuPredictor(model).predict_coverage_batch(seqs, ["+", "-"])
    np.testing.assert_allclose(summed, 4, rtol=1e-5)


def test_chen_host_strands_and_forward_chunking():
    model = wrapper(seq_len=12, crop=2, width=4)
    adapter = object.__new__(ExoShorkieChenPredictor)
    adapter.model, adapter.batch_size = model, 2
    contexts = [
        SimpleNamespace(cds_base_lo=1, cds_base_hi=6, host=SimpleNamespace(strand=s))
        for s in ("+", "-")
    ]
    x = torch.zeros(6, 4, 12)
    x[:, 0] = 1
    got = adapter._predict_exon_sums(x, contexts)
    torch.testing.assert_close(got, torch.tensor([5.0, 15.0, 5.0, 15.0, 5.0, 15.0]))
    assert all(shape[0] <= 2 for shape in model.model.calls)


def test_eqtl_real_window_and_exon_coordinates(tmp_path):
    from yeastbench.adapters.protocols import Variant

    fasta = tmp_path / "genome.fa"
    fasta.write_text(">I\n" + "A" * 50000 + "\n")
    gtf = tmp_path / "genes.gtf"
    gtf.write_text(
        "".join(
            f'I\ttest\t{feature}\t20001\t20101\t.\t{strand}\t.\tgene_id "{gene}";\n'
            for gene, strand in (("plus", "+"), ("minus", "-"))
            for feature in ("gene", "exon")
        )
    )
    adapter = ExoShorkieVariantScorer(wrapper(), fasta, gtf, batch_size=2)
    variants = [Variant("1", 20050, "A", "C", gene) for gene in ("plus", "minus")]
    got = adapter.score_variants(variants)
    # 101 exact CDS bases. Forward coverage 1 -> 2; opposite strand 3 -> 5.
    np.testing.assert_allclose(got, [np.log2(203 / 102), np.log2(506 / 304)], rtol=1e-5)
    with pytest.raises(ValueError, match="REF mismatch"):
        adapter.score_variants([Variant("1", 20050, "T", "C", "plus")])


def test_reporter_streaming_passes_strand_and_exact_bases(monkeypatch):
    from yeastbench.adapters import _ytk_scaffold

    model = wrapper(seq_len=12, crop=2, width=4)
    adapter = object.__new__(ExoShorkieYTKPredictor)
    adapter.model, adapter.batch_size, adapter.fasta = model, 1, None
    constructs = [SimpleNamespace(strand=s) for s in ("+", "-", "+")]

    def build(construct, fasta, **geometry):
        assert geometry["seq_len"] == 12
        return SimpleNamespace(
            window_seq="A" * 12, readout_base_positions=np.array([1, 2, 5])
        )

    monkeypatch.setattr(_ytk_scaffold, "build_context", build)
    np.testing.assert_allclose(
        adapter.predict_reporter_expressions(constructs), [3, 9, 3], rtol=1e-5
    )


def test_marginalized_readout_row_and_batch_agree():
    model = wrapper(seq_len=12, crop=2, width=4)
    strategy = ExoShorkieCoverage(model)
    x = torch.from_numpy(np.stack([strategy.one_hot("A" * 12)] * 2))
    coverage = strategy.forward(x)
    bases = torch.tensor([1, 2, 5])
    for strand in ("+", "-"):
        batch = strategy.readout_batch(coverage, bases, strand)
        rows = torch.stack(
            [strategy.readout(coverage, i, bases, strand) for i in range(2)]
        )
        torch.testing.assert_close(batch, rows)


def test_config_has_all_students_tasks_and_unique_output_paths():
    from yeastbench.config import load_config
    from yeastbench.data.manifest import artifacts_for, validate_against_registry
    from yeastbench.registry import TASKS

    config = load_config("configs/exoshorkie_students.yaml")
    assert {r.model for r in config.runs} == set(STUDENT_MODELS)
    assert len({(r.model, t) for r in config.runs for t in r.tasks}) == 66
    for run in config.runs:
        assert set(run.tasks) == set(TASKS)
        artifacts = artifacts_for(models=[run.model])
        assert len(artifacts) == 1
        assert run.model_config["checkpoint_path"].endswith(artifacts[0].include[0])
    assert validate_against_registry() == []


@pytest.mark.parametrize("kind,length", [("mpra", 110), ("shalem", 150)])
def test_marginalized_scaffolds_batch_invariance(tmp_path, kind, length):
    from test_marginalized_logsed import build_synthetic_shalem_genome

    from yeastbench.adapters.exoshorkie import (
        ExoShorkieMPRAPredictor,
        ExoShorkieShalemPredictor,
    )

    fa, gt, host_json = build_synthetic_shalem_genome(tmp_path)
    cls = ExoShorkieMPRAPredictor if kind == "mpra" else ExoShorkieShalemPredictor
    extra = {} if kind == "mpra" else {"host_genes_json": host_json}
    adapter = cls(wrapper(), fa, gt, batch_size=2, **extra)
    assert adapter._contexts
    seqs = ["A" * length, "C" * length, "G" * length]
    predict = (
        adapter.predict_expression_scores
        if kind == "mpra"
        else adapter.predict_terminator_marginalized
    )
    batch = predict(seqs)
    adapter.batch_size = 1
    single = predict(seqs)
    np.testing.assert_allclose(batch, single, atol=1e-6)
    assert np.isfinite(batch).all()
    assert np.ptp(batch) > 0


@pytest.mark.parametrize("model_name", STUDENT_MODELS)
def test_registry_selects_variant_and_records_overlap(monkeypatch, model_name):
    from yeastbench.adapters.protocols import TiledCoverageTrackPredictor
    from yeastbench.registry import MODELS

    variant = STUDENT_MODELS[model_name]
    calls = []

    def load(path, selected, **kwargs):
        calls.append((path, selected))
        return wrapper(variant=selected)

    monkeypatch.setattr(ExoShorkie, "from_checkpoint", load)
    task = SimpleNamespace(
        adapter_protocol=TiledCoverageTrackPredictor,
        info=SimpleNamespace(name="meneu_foreign_dna"),
    )
    adapter = MODELS[model_name](task, device="cpu", checkpoint_path="weights.h5")
    assert calls == [("weights.h5", variant)]
    assert isinstance(adapter, TiledCoverageTrackPredictor)
    assert adapter.benchmark_metadata["variant"] == variant
    assert adapter.benchmark_metadata["training_overlap_contigs"] == (
        [variant] if variant in ("Mpneumo", "Mmmyco") else []
    )


@pytest.mark.integration
@pytest.mark.parametrize("ensemble", [False, True])
def test_real_data_task_scaffolds(monkeypatch, ensemble):
    """Exercise real task input construction with an analytic CPU model.

    Run explicitly after `ybench data get --config configs/exoshorkie_students.yaml`.
    These are integration checks, not measurements of student performance.
    """
    from pathlib import Path
    from yeastbench.config import load_config
    from yeastbench.data.fetch import check_run_data
    from yeastbench.registry import MODELS, TASKS

    config = load_config("configs/exoshorkie_students.yaml")
    if not check_run_data(list(TASKS), [], Path.cwd()).ready:
        pytest.skip("Complete real task data required")
    monkeypatch.setattr(ExoShorkie, "from_checkpoint", lambda path, variant, **kw: wrapper(variant=variant))
    for name in TASKS:
        task = TASKS[name](**config.tasks_config[name])
        model_name = "exoshorkie_ensemble" if ensemble else "exoshorkie_dchr"
        weights = {"checkpoint_dir": "."} if ensemble else {"checkpoint_path": "unused.h5"}
        adapter = MODELS[model_name](task, device="cpu", batch_size=8, **weights)
        assert isinstance(adapter, task.adapter_protocol), name
        if name == "wu_rfpins":
            loci = [locus for locus in task.loci if locus is not None][:2]
            scores = adapter.predict_expressions(loci)
        elif name == "hong_igr":
            loci = task.loci[:2]
            scores = adapter.predict_expressions(loci)
            readouts = adapter.predict_diagnostic_readouts(loci)
            np.testing.assert_allclose(scores, readouts["primary"])
            assert len(readouts) == 7
        elif name == "cuperus_utr":
            scores = adapter.predict_utr_expressions(["A" * 50, "C" * 50])
        elif name == "ytk_promoter":
            scores = adapter.predict_reporter_expressions(task.constructs[:2])
        elif name == "chen_synonymous":
            scores = []
            for lib in task.libraries:
                values = adapter.predict_local_variants([lib.library] * 2, lib.variant_seqs[:2])
                # GFP uses row 0, while TDH3 uses the native reference CDS.
                from yeastbench.adapters._chen_marginalized import build_cassette, VAR_LEN
                cassette = build_cassette(lib.library, adapter.fasta, adapter.data_dir)
                start = cassette.var_offset_in_cassette
                ref = cassette.sequence[start:start + VAR_LEN]
                ref_score = adapter.predict_local_variants([lib.library], [ref])
                assert abs(ref_score[0]) < 1e-5
                scores.extend(values)
        elif name == "rafi_mpra_marginalized":
            scores = adapter.predict_expression_scores(["A" * 110, "C" * 110])
        elif name == "shalem_mpra_marginalized":
            scores = adapter.predict_terminator_marginalized(["A" * 150, "C" * 150])
        elif name == "brooks_scramble":
            row = task._materialize(adapter.seq_len).iloc[0]
            scores = adapter.predict_coverage_batch([row.alt_seq], [row.strand])
        elif name == "meneu_foreign_dna":
            seq = task._load_contig_seq(task.contigs[0])[:adapter.seq_len]
            scores = adapter.predict_coverage_batch([seq], ["+"])
        else:
            # eQTL window/allele preparation has its own exact-coordinate test.
            scores = adapter.score_variants([])
        assert np.isfinite(scores).all(), name
