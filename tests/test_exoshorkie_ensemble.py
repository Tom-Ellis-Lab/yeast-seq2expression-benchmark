"""Coverage-space averaging, readout order and task-specific exclusions."""
# ruff: noqa: E402
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from test_exoshorkie import wrapper
from yeastbench.models.exoshorkie import ExoShorkie, ExoShorkieEnsemble
from yeastbench.models.exoshorkie.checkpoints import CHECKPOINTS


def test_decode_and_clamp_before_mean_and_log_effect(monkeypatch):
    members = [wrapper(variant=v, seq_len=12, crop=2, width=4) for v in ("HPRT1", "dChr")]
    # Two examples (reference, alternate), two bins; the second student has a
    # different normalization. This distinguishes averaging coverage from
    # averaging normalized values, unclipped values, or final log effects.
    raw = [torch.tensor([[-0.5, 1.], [2., 3.]]), torch.tensor([[8., 99.], [2., 3.]])]
    for member, values in zip(members, raw):
        z = (torch.log1p(values) - member.checkpoint.mu) / member.checkpoint.sigma
        monkeypatch.setattr(member, "forward_normalized_binned", lambda x, z=z: z)
    model = ExoShorkieEnsemble(members)
    coverage = model.forward_coverage_binned(torch.zeros(2, 4, 12))
    torch.testing.assert_close(coverage, torch.tensor([[4., 50.], [2., 3.]]))
    sums = model.sum_region(torch.stack((coverage, coverage), dim=1), torch.arange(8), "+")
    effect = torch.log2(sums[1] + 1) - torch.log2(sums[0] + 1)
    torch.testing.assert_close(effect, torch.log2(torch.tensor(6. / 55.)))
    wrong = torch.stack([torch.log2(v[1].clamp_min(0).sum()+1)
                         - torch.log2(v[0].clamp_min(0).sum()+1) for v in raw]).mean()
    assert not torch.isclose(effect, wrong)
    stats = model.coverage_statistics()
    assert stats["coverage_bins_processed"] == 8
    assert stats["negative_bins_clipped"] == 1
    assert set(stats["per_student_coverage_statistics"]) == {"HPRT1", "dChr"}


def test_ensemble_strands_partial_bins_and_unbinning():
    members = [wrapper(variant=v, seq_len=12, crop=2, width=4) for v in ("chr7", "Mmmyco")]
    model = ExoShorkieEnsemble(members)
    x = torch.zeros(2, 4, 12)
    x[:, 0] = 1
    x[1, 0, 1] = 0
    x[1, 1, 1] = 1
    expected = torch.stack([m.forward_strands_binned(x) for m in members]).mean(0)
    actual = model.forward_strands_binned(x)
    torch.testing.assert_close(actual, expected)
    regions, strands = [[0, 2, 5], [1, 4, 7]], ["+", "-"]
    sums = model.predict_region_sums(x, regions, strands)
    torch.testing.assert_close(sums, torch.stack([
        model.sum_region(expected[i], region, strand)
        for i, (region, strand) in enumerate(zip(regions, strands))
    ]))
    torch.testing.assert_close(model.forward_unstranded_perbase(x),
                               expected.sum(1).repeat_interleave(4, -1) / 4)


def test_invalid_ensembles():
    with pytest.raises(ValueError, match="at least one"):
        ExoShorkieEnsemble([])
    with pytest.raises(ValueError, match="unique"):
        ExoShorkieEnsemble([wrapper(), wrapper()])
    with pytest.raises(ValueError, match="seq_len"):
        ExoShorkieEnsemble([wrapper(), wrapper(variant="chr7", seq_len=4096)])


@pytest.mark.parametrize("task_name,excluded", [("meneu_foreign_dna", {"Mpneumo", "Mmmyco"}),
                                               ("brooks_scramble", set())])
def test_registry_excludes_both_bacterial_students_only_for_meneu(monkeypatch, task_name, excluded):
    from yeastbench.adapters.protocols import CoverageTrackPredictor, TiledCoverageTrackPredictor
    from yeastbench.registry import MODELS

    loaded = []

    def load(path, variant, **kwargs):
        loaded.append(variant)
        return wrapper(variant=variant)

    monkeypatch.setattr(ExoShorkie, "from_checkpoint", load)
    task = SimpleNamespace(info=SimpleNamespace(name=task_name), adapter_protocol=(
        TiledCoverageTrackPredictor if excluded else CoverageTrackPredictor))
    adapter = MODELS["exoshorkie_ensemble"](task, device="cpu", checkpoint_dir="weights")
    assert set(loaded) == set(CHECKPOINTS) - excluded
    metadata = adapter.benchmark_metadata
    assert set(metadata["excluded_students"]) == excluded
    assert set(metadata["ensemble_members"]) == set(loaded)
    assert sum(metadata["ensemble_weights"].values()) == pytest.approx(1)
    assert metadata["species_index"] == 114
    assert metadata["training_overlap_contigs"] == []


def test_ensemble_config_and_downloads():
    from yeastbench.config import load_config
    from yeastbench.data.manifest import artifacts_for, validate_against_registry
    from yeastbench.registry import TASKS

    cfg = load_config("configs/exoshorkie_ensemble.yaml")
    assert cfg.device == "cuda:0"
    assert len(cfg.runs) == 1
    assert cfg.runs[0].model == "exoshorkie_ensemble"
    assert set(cfg.runs[0].tasks) == set(TASKS)
    assert "gui_gfp_rna" in cfg.runs[0].tasks
    assert len(artifacts_for(models=["exoshorkie_ensemble"])) == 6
    assert validate_against_registry() == []
