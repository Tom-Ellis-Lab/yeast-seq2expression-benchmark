"""Pinned model references must reach the actual model and tokenizer loaders."""
from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from yeastbench.data import model_sources
from yeastbench.data.lock import LockError


REVISION = "a" * 40
OTHER_REVISION = "b" * 40
REPO = "tom-ellis-lab/yorzoi"


def _lock(artifact_id="yorzoi", repo_id=REPO):
    return {
        "version": 2,
        "artifacts": {
            artifact_id: {
                "files": {},
                "sources": [{
                    "backend": "hf",
                    "repo_id": repo_id,
                    "repo_type": "model",
                    "prefix": "",
                    "revision": REVISION,
                }],
            },
        },
    }


@pytest.fixture
def snapshot(monkeypatch, tmp_path):
    download = Mock(return_value=str(tmp_path / "snapshot"))
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=download))
    return download


def test_locked_model_uses_exact_snapshot(snapshot):
    result = model_sources.resolve_model_source("yorzoi", REPO, lock=_lock())
    assert result == snapshot.return_value
    snapshot.assert_called_once_with(repo_id=REPO, repo_type="model", revision=REVISION)


def test_explicit_model_revision_overrides_lock(snapshot):
    result = model_sources.resolve_model_source("yorzoi", REPO, OTHER_REVISION, lock=_lock())
    assert result == snapshot.return_value
    assert snapshot.call_args.kwargs["revision"] == OTHER_REVISION


@pytest.mark.parametrize("revision", ["main", "v1", "a" * 7, "g" * 40, ""])
def test_explicit_model_revision_must_be_immutable(snapshot, revision):
    with pytest.raises(LockError, match="40-character"):
        model_sources.resolve_model_source("yorzoi", REPO, revision, lock=_lock())
    snapshot.assert_not_called()


def test_legacy_lock_retains_unpinned_behaviour(snapshot, caplog):
    result = model_sources.resolve_model_source(
        "yorzoi", REPO, lock={"version": 1, "artifacts": {}},
    )
    assert result == REPO
    snapshot.assert_not_called()
    assert "no pinned revision" in caplog.text


def test_override_repository_never_inherits_default_pin(snapshot):
    other = "another-lab/yorzoi"
    result = model_sources.resolve_model_source("yorzoi", other, lock=_lock())
    assert result == other
    snapshot.assert_not_called()


def test_override_repository_with_explicit_pin(snapshot):
    other = "another-lab/yorzoi"
    assert model_sources.resolve_model_source("yorzoi", other, OTHER_REVISION, lock=_lock()) == snapshot.return_value
    snapshot.assert_called_once_with(repo_id=other, repo_type="model", revision=OTHER_REVISION)


def test_strict_model_loading_rejects_unpinned_input(snapshot):
    with pytest.raises(LockError):
        model_sources.resolve_model_source(
            "yorzoi", REPO, lock={"version": 1, "artifacts": {}}, require_revision=True,
        )
    snapshot.assert_not_called()


def test_snapshot_failure_does_not_fall_back_to_latest(snapshot):
    snapshot.side_effect = OSError("download interrupted")
    with pytest.raises(OSError, match="interrupted"):
        model_sources.resolve_model_source("yorzoi", REPO, lock=_lock())
    assert snapshot.call_count == 1


def test_yorzoi_underlying_loader_receives_locked_snapshot(monkeypatch, snapshot):
    from yeastbench.models.yorzoi import Yorzoi

    load = Mock(return_value=object())
    monkeypatch.setitem(sys.modules, "yorzoi.model.borzoi", SimpleNamespace(Borzoi=SimpleNamespace(from_pretrained=load)))
    monkeypatch.setattr(model_sources, "read_lock", lambda: _lock())

    class CaptureYorzoi(Yorzoi):
        def __init__(self, model, **kwargs):
            self.model = model

    result = CaptureYorzoi.from_pretrained(REPO, device="cpu")
    load.assert_called_once_with(snapshot.return_value)
    assert result.model is load.return_value


@pytest.mark.parametrize("module_name,class_name,required", [
    ("yorzoi_brooks", "YorzoiBrooksPredictor", {}),
    ("yorzoi_meneu", "YorzoiMeneuPredictor", {}),
    ("yorzoi_eqtl", "YorzoiVariantScorer", {"fasta_path": "ref.fa", "gtf_path": "ref.gtf"}),
    ("yorzoi_mpra_marginalized", "YorzoiMPRAMarginalizedPredictor", {"fasta_path": "ref.fa", "gtf_path": "ref.gtf"}),
    ("yorzoi_shalem", "YorzoiShalemPredictor", {"fasta_path": "ref.fa", "gtf_path": "ref.gtf"}),
    ("yorzoi_wu", "YorzoiWuPredictor", {"fasta_path": "ref.fa", "gtf_path": "ref.gtf"}),
    ("yorzoi_hong", "YorzoiHongPredictor", {"fasta_path": "ref.fa"}),
    ("yorzoi_cuperus", "YorzoiCuperusPredictor", {"fasta_path": "ref.fa"}),
    ("yorzoi_ytk", "YorzoiYTKPredictor", {"fasta_path": "ref.fa"}),
    ("yorzoi_chen_marginalized", "YorzoiChenPredictor", {"fasta_path": "ref.fa", "hosts_path": "hosts.json", "data_dir": "chen"}),
])
def test_every_yorzoi_adapter_forwards_revision(monkeypatch, module_name, class_name, required):
    module = importlib.import_module(f"yeastbench.adapters.{module_name}")
    base = getattr(module, class_name)
    load = Mock(return_value=object())
    monkeypatch.setattr(module.Yorzoi, "from_pretrained", load)

    class CaptureAdapter(base):
        def __init__(self, model, **kwargs):
            self.model = model

    result = CaptureAdapter.from_pretrained(hf_repo=REPO, hf_revision=REVISION, device="cpu", **required)
    assert load.call_args.args == (REPO,)
    assert load.call_args.kwargs["hf_revision"] == REVISION
    assert result.model is load.return_value


def test_codon_transformer_model_and_tokenizer_share_locked_snapshot(monkeypatch, snapshot):
    pytest.importorskip("torch")
    from yeastbench.adapters.baselines.codon_transformer import CodonTransformerBaselinePredictor

    repo = "adibvafa/CodonTransformer"
    monkeypatch.setattr(model_sources, "read_lock", lambda: _lock("codon_transformer", repo))
    tokenizer_load = Mock(return_value=object())
    model = Mock()
    model.eval.return_value = model
    model.to.return_value = model
    model_load = Mock(return_value=model)
    monkeypatch.setitem(sys.modules, "CodonTransformer.CodonUtils", SimpleNamespace(TOKEN2INDEX={"a": 1}))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=tokenizer_load),
        BigBirdForMaskedLM=SimpleNamespace(from_pretrained=model_load),
    ))

    result = CodonTransformerBaselinePredictor.from_task(object(), device="cpu")
    snapshot.assert_called_once_with(repo_id=repo, repo_type="model", revision=REVISION)
    tokenizer_load.assert_called_once_with(snapshot.return_value)
    model_load.assert_called_once_with(snapshot.return_value)
    assert result.model is model


def test_codon_transformer_forwards_explicit_repository_and_revision(monkeypatch):
    pytest.importorskip("torch")
    from yeastbench.adapters.baselines.codon_transformer import CodonTransformerBaselinePredictor

    resolve = Mock(return_value="/frozen/model")
    monkeypatch.setattr(model_sources, "resolve_model_source", resolve)
    model = Mock()
    monkeypatch.setitem(sys.modules, "CodonTransformer.CodonUtils", SimpleNamespace(TOKEN2INDEX={}))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=Mock()),
        BigBirdForMaskedLM=SimpleNamespace(from_pretrained=Mock(return_value=model)),
    ))
    CodonTransformerBaselinePredictor.from_task(
        object(), device="cpu", hf_repo="custom/model", hf_revision=OTHER_REVISION,
    )
    resolve.assert_called_once_with("codon_transformer", "custom/model", OTHER_REVISION)
