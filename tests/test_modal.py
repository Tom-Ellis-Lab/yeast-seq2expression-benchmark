"""Unit tests for the Modal backend's pure layer + CLI wiring.

The pure helpers (``yeastbench.modal.plan``) need no ``modal`` install and run in
CI. Anything that imports the Modal client is guarded with ``importorskip`` and
makes no remote calls — the actual remote dispatch is exercised manually behind
``modal setup``, not here.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from yeastbench.modal.plan import build_plan

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CFG = REPO / "configs" / "default.yaml"


# ── plan.build_plan ──────────────────────────────────────────────────────────


def test_build_plan_default_config():
    plan = build_plan(DEFAULT_CFG)
    assert plan.config_name == "default.yaml"
    assert plan.config_bytes == DEFAULT_CFG.read_bytes()  # exact bytes → hash parity
    assert plan.out_dir == "results/default"
    assert plan.pairs  # non-empty
    assert ("codon_transformer", "chen_synonymous") in plan.pairs  # now supported
    assert all("__" in d for d in plan.pair_dirs())


def test_build_plan_filters_to_single_pair():
    plan = build_plan(DEFAULT_CFG, model="cai", task="chen_synonymous")
    assert plan.pairs == [("cai", "chen_synonymous")]
    assert plan.pair_dirs() == ["cai__chen_synonymous"]


def test_build_plan_no_match_raises():
    with pytest.raises(ValueError, match="No runs match"):
        build_plan(DEFAULT_CFG, model="does-not-exist")


def test_build_plan_allows_codon_transformer():
    # codon_transformer is supported on Modal (installed --no-deps in the image)
    plan = build_plan(DEFAULT_CFG, model="codon_transformer")
    assert plan.pairs == [("codon_transformer", "chen_synonymous")]


@pytest.mark.parametrize(
    "out_dir",
    [
        "elsewhere/run",          # plainly outside results/
        "results/../scratch/run",  # traversal that escapes the mount
        "/results/absolute",       # absolute path is never under the mount
        "results/..",              # normalizes to "."
    ],
)
def test_build_plan_rejects_out_dir_outside_results(tmp_path, out_dir):
    cfg = tmp_path / "bad.yaml"
    cfg.write_text(
        f"out_dir: {out_dir}\ndevice: cpu\n"
        "runs:\n  - model: cai\n    tasks: [chen_synonymous]\n"
    )
    with pytest.raises(ValueError, match="out_dir under 'results/'"):
        build_plan(cfg)


def test_build_plan_normalizes_out_dir(tmp_path):
    cfg = tmp_path / "ok.yaml"
    cfg.write_text(
        "out_dir: results/./nested/../custom\ndevice: cpu\n"
        "runs:\n  - model: cai\n    tasks: [chen_synonymous]\n"
    )
    assert build_plan(cfg).out_dir == "results/custom"


def test_build_plan_accepts_results_subdir(tmp_path):
    cfg = tmp_path / "ok.yaml"
    cfg.write_text(
        "out_dir: results/custom\ndevice: cpu\n"
        "runs:\n  - model: cai\n    tasks: [chen_synonymous]\n"
    )
    assert build_plan(cfg).out_dir == "results/custom"


# ── CLI wiring + app spec (needs the modal client, but no remote calls) ──────


def test_modal_subapp_registered_on_main_cli():
    pytest.importorskip("modal")
    from typer.testing import CliRunner

    from yeastbench.cli import app

    result = CliRunner().invoke(app, ["modal", "--help"])
    assert result.exit_code == 0
    for cmd in ("run", "pull", "status", "data"):
        assert cmd in result.stdout


def test_validate_gpu_rejects_a_device_index():
    pytest.importorskip("modal")
    import typer

    from yeastbench.modal.cli import _validate_gpu

    assert _validate_gpu("A10") == "A10"
    assert _validate_gpu("A100-80GB") == "A100-80GB"
    with pytest.raises(typer.BadParameter, match="not a device index"):
        _validate_gpu("0")


def test_modal_executable_falls_back_beside_python(tmp_path, monkeypatch):
    pytest.importorskip("modal")
    from yeastbench.modal import cli

    python = tmp_path / "python"
    modal = tmp_path / "modal"
    modal.touch()
    monkeypatch.setattr(cli.shutil, "which", lambda name: None)
    monkeypatch.setattr(cli.sys, "executable", str(python))
    assert cli._modal_executable() == str(modal)


def test_modal_app_spec_is_two_volumes():
    pytest.importorskip("modal")
    from yeastbench.modal import app as modal_app

    assert modal_app.APP_NAME == "ybench"
    assert set(modal_app.VOLUMES) == {"/repo/data", "/repo/results"}
    assert modal_app.HF_ENV["HF_HOME"] == "/repo/data/.hf"
