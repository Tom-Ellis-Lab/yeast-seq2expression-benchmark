"""Tests for the data manifest / lock / fetch machinery.

Pure-logic tests run anywhere (no network, no big local data). One guarded
network test fetches a small public file to exercise the real download+verify
path; it skips when offline.
"""

from __future__ import annotations

import urllib.error
from dataclasses import replace

import pytest

from yeastbench.data import manifest
from yeastbench.data.artifacts import (
    BackendKind,
    InputArtifact,
    InputKind,
    Mirror,
)
from yeastbench.data.backends import HttpBackend
from yeastbench.data.fetch import (
    _file_state,
    build_plans,
    check_run_data,
)
from yeastbench.data.lock import (
    LOCK_PATH,
    build_lock_entries,
    local_files,
    locked_files,
    read_lock,
)
from yeastbench.data.manifest import artifact_by_id, artifacts_for
from yeastbench.storage.files import sha256_file

# ──────────────────────────────────────────────────────────────
# Registry coverage — every task/model has declared data
# ──────────────────────────────────────────────────────────────


def test_every_registered_task_and_model_has_an_artifact():
    problems = manifest.validate_against_registry()
    assert problems == [], "\n".join(problems)


def test_artifact_ids_unique():
    ids = [a.id for a in manifest.ARTIFACTS]
    assert len(ids) == len(set(ids))


# ──────────────────────────────────────────────────────────────
# Selection — closes over `requires`, rejects typos, tolerates baselines
# ──────────────────────────────────────────────────────────────


def test_selection_pulls_required_shared_refs():
    arts = artifacts_for(tasks=["cuperus_utr"])
    ids = {a.id for a in arts}
    assert "cuperus_mpra_5utr" in ids
    assert "refs" in ids  # pulled in via `requires`


def test_selection_chen_task_pulls_its_artifact():
    arts = artifacts_for(tasks=["chen_synonymous"])
    assert sum(a.id == "chen_synonymous" for a in arts) == 1


def test_selection_unknown_name_raises():
    with pytest.raises(KeyError):
        artifacts_for(tasks=["does_not_exist"])


def test_selection_tolerates_weightless_baseline():
    # cai has no downloadable artifact; selecting it alongside a real task
    # must not raise.
    arts = artifacts_for(tasks=["chen_synonymous"], models=["cai"])
    assert any(a.id == "chen_synonymous" for a in arts)


def test_selection_all_when_unfiltered():
    assert artifacts_for() == list(manifest.ARTIFACTS)


# ──────────────────────────────────────────────────────────────
# Lock — build / verify on a synthetic tree (no real data needed)
# ──────────────────────────────────────────────────────────────


def _fake_artifact(dest: str) -> InputArtifact:
    return InputArtifact(
        id="fake",
        kind=InputKind.TASK_PROCESSED,
        dest=dest,
        mirrors=(Mirror(BackendKind.HTTP, base="https://example.invalid/"),),
        include=("**/*",),
    )


def test_lock_build_and_excludes_cache(tmp_path):
    root = tmp_path
    base = root / "blob"
    (base / "sub").mkdir(parents=True)
    (base / "a.tsv").write_text("hello")
    (base / "sub" / "b.tsv").write_text("world")
    (base / "_cache").mkdir()
    (base / "_cache" / "raw.bin").write_text("ignore me")

    art = _fake_artifact("blob")
    rels = local_files(art, root)
    assert rels == ["a.tsv", "sub/b.tsv"]  # _cache excluded, sorted

    lock = build_lock_entries([art], root)
    assert set(lock["fake"]["files"]) == {"a.tsv", "sub/b.tsv"}
    assert lock["fake"]["files"]["a.tsv"]["sha256"] == sha256_file(base / "a.tsv")
    assert lock["fake"]["files"]["a.tsv"]["size"] == 5


def test_file_state_ok_stale_missing(tmp_path):
    f = tmp_path / "x"
    f.write_text("data")
    good = sha256_file(f)
    assert _file_state(f, good) == "ok"
    assert _file_state(f, "0" * 64) == "stale"
    assert _file_state(tmp_path / "nope", good) == "missing"


# ──────────────────────────────────────────────────────────────
# Planning — mirror choice + per-file state, no network
# ──────────────────────────────────────────────────────────────


def test_build_plans_reports_present_files_as_ok(tmp_path):
    root = tmp_path
    base = root / "blob"
    base.mkdir()
    (base / "a.tsv").write_text("hello")
    art = _fake_artifact("blob")
    lock = {"version": 1, "artifacts": build_lock_entries([art], root)}

    plans = build_plans([art], lock, from_kind=BackendKind.HTTP, data_root=root)
    assert len(plans) == 1
    p = plans[0]
    assert p.mirror is not None and p.mirror.backend == BackendKind.HTTP
    assert [f.state for f in p.files] == ["ok"]
    assert p.to_fetch == []


def test_build_plans_marks_missing_files_for_fetch(tmp_path):
    root = tmp_path
    (root / "blob").mkdir()
    art = _fake_artifact("blob")
    # lock references a file that isn't on disk
    lock = {"version": 1, "artifacts": {"fake": {"files": {
        "ghost.tsv": {"sha256": "0" * 64, "size": 1}}}}}
    plans = build_plans([art], lock, from_kind=None, data_root=root)
    assert [f.state for f in plans[0].files] == ["missing"]
    assert len(plans[0].to_fetch) == 1


# ──────────────────────────────────────────────────────────────
# Committed lock — internal consistency (no local data dependency)
# ──────────────────────────────────────────────────────────────


def test_committed_lock_covers_shorkie_and_refs():
    lock = read_lock(LOCK_PATH)
    shorkie = {lf.path for lf in locked_files(lock, "shorkie")}
    assert "params.json" in shorkie
    assert "targets.txt" in shorkie
    assert sum(r.startswith("checkpoints/f") for r in shorkie) == 8

    refs = {lf.path for lf in locked_files(lock, "refs")}
    assert {"R64-1-1.fa", "R64-5-1.fa"} <= refs


# ──────────────────────────────────────────────────────────────
# Real network fetch (small, public) — proves download + checksum gate
# ──────────────────────────────────────────────────────────────


@pytest.mark.network
@pytest.mark.parametrize("model", ["shorkie", "shorkie_random_init"])
def test_http_backend_fetches_shorkie_params_matching_lock(tmp_path, model):
    """Fetch the (2.7 KB) Shorkie params.json from its public source and check
    it hashes to what the committed lock expects."""
    art = artifact_by_id(model)
    mirror = art.mirrors[0]
    lock = read_lock(LOCK_PATH)
    want = {lf.path: lf.sha256 for lf in locked_files(lock, model)}

    out = tmp_path / "params.json"
    try:
        HttpBackend().fetch(mirror, mirror.remote_for("params.json"), out)
    except urllib.error.URLError as e:  # offline / CI without egress
        pytest.skip(f"network unavailable: {e}")
    assert sha256_file(out) == want["params.json"]


# ──────────────────────────────────────────────────────────────
# Run preflight — check_run_data over the manifest + lock
# ──────────────────────────────────────────────────────────────


def test_check_run_data_flags_missing(tmp_path):
    """Empty data root → every locked file for the selection is missing."""
    check = check_run_data(["caudal_eqtl"], [], tmp_path, lock=read_lock())
    assert check.files_total > 0
    assert check.files_present == 0
    assert not check.ready
    assert len(check.missing) == check.files_total


def test_check_run_data_counts_present_files(tmp_path, monkeypatch):
    """A file present and matching the (synthetic) lock counts as ready."""
    art = artifact_by_id("hong")
    rel = locked_files(read_lock(), "hong")[0].path  # a real locked relpath
    # This test isolates one file; real Hong also requires other files and refs.
    monkeypatch.setattr(
        "yeastbench.data.fetch.artifacts_for",
        lambda *_: [replace(art, include=(rel,), requires=())],
    )
    f = tmp_path / art.dest / rel
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes(b"hello world")
    synth = {
        "version": 1,
        "artifacts": {
            "hong": {
                "files": {rel: {"sha256": sha256_file(f), "size": f.stat().st_size}}
            }
        },
    }
    check = check_run_data(["hong_igr"], [], tmp_path, lock=synth)
    assert check.files_total == 1
    assert check.files_present == 1
    assert check.ready


def test_check_run_data_detects_stale(tmp_path):
    """A file present but with the wrong bytes is stale, not ready."""
    art = artifact_by_id("hong")
    rel = locked_files(read_lock(), "hong")[0].path
    f = tmp_path / art.dest / rel
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes(b"wrong bytes")
    synth = {
        "version": 1,
        "artifacts": {"hong": {"files": {rel: {"sha256": "0" * 64, "size": 999}}}},
    }
    check = check_run_data(["hong_igr"], [], tmp_path, lock=synth)
    assert check.stale == [("hong", rel)]
    assert not check.ready


def test_check_run_data_lists_cache_only_models(tmp_path):
    """cache-only HF models are reported, not counted as a missing-file failure."""
    check = check_run_data([], ["yorzoi"], tmp_path, lock=read_lock())
    assert "yorzoi" in check.cache_only
    assert check.ready


def test_cuperus_reporter_construct_is_locked():
    artifact = artifact_by_id("cuperus_mpra_5utr")
    assert "construct.json" in artifact.include
    files = {f.path for f in locked_files(read_lock(), artifact.id)}
    assert "construct.json" in files
