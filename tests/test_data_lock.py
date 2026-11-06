"""Input locks preserve byte identity and pin only verified Hub sources."""
from __future__ import annotations

import copy
import json
from dataclasses import replace

import pytest

from yeastbench.data.artifacts import BackendKind, InputArtifact, InputKind, Mirror
from yeastbench.data.lock import (
    LockError,
    build_lock_entries,
    locked_files,
    merge_lock,
    prune_sources,
    read_lock,
    source_for,
    validate_artifact_lock,
    with_hf_revision,
    write_lock,
)

REVISION = "a" * 40
OTHER_REVISION = "b" * 40


def artifact(**overrides):
    art = InputArtifact(
        id="example",
        kind=InputKind.TASK_PROCESSED,
        dest="data/example",
        mirrors=(Mirror(BackendKind.HF, base="lab/data", prefix="example/"),),
        include=("measurements.tsv", "sequences.fa"),
    )
    return replace(art, **overrides)


@pytest.fixture
def local_input(tmp_path):
    art = artifact()
    base = tmp_path / art.dest
    base.mkdir(parents=True)
    (base / "measurements.tsv").write_text("sample\tmeasurement\na\t1\n")
    (base / "sequences.fa").write_text(">a\nACGT\n")
    return art, tmp_path


@pytest.fixture
def unpinned_lock(local_input):
    art, root = local_input
    return {"version": 1, "artifacts": build_lock_entries([art], root)}


def pin(lock, art=None, revision=REVISION):
    return with_hf_revision(lock, [art or artifact()], "lab/data", "dataset", revision)


def test_legacy_lock_does_not_invent_source(unpinned_lock, tmp_path):
    path = tmp_path / "lock.json"
    path.write_text(json.dumps(unpinned_lock))
    loaded = read_lock(path)
    assert loaded == unpinned_lock
    assert source_for(loaded, "example", artifact().mirrors[0]).revision is None
    with pytest.raises(LockError, match="no pinned"):
        validate_artifact_lock(artifact(), loaded, require_pinned=True)


def test_build_refuses_missing_artifact_directory(tmp_path):
    with pytest.raises(LockError, match="no locked files"):
        build_lock_entries([artifact()], tmp_path)


def test_build_refuses_missing_explicit_member(local_input):
    art, root = local_input
    (root / art.dest / "sequences.fa").unlink()
    with pytest.raises(LockError, match="sequences.fa"):
        build_lock_entries([art], root)


def test_build_requires_each_declared_pattern(local_input):
    art, root = local_input
    art = replace(art, include=("*.tsv", "missing/*.fa"))
    with pytest.raises(LockError, match="missing/\\*.fa"):
        build_lock_entries([art], root)


def test_recursive_pattern_includes_top_level_and_nested_files(local_input):
    art, root = local_input
    base = root / art.dest
    (base / "nested").mkdir()
    (base / "nested" / "more.tsv").write_text("value\n2\n")
    art = replace(art, include=("**/*.tsv",))
    lock = {"version": 2, "artifacts": build_lock_entries([art], root)}
    validate_artifact_lock(art, lock)
    assert {x.path for x in locked_files(lock, art.id)} == {
        "measurements.tsv", "nested/more.tsv",
    }


def test_build_rejects_file_symlink_outside_artifact(local_input):
    art, root = local_input
    outside = root / "outside.fa"
    outside.write_text(">other\nAAAA\n")
    path = root / art.dest / "sequences.fa"
    path.unlink()
    path.symlink_to(outside)
    with pytest.raises(LockError, match="outside"):
        build_lock_entries([art], root)


def test_build_rejects_file_symlink_inside_artifact(local_input):
    art, root = local_input
    base = root / art.dest
    path = base / "sequences.fa"
    path.rename(base / "original.fa")
    path.symlink_to("original.fa")
    with pytest.raises(LockError, match="symlink"):
        build_lock_entries([art], root)


def test_relocking_unchanged_bytes_preserves_revision(local_input, unpinned_lock):
    art, root = local_input
    original = pin(unpinned_lock)
    updated = merge_lock(original, build_lock_entries([art], root))
    assert updated == original


def test_relocking_changed_bytes_clears_revision(local_input, unpinned_lock):
    art, root = local_input
    original = pin(unpinned_lock)
    original_copy = copy.deepcopy(original)
    (root / art.dest / "sequences.fa").write_text(">a\nTGCA\n")
    updated = merge_lock(original, build_lock_entries([art], root))
    assert "sources" not in updated["artifacts"][art.id]
    assert original == original_copy
    assert source_for(updated, art.id, art.mirrors[0]).revision is None


def test_relocking_preserves_unselected_artifacts(local_input, unpinned_lock):
    art, root = local_input
    original = pin(unpinned_lock)
    original["artifacts"]["unselected"] = copy.deepcopy(original["artifacts"][art.id])
    updated = merge_lock(original, build_lock_entries([art], root))
    assert updated["artifacts"]["unselected"] == original["artifacts"]["unselected"]
    updated["artifacts"]["unselected"]["files"].clear()
    assert original["artifacts"]["unselected"]["files"]


def test_pin_updates_copy_and_replaces_prior_source(unpinned_lock):
    original_copy = copy.deepcopy(unpinned_lock)
    locked = pin(unpinned_lock)
    newer = pin(locked, revision=OTHER_REVISION)
    assert unpinned_lock == original_copy
    assert newer["version"] == 2
    assert len(newer["artifacts"]["example"]["sources"]) == 1
    assert source_for(newer, "example", artifact().mirrors[0]).revision == OTHER_REVISION
    assert source_for(locked, "example", artifact().mirrors[0]).revision == REVISION


@pytest.mark.parametrize("change", [
    {"base": "other/data"}, {"repo_type": "model"}, {"prefix": "other/"},
])
def test_source_pin_does_not_leak_to_other_repo_type_or_prefix(unpinned_lock, change):
    mirror = replace(artifact().mirrors[0], **change)
    assert source_for(pin(unpinned_lock), "example", mirror).revision is None
    with pytest.raises(LockError, match="no pinned"):
        source_for(pin(unpinned_lock), "example", mirror, require_revision=True)


def test_explicit_revision_cannot_conflict_with_locked_revision(unpinned_lock):
    mirror = replace(artifact().mirrors[0], revision=OTHER_REVISION)
    with pytest.raises(LockError, match="differs"):
        source_for(pin(unpinned_lock), "example", mirror)


def test_declared_source_pin_can_be_used_without_lock_source(unpinned_lock):
    mirror = replace(artifact().mirrors[0], revision=REVISION)
    assert source_for(unpinned_lock, "example", mirror, require_revision=True) == mirror


def test_pin_requires_declared_repository(unpinned_lock):
    with pytest.raises(LockError, match="no declared"):
        with_hf_revision(unpinned_lock, [artifact()], "other/data", "dataset", REVISION)


def test_cache_only_model_gets_revision_without_local_file_inventory():
    art = artifact(
        id="model", kind=InputKind.MODEL_WEIGHTS, dest="", cache_only=True, include=(),
        mirrors=(Mirror(BackendKind.HF, base="lab/model", repo_type="model"),),
    )
    lock = {"version": 1, "artifacts": {}}
    with pytest.raises(LockError, match="no pinned"):
        validate_artifact_lock(art, lock, require_pinned=True)
    pinned = with_hf_revision(lock, [art], "lab/model", "model", REVISION)
    validate_artifact_lock(art, pinned, require_pinned=True)
    assert pinned["artifacts"]["model"]["files"] == {}
    assert source_for(pinned, "model", art.mirrors[0]).revision == REVISION


def test_lock_with_no_entry_is_not_ready():
    with pytest.raises(LockError, match="no locked files"):
        validate_artifact_lock(artifact(), {"version": 2, "artifacts": {}})


def test_lock_rejects_stale_declared_inventory(unpinned_lock):
    art = artifact(include=("measurements.tsv",))
    with pytest.raises(LockError, match="outside the declared"):
        validate_artifact_lock(art, unpinned_lock)


def test_lock_rejects_source_after_mirror_change(unpinned_lock):
    art = artifact(mirrors=(Mirror(BackendKind.HF, base="changed/repo", prefix="example/"),))
    with pytest.raises(LockError, match="not a declared"):
        validate_artifact_lock(art, pin(unpinned_lock))


def test_prune_preserves_matching_sources_and_unselected_artifacts(unpinned_lock):
    original = pin(unpinned_lock)
    original["artifacts"]["other"] = copy.deepcopy(original["artifacts"]["example"])
    pruned = prune_sources(original, [artifact()])
    assert pruned == original
    pruned["artifacts"]["other"]["sources"].clear()
    assert original["artifacts"]["other"]["sources"]


@pytest.mark.parametrize("change", [
    {"base": "renamed/data"}, {"prefix": "renamed/"}, {"repo_type": "model"},
])
def test_prune_removes_renamed_source_only_for_selected_artifact(unpinned_lock, change):
    original = pin(unpinned_lock)
    original["artifacts"]["other"] = copy.deepcopy(original["artifacts"]["example"])
    art = artifact(mirrors=(replace(artifact().mirrors[0], **change),))
    pruned = prune_sources(original, [art])
    assert "sources" not in pruned["artifacts"]["example"]
    assert pruned["artifacts"]["other"] == original["artifacts"]["other"]
    assert original["artifacts"]["example"]["sources"]


def test_repin_replaces_renamed_source_without_mutating_original(unpinned_lock):
    original = pin(unpinned_lock)
    art = artifact(mirrors=(Mirror(BackendKind.HF, base="renamed/data", prefix="new/"),))
    updated = with_hf_revision(original, [art], "renamed/data", "dataset", OTHER_REVISION)
    validate_artifact_lock(art, updated, require_pinned=True)
    assert len(updated["artifacts"][art.id]["sources"]) == 1
    assert source_for(updated, art.id, art.mirrors[0]).revision == OTHER_REVISION
    assert original["artifacts"][art.id]["sources"][0]["repo_id"] == "lab/data"


def test_prune_does_not_hide_malformed_lock():
    with pytest.raises(LockError, match="unsupported"):
        prune_sources({"version": 999, "artifacts": {}}, [artifact()])


@pytest.mark.parametrize("path", ["../escape", "/absolute", "x/../y", "x//y", "x\\y"])
def test_read_rejects_unsafe_filenames(unpinned_lock, tmp_path, path):
    entry = unpinned_lock["artifacts"]["example"]["files"]
    entry[path] = entry.pop("sequences.fa")
    file = tmp_path / "lock.json"
    file.write_text(json.dumps(unpinned_lock))
    with pytest.raises(LockError, match="invalid relative path"):
        read_lock(file)


@pytest.mark.parametrize("field,value", [
    ("sha256", "short"), ("sha256", None), ("size", -1), ("size", True), ("size", "4"),
])
def test_read_rejects_invalid_file_metadata(unpinned_lock, tmp_path, field, value):
    unpinned_lock["artifacts"]["example"]["files"]["sequences.fa"][field] = value
    file = tmp_path / "lock.json"
    file.write_text(json.dumps(unpinned_lock))
    with pytest.raises(LockError):
        read_lock(file)


@pytest.mark.parametrize("revision", ["main", "abc1234", "z" * 40, None])
def test_pin_rejects_nonimmutable_revision(unpinned_lock, revision):
    with pytest.raises(LockError, match="40-character"):
        pin(unpinned_lock, revision=revision)


def test_read_rejects_duplicate_json_keys(tmp_path):
    file = tmp_path / "lock.json"
    file.write_text('{"version": 1, "version": 2, "artifacts": {}}')
    with pytest.raises(LockError, match="duplicate JSON key"):
        read_lock(file)


def test_read_rejects_unknown_schema(tmp_path):
    file = tmp_path / "lock.json"
    file.write_text('{"version": 999, "artifacts": {}}')
    with pytest.raises(LockError, match="unsupported"):
        read_lock(file)


def test_write_roundtrips_pinned_lock(unpinned_lock, tmp_path):
    file = tmp_path / "lock.json"
    expected = pin(unpinned_lock)
    write_lock(expected, file)
    assert read_lock(file) == expected


def test_write_invalid_lock_preserves_existing_file(unpinned_lock, tmp_path):
    file = tmp_path / "lock.json"
    write_lock(unpinned_lock, file)
    before = file.read_bytes()
    with pytest.raises(LockError):
        write_lock({"version": 999, "artifacts": {}}, file)
    assert file.read_bytes() == before
