"""File inventories reject unsafe paths and changes to saved payloads."""

from __future__ import annotations

import pytest

from yeastbench.storage.files import (
    inspect_file,
    inventory_files,
    sha256_file,
    validate_files,
    validate_relative_path,
)


@pytest.mark.parametrize(
    "path",
    ["", "/file", "../file", "a/../file", "./file", "a//file", "a/./file",
     "a/", "a\\file", "C:/file", "a\nfile", "a\x00file"],
)
def test_relative_paths_reject_aliases_and_traversal(path):
    with pytest.raises(ValueError):
        validate_relative_path(path)


def test_inventory_detects_changes_and_unlisted_files(tmp_path):
    (tmp_path / "scores.npy").write_bytes(b"predictions")
    (tmp_path / "summary.json").write_text('{"score": 0.8}')
    records = inventory_files(tmp_path)
    validate_files(tmp_path, records, exact=True)

    (tmp_path / "extra.log").write_text("extra")
    validate_files(tmp_path, records)
    with pytest.raises(ValueError, match="do not match"):
        validate_files(tmp_path, records, exact=True)

    (tmp_path / "scores.npy").write_bytes(b"changed!!!!")
    with pytest.raises(ValueError, match="differs.*scores.npy"):
        validate_files(tmp_path, records)


def test_inventory_rejects_missing_files_and_symlinked_directories(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "file").write_text("content")
    root = tmp_path / "root"
    root.mkdir()
    (root / "linked").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        inventory_files(root, ["linked/file"])
    with pytest.raises(ValueError, match="missing"):
        inventory_files(root, ["missing"])
    with pytest.raises(ValueError, match="symlink"):
        inventory_files(root)


def test_inventory_order_is_stable_and_duplicates_are_rejected(tmp_path):
    (tmp_path / "a").write_text("a")
    (tmp_path / "b").write_text("b")
    assert inventory_files(tmp_path, ["b", "a"]) == inventory_files(tmp_path)
    with pytest.raises(ValueError, match="duplicate"):
        inventory_files(tmp_path, ["a", "a"])


@pytest.mark.parametrize("inventory", [False, True])
def test_file_changed_while_hashing_is_rejected(tmp_path, monkeypatch, inventory):
    path = tmp_path / "predictions.tsv"
    path.write_text("original")

    def mutate_after_hash(local_path):
        digest = sha256_file(local_path)
        local_path.write_text("updated predictions")
        return digest

    monkeypatch.setattr("yeastbench.storage.files.sha256_file", mutate_after_hash)
    with pytest.raises(ValueError, match="changed while hashing"):
        if inventory:
            inventory_files(tmp_path)
        else:
            inspect_file(path, "run/predictions.tsv")
