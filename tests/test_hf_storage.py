"""Snapshot publication boundaries and commit semantics, without Hub access."""

from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from yeastbench.artifacts import sha256_file
from yeastbench.storage.hf import HfStore, StorageError

HEAD = "a" * 40
NEW_HEAD = "b" * 40


class FakeApi:
    def __init__(self, files=(), lfs_paths=()):
        self.files = list(files)
        self.contents = {path: b"existing" for path in files}
        self.lfs_paths = set(lfs_paths)
        self.calls = []
        self.head = HEAD

    def repo_info(self, **kwargs):
        self.calls.append(("repo_info", kwargs))
        if kwargs.get("files_metadata"):
            assert kwargs["revision"] == self.head
            return SimpleNamespace(sha=self.head, siblings=[
                SimpleNamespace(
                    rfilename=path, size=len(content),
                    blob_id=hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest(),
                    lfs=(SimpleNamespace(sha256=hashlib.sha256(content).hexdigest())
                         if path in self.lfs_paths else None),
                ) for path, content in self.contents.items()
            ])
        return SimpleNamespace(sha=self.head)

    def list_repo_files(self, **kwargs):
        self.calls.append(("list_repo_files", kwargs))
        assert kwargs["revision"] == HEAD
        return list(self.files)

    def create_commit(self, **kwargs):
        self.calls.append(("create_commit", kwargs))
        if kwargs["parent_commit"] != self.head:
            raise RuntimeError("parent commit conflict")
        for operation in kwargs["operations"]:
            if type(operation).__name__ == "CommitOperationDelete":
                self.contents.pop(operation.path_in_repo)
            else:
                self.contents[operation.path_in_repo] = Path(operation.path_or_fileobj).read_bytes()
        self.head = NEW_HEAD
        return SimpleNamespace(
            oid=NEW_HEAD,
            commit_url=f"https://huggingface.co/datasets/lab/data/commit/{NEW_HEAD}",
        )


@pytest.fixture
def local_file(tmp_path):
    path = tmp_path / "data.tsv"
    path.write_text("id\tmeasurement\na\t1.5\n")
    return path


def test_plan_reads_one_head_and_limits_deletion_to_owned_prefix(local_file):
    api = FakeApi([
        "README.md", ".gitattributes", "task/data.tsv", "task/old.tsv",
        "task_extra/keep.tsv", "other/task/file.tsv",
    ])
    plan = HfStore(api).plan(
        "lab/data", "dataset", [(local_file, "task/data.tsv")], ["task"],
        {"task/data.tsv": sha256_file(local_file)},
    )
    assert plan.parent_commit == HEAD
    assert plan.deletions == ("task/old.tsv",)
    assert plan.files[0].record.path == "task/data.tsv"
    assert plan.items == ((local_file, "task/data.tsv"),)
    assert [name for name, _ in api.calls] == ["repo_info", "list_repo_files"]
    assert api.calls[0][1]["revision"] == "main"
    assert api.calls[1][1]["revision"] == HEAD


def test_plan_digest_is_independent_of_local_path_and_item_order(tmp_path, local_file):
    duplicate = tmp_path / "duplicate.tsv"
    duplicate.write_bytes(local_file.read_bytes())
    store = HfStore(FakeApi())
    first = store.plan(
        "lab/data", "dataset",
        [(local_file, "task/a"), (duplicate, "task/b")], ["task/"],
    )
    second = store.plan(
        "lab/data", "dataset",
        [(local_file, "task/b"), (duplicate, "task/a")], ["task"],
    )
    assert first.manifest_sha256 == second.manifest_sha256


@pytest.mark.parametrize("prefixes", [[], [""], ["/"], ["../task"], ["task", "task/child"], ["task", "task/"]])
def test_plan_rejects_unsafe_scopes_before_contacting_hub(local_file, prefixes):
    api = FakeApi()
    with pytest.raises(StorageError):
        HfStore(api).plan("lab/data", "dataset", [(local_file, "task/data.tsv")], prefixes)
    assert api.calls == []


@pytest.mark.parametrize("remote", ["other/data.tsv", "task/../data.tsv", "task//data.tsv", "task/./data.tsv"])
def test_plan_rejects_unsafe_file_paths(local_file, remote):
    api = FakeApi()
    with pytest.raises(StorageError):
        HfStore(api).plan("lab/data", "dataset", [(local_file, remote)], ["task"])
    assert api.calls == []


def test_all_checksums_are_verified_before_remote_inspection(tmp_path, local_file):
    other = tmp_path / "other.tsv"
    other.write_text("incorrect")
    api = FakeApi()
    with pytest.raises(StorageError, match="expected checksum.*other"):
        HfStore(api).plan(
            "lab/data", "dataset",
            [(local_file, "task/data.tsv"), (other, "task/other.tsv")], ["task"],
            {"task/data.tsv": sha256_file(local_file), "task/other.tsv": "0" * 64},
        )
    assert api.calls == []


def test_duplicate_and_empty_uploads_are_rejected(local_file):
    api = FakeApi()
    with pytest.raises(StorageError, match="empty"):
        HfStore(api).plan("lab/data", "dataset", [], ["task"])
    with pytest.raises(StorageError, match="duplicate upload"):
        HfStore(api).plan(
            "lab/data", "dataset", [(local_file, "task/a")] * 2, ["task"],
        )
    assert api.calls == []


def test_each_owned_scope_must_have_a_file(local_file):
    api = FakeApi()
    with pytest.raises(StorageError, match="each owned prefix"):
        HfStore(api).plan("lab/data", "dataset", [(local_file, "task/a")], ["task", "other"])
    assert api.calls == []


def test_commit_submits_additions_deletions_and_parent_together(local_file):
    pytest.importorskip("huggingface_hub")
    api = FakeApi(["task/obsolete.tsv"])
    store = HfStore(api)
    plan = store.plan("lab/data", "dataset", [(local_file, "task/data.tsv")], ["task"])
    receipt = store.commit(plan, "Publish task data")
    commits = [kwargs for name, kwargs in api.calls if name == "create_commit"]
    assert len(commits) == 1
    commit = commits[0]
    assert commit["parent_commit"] == HEAD
    assert commit["revision"] == "main"
    assert commit["commit_message"] == "Publish task data"
    assert [(type(op).__name__, op.path_in_repo) for op in commit["operations"]] == [
        ("CommitOperationAdd", "task/data.tsv"),
        ("CommitOperationDelete", "task/obsolete.tsv"),
    ]
    assert receipt.revision == NEW_HEAD
    assert receipt.manifest_sha256 == plan.manifest_sha256
    assert receipt.to_dict()["url"].endswith(NEW_HEAD)
    assert api.calls[-1] == ("repo_info", {
        "repo_id": "lab/data", "repo_type": "dataset",
        "revision": NEW_HEAD, "files_metadata": True,
    })


def test_commit_verifies_lfs_content_hashes(local_file):
    pytest.importorskip("huggingface_hub")
    store = HfStore(FakeApi(lfs_paths=["task/data.tsv"]))
    plan = store.plan("lab/data", "dataset", [(local_file, "task/data.tsv")], ["task"])
    assert store.commit(plan, "Publish").revision == NEW_HEAD


def test_noop_response_with_changed_scope_is_not_recorded(local_file):
    pytest.importorskip("huggingface_hub")

    class NoOpApi(FakeApi):
        def create_commit(self, **kwargs):
            # Mimic the SDK's empty-commit shortcut: it returns main without
            # enforcing parent_commit, even if main changed since planning.
            self.head = NEW_HEAD
            self.contents["task/unexpected.tsv"] = b"concurrent upload"
            return SimpleNamespace(oid=NEW_HEAD, commit_url=f"https://example/{NEW_HEAD}")

    api = NoOpApi(["task/data.tsv"])
    api.contents["task/data.tsv"] = local_file.read_bytes()
    store = HfStore(api)
    plan = store.plan("lab/data", "dataset", [(local_file, "task/data.tsv")], ["task"])
    with pytest.raises(StorageError, match=f"returned commit {NEW_HEAD}.*unexpected files"):
        store.commit(plan, "Publish")


def test_noop_response_with_matching_scope_returns_valid_receipt(local_file):
    pytest.importorskip("huggingface_hub")

    class NoOpApi(FakeApi):
        def create_commit(self, **kwargs):
            return SimpleNamespace(oid=self.head, commit_url=f"https://example/{self.head}")

    api = NoOpApi(["task/data.tsv"])
    api.contents["task/data.tsv"] = local_file.read_bytes()
    store = HfStore(api)
    plan = store.plan("lab/data", "dataset", [(local_file, "task/data.tsv")], ["task"])
    assert store.commit(plan, "Publish").revision == HEAD


@pytest.mark.parametrize("lfs", [False, True])
def test_wrong_remote_contents_fail_verification_and_preserve_commit_identity(local_file, lfs):
    pytest.importorskip("huggingface_hub")

    class WrongContentApi(FakeApi):
        def create_commit(self, **kwargs):
            result = super().create_commit(**kwargs)
            content = self.contents["task/data.tsv"]
            self.contents["task/data.tsv"] = b"x" * len(content)
            return result

    store = HfStore(WrongContentApi(lfs_paths=["task/data.tsv"] if lfs else []))
    plan = store.plan("lab/data", "dataset", [(local_file, "task/data.tsv")], ["task"])
    with pytest.raises(StorageError, match=f"returned commit {NEW_HEAD}.*checksum differs"):
        store.commit(plan, "Publish")


def test_file_changed_after_planning_prevents_mutation(local_file):
    api = FakeApi()
    store = HfStore(api)
    plan = store.plan("lab/data", "dataset", [(local_file, "task/data.tsv")], ["task"])
    local_file.write_text("changed after validation")
    with pytest.raises(StorageError, match="changed after planning"):
        store.commit(plan, "Publish")
    assert all(name != "create_commit" for name, _ in api.calls)


def test_upload_reads_staged_bytes_when_source_changes(local_file):
    pytest.importorskip("huggingface_hub")
    expected = local_file.read_bytes()

    class MutatingApi(FakeApi):
        def create_commit(self, **kwargs):
            local_file.write_text("changed while uploading")
            operation = kwargs["operations"][0]
            assert Path(operation.path_or_fileobj).read_bytes() == expected
            return super().create_commit(**kwargs)

    store = HfStore(MutatingApi())
    plan = store.plan("lab/data", "dataset", [(local_file, "task/data.tsv")], ["task"])
    receipt = store.commit(plan, "Publish")
    assert receipt.revision == NEW_HEAD


def test_remote_concurrent_update_is_reported(local_file):
    pytest.importorskip("huggingface_hub")
    api = FakeApi()
    store = HfStore(api)
    plan = store.plan("lab/data", "dataset", [(local_file, "task/data.tsv")], ["task"])
    api.head = "c" * 40
    with pytest.raises(StorageError, match="parent commit conflict"):
        store.commit(plan, "Publish")


def test_commit_rejects_deletion_outside_scope(local_file):
    api = FakeApi()
    store = HfStore(api)
    plan = store.plan("lab/data", "dataset", [(local_file, "task/data.tsv")], ["task"])
    plan = replace(plan, deletions=("README.md",))
    with pytest.raises(StorageError, match="outside owned prefixes"):
        store.commit(plan, "Publish")
    assert all(name != "create_commit" for name, _ in api.calls)


def test_missing_repository_is_not_created(local_file):
    class MissingApi(FakeApi):
        def repo_info(self, **kwargs):
            raise RuntimeError("repository not found")

    with pytest.raises(StorageError, match="must already exist"):
        HfStore(MissingApi()).plan(
            "lab/data", "dataset", [(local_file, "task/data.tsv")], ["task"],
        )
