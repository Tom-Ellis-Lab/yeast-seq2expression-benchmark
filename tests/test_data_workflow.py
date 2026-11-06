"""Data publication/pinning contracts without network access or model weights."""
from __future__ import annotations

import copy
import hashlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from yeastbench.data import fetch
from yeastbench.data.artifacts import BackendKind, InputArtifact, InputKind, Mirror
from yeastbench.data.backends import HfBackend
from yeastbench.data.cli import app
from yeastbench.data.lock import (
    LockError,
    build_lock_entries,
    read_lock,
    with_hf_revision,
    write_lock,
)
from yeastbench.data.pinning import pin_inputs
from yeastbench.data.publish import run_publish
from yeastbench.storage.hf import HfPublisher

OLD = "1" * 40
NEW = "2" * 40


class FakeHub:
    def __init__(self):
        self.head = OLD
        self.paths = ["a/old.tsv", "a/input.tsv", "b/input.tsv", "other/keep", "README.md"]
        self.calls = []
        self.commits = []
        self.contents = {path: b"existing" for path in self.paths}

    def repo_info(self, **kwargs):
        self.calls.append(("info", kwargs))
        if kwargs.get("files_metadata"):
            assert kwargs["revision"] == self.head
            siblings = [SimpleNamespace(
                rfilename=path, size=len(content), lfs=None,
                blob_id=hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest(),
            ) for path, content in self.contents.items()]
            return SimpleNamespace(sha=self.head, siblings=siblings)
        return SimpleNamespace(sha=self.head)

    def list_repo_files(self, **kwargs):
        self.calls.append(("files", kwargs))
        assert kwargs["revision"] == self.head
        return self.paths

    def create_commit(self, **kwargs):
        assert kwargs["parent_commit"] == self.head
        self.commits.append(kwargs)
        for operation in kwargs["operations"]:
            if hasattr(operation, "path_or_fileobj"):
                self.contents[operation.path_in_repo] = Path(operation.path_or_fileobj).read_bytes()
            else:
                self.contents.pop(operation.path_in_repo, None)
        self.paths = list(self.contents)
        self.head = NEW
        return SimpleNamespace(oid=NEW, commit_url=f"https://example.invalid/commit/{NEW}")


@pytest.fixture
def inputs(tmp_path):
    artifacts = []
    for name in ("a", "b"):
        artifact = InputArtifact(
            id=name, kind=InputKind.TASK_PROCESSED, dest=f"data/{name}",
            mirrors=(Mirror(BackendKind.HF, base="example/data", prefix=f"{name}/"),),
            include=("input.tsv",), redistributable=True,
        )
        folder = tmp_path / artifact.dest
        folder.mkdir(parents=True)
        (folder / "input.tsv").write_text(f"{name}\t1\n")
        artifacts.append(artifact)
    lock = {"version": 1, "artifacts": build_lock_entries(artifacts, tmp_path)}
    return artifacts, lock, tmp_path


def test_missing_lock_is_failure_in_get_verify_and_preflight(inputs, monkeypatch):
    artifacts, _, root = inputs
    lock = {"version": 2, "artifacts": {}}
    plans = fetch.build_plans(artifacts, lock, None, root)
    logs = []
    summary = fetch.run_get(plans, log=logs.append)
    assert not summary.ok and len(summary.failures) == 2
    assert not any("up to date" in line for line in logs)
    assert all(row.status != "ok" for row in fetch.run_verify(artifacts, lock, root))
    monkeypatch.setattr(fetch, "artifacts_for", lambda *_: artifacts)
    check = fetch.check_run_data([], [], root, lock)
    assert not check.ready and len(check.invalid) == 2


def test_pinned_download_uses_old_revision_even_with_new_head(inputs, monkeypatch, tmp_path):
    artifacts, lock, root = inputs
    pinned = with_hf_revision(lock, artifacts, "example/data", "dataset", OLD)
    plans = fetch.build_plans(artifacts, pinned, BackendKind.HF, root, require_pinned=True)
    assert all(p.mirror.revision == OLD for p in plans)
    hub = pytest.importorskip("huggingface_hub")
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        assert kwargs["revision"] == OLD
        name = kwargs["filename"].split("/")[0]
        return root / f"data/{name}/input.tsv"

    monkeypatch.setattr(hub, "hf_hub_download", download)
    destination = tmp_path / "old-copy.tsv"
    HfBackend().fetch(plans[0].mirror, "a/input.tsv", destination)
    assert destination.read_text() == "a\t1\n"
    assert len(calls) == 1


@pytest.mark.parametrize("corrupt", [False, True])
def test_download_plan_restores_locked_files_and_rejects_wrong_bytes(inputs, monkeypatch, corrupt):
    artifacts, lock, root = inputs
    pinned = with_hf_revision(lock, artifacts, "example/data", "dataset", OLD)
    payloads = {
        f"{art.id}/input.tsv": (root / art.dest / "input.tsv").read_bytes()
        for art in artifacts
    }
    if corrupt:
        payloads["a/input.tsv"] = b"a\t9\n"  # Same size, different fingerprint.

    def download(mirror, remote, destination):
        assert mirror.revision == OLD
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payloads[remote])

    backend = SimpleNamespace(available=lambda: True, fetch=download)
    monkeypatch.setattr(fetch, "get_backend", lambda _: backend)
    destination = root / "downloaded"
    plans = fetch.build_plans(artifacts, pinned, BackendKind.HF, destination, require_pinned=True)
    summary = fetch.run_get(plans)

    assert summary.ok is not corrupt
    assert summary.fetched == (1 if corrupt else 2)
    assert not list(destination.rglob("*.part"))
    for art in artifacts:
        saved = destination / art.dest / "input.tsv"
        if corrupt and art.id == "a":
            assert not saved.exists()
            assert "checksum mismatch" in summary.failures[0]
        else:
            assert saved.read_bytes() == (root / art.dest / "input.tsv").read_bytes()


def test_cli_download_preview_shows_pinned_missing_files(inputs, monkeypatch):
    artifacts, lock, root = inputs
    pinned = with_hf_revision(lock, artifacts, "example/data", "dataset", OLD)
    lock_path = root / "lock.json"
    write_lock(pinned, lock_path)
    monkeypatch.setattr("yeastbench.data.cli._select", lambda *_: artifacts)
    monkeypatch.setattr(fetch, "get_backend", lambda _: SimpleNamespace(available=lambda: True))
    result = CliRunner().invoke(app, [
        "get", "--dry-run", "--require-pinned", "--lock-file", str(lock_path),
        "--data-root", str(root / "empty"),
    ])
    assert result.exit_code == 0, result.output
    assert f"example/data@{OLD}" in result.output
    assert "1 to fetch, 4B" in result.output


def test_cache_warm_uses_model_pin(monkeypatch):
    hub = pytest.importorskip("huggingface_hub")
    calls = []
    monkeypatch.setattr(hub, "snapshot_download", lambda **kw: calls.append(kw))
    HfBackend().warm(Mirror(BackendKind.HF, base="example/model", repo_type="model", revision=OLD))
    assert calls == [{"repo_id": "example/model", "repo_type": "model", "revision": OLD}]


@pytest.mark.parametrize("dry_run", [False, True])
def test_gcs_publication_preserves_file_mapping_without_hf_receipts(inputs, monkeypatch, dry_run):
    artifacts, lock, root = inputs
    artifacts = [replace(art, mirrors=(Mirror(
        BackendKind.GCS, base="gs://example/", prefix=f"{art.id}_v1/",
    ),)) for art in artifacts]
    copies = []
    backend = SimpleNamespace(upload=lambda mirror, local, remote: copies.append((mirror, local, remote)))
    monkeypatch.setattr("yeastbench.data.publish.GcsBackend", lambda _: backend)
    summary = run_publish(artifacts, lock, BackendKind.GCS, root, "copy", dry_run)
    assert summary.ok and summary.artifacts == 2 and summary.files == 2
    assert summary.updated_lock == lock and not summary.receipts
    expected = [
        (art.mirrors[0], root / art.dest / "input.tsv", f"{art.id}_v1/input.tsv")
        for art in artifacts
    ]
    assert copies == ([] if dry_run else expected)


def test_strict_download_rejects_legacy_pins_before_fetch(inputs):
    artifacts, lock, root = inputs
    assert all(p.note is None for p in fetch.build_plans(artifacts, lock, None, root))
    plans = fetch.build_plans(artifacts, lock, None, root, require_pinned=True)
    assert all("no pinned" in p.note for p in plans)
    assert not fetch.run_get(plans).ok


def test_publish_validates_whole_selection_before_remote_access(inputs):
    artifacts, lock, root = inputs
    (root / artifacts[1].dest / "input.tsv").write_text("changed")
    api = FakeHub()
    summary = run_publish(artifacts, lock, BackendKind.HF, root, "update", False, publisher=HfPublisher(api))
    assert not summary.ok
    assert not api.calls and not api.commits


def test_publish_one_commit_for_all_selected_artifacts_and_pins(inputs):
    pytest.importorskip("huggingface_hub")
    artifacts, lock, root = inputs
    original = copy.deepcopy(lock)
    api = FakeHub()
    receipts = []
    summary = run_publish(
        artifacts, lock, BackendKind.HF, root, "update", False,
        publisher=HfPublisher(api), on_commit=lambda receipt, updated: receipts.append((receipt, updated)),
    )
    assert summary.ok and summary.artifacts == 2 and summary.files == 2
    assert len(api.commits) == 1 and len(receipts) == 1
    assert summary.plans[0].deletions == ("a/old.tsv",)
    assert summary.receipts[0].revision == NEW
    assert lock == original
    assert all(summary.updated_lock["artifacts"][a.id]["sources"][0]["revision"] == NEW for a in artifacts)


def test_publish_dry_run_shows_deletion_without_writes(inputs):
    artifacts, lock, root = inputs
    api = FakeHub()
    logs = []
    summary = run_publish(
        artifacts[:1], lock, BackendKind.HF, root, "update", True,
        publisher=HfPublisher(api), log=logs.append,
    )
    assert summary.ok and not api.commits and not summary.receipts
    assert summary.updated_lock == lock
    assert any("delete a/old.tsv" in line for line in logs)
    assert summary.plans[0].deletions == ("a/old.tsv",)


def test_pin_existing_dataset_verifies_bytes_at_resolved_sha(inputs):
    pytest.importorskip("huggingface_hub")
    artifacts, lock, root = inputs
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        assert kwargs["revision"] == OLD
        return root / "data" / kwargs["filename"]

    pinned = pin_inputs(artifacts, lock, api=FakeHub(), download=download)
    assert pinned["version"] == 2 and len(calls) == 2
    assert all(pinned["artifacts"][a.id]["sources"][0]["revision"] == OLD for a in artifacts)
    assert "sources" not in lock["artifacts"]["a"]


def test_pin_mismatch_leaves_original_lock_unchanged(inputs):
    pytest.importorskip("huggingface_hub")
    artifacts, lock, root = inputs
    original = copy.deepcopy(lock)
    with pytest.raises(LockError, match="differs from lock"):
        pin_inputs(artifacts, lock, api=FakeHub(), download=lambda **_: root / "data/b/input.tsv")
    assert lock == original


def test_pin_can_verify_a_renamed_repository(inputs):
    pytest.importorskip("huggingface_hub")
    artifacts, lock, root = inputs
    pinned = with_hf_revision(lock, artifacts, "example/data", "dataset", OLD)
    renamed = [replace(
        a, mirrors=(replace(a.mirrors[0], base="example/renamed-data"),),
    ) for a in artifacts]
    updated = pin_inputs(
        renamed, pinned, api=FakeHub(),
        download=lambda **kw: root / "data" / kw["filename"],
    )
    assert all(updated["artifacts"][a.id]["sources"][0]["repo_id"] == "example/renamed-data" for a in artifacts)
    assert pinned["artifacts"]["a"]["sources"][0]["repo_id"] == "example/data"


def test_publish_can_replace_obsolete_source_pin(inputs):
    pytest.importorskip("huggingface_hub")
    artifacts, lock, root = inputs
    old = [replace(a, mirrors=(replace(a.mirrors[0], base="example/old-data"),)) for a in artifacts]
    lock = with_hf_revision(lock, old, "example/old-data", "dataset", OLD)
    result = run_publish(artifacts, lock, BackendKind.HF, root, "rename", False, publisher=HfPublisher(FakeHub()))
    assert result.ok
    assert all(result.updated_lock["artifacts"][a.id]["sources"][0]["repo_id"] == "example/data" for a in artifacts)


def test_pin_model_records_revision_without_downloading_weights(inputs):
    pytest.importorskip("huggingface_hub")
    artifacts, lock, _ = inputs
    model = replace(
        artifacts[0], cache_only=True, dest="", include=(),
        mirrors=(Mirror(BackendKind.HF, base="example/model", repo_type="model"),),
    )
    lock = {"version": 1, "artifacts": {}}
    pinned = pin_inputs([model], lock, api=FakeHub(), download=lambda **_: pytest.fail("download"))
    assert pinned["artifacts"]["a"]["files"] == {}
    assert pinned["artifacts"]["a"]["sources"][0]["revision"] == OLD


@pytest.mark.parametrize("flag", ["--artifacts", "--tasks"])
def test_cli_publish_alias_preview_and_receipt(inputs, monkeypatch, flag):
    pytest.importorskip("huggingface_hub")
    artifacts, lock, root = inputs
    lock_path = root / "lock.json"
    write_lock(lock, lock_path)
    api = FakeHub()
    monkeypatch.setattr("yeastbench.data.cli.artifact_by_id", lambda _: artifacts[0])
    monkeypatch.setattr("yeastbench.data.publish.HfPublisher", lambda: HfPublisher(api))
    args = ["publish", "--to", "hf", flag, "a", "--data-root", str(root),
            "--lock-file", str(lock_path), "--receipt-dir", str(root / "receipts")]
    preview = CliRunner().invoke(app, args)
    assert preview.exit_code == 0, preview.output
    assert "delete a/old.tsv" in preview.output
    assert not api.commits and read_lock(lock_path) == lock
    published = CliRunner().invoke(app, [*args, "--yes"])
    assert published.exit_code == 0, published.output
    assert NEW in published.output
    assert read_lock(lock_path)["artifacts"]["a"]["sources"][0]["revision"] == NEW
    assert len(list((root / "receipts").glob("*.json"))) == 1
