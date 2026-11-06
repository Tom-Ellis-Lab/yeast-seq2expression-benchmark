"""Publish selected input artifacts using the shared snapshot publisher.

Validate the whole selection before any remote writes. Hugging Face receives
one commit per repository; GCS remains a per-file mirror without deletions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from yeastbench.data.artifacts import BackendKind, InputArtifact, Mirror
from yeastbench.data.backends import BackendError, GcsBackend
from yeastbench.data.lock import (
    LockError,
    locked_files,
    prune_sources,
    validate_artifact_lock,
    with_hf_revision,
)
from yeastbench.storage.files import FileRecord, UploadFile, inspect_file
from yeastbench.storage.hf import (
    HfCommitPlan,
    HfPublicationReceipt,
    HfPublisher,
    StorageError,
)


@dataclass
class PublishSummary:
    artifacts: int = 0
    files: int = 0
    skipped: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    plans: list[HfCommitPlan] = field(default_factory=list)
    receipts: list[HfPublicationReceipt] = field(default_factory=list)
    updated_lock: dict | None = None

    @property
    def ok(self) -> bool:
        return not self.failures


@dataclass(frozen=True)
class _ArtifactUpload:
    """A selected input and the exact files to publish to its mirror."""

    artifact: InputArtifact
    mirror: Mirror
    files: tuple[UploadFile, ...]


def run_publish(
    artifacts: list[InputArtifact],
    lock: dict,
    to_kind: BackendKind,
    data_root: Path,
    message: str,
    dry_run: bool,
    log: Callable[[str], None] = lambda _: None,
    billing_project: str | None = None,
    *,
    publisher: HfPublisher | None = None,
    on_commit: Callable[[HfPublicationReceipt, dict], None] | None = None,
) -> PublishSummary:
    """Synchronize selected artifact prefixes, preserving unrelated artifacts.

``on_commit`` persists each receipt and revised local lock immediately after a
successful commit, including when a later repository fails. No revisions are
invented during a dry run. The input ``lock`` is never mutated.
    """
    summary = PublishSummary(updated_lock=lock)
    if to_kind not in (BackendKind.HF, BackendKind.GCS):
        summary.failures.append("publish supports hf or gcs")
        return summary
    try:
        validation_lock = prune_sources(lock, artifacts)
    except LockError as exc:
        summary.failures.append(str(exc))
        return summary

    selected: list[_ArtifactUpload] = []
    remote_paths: dict[tuple[str, str], set[str]] = {}
    seen: set[str] = set()
    for art in artifacts:
        if art.id in seen:
            summary.failures.append(f"duplicate artifact: {art.id}")
            continue
        seen.add(art.id)
        if art.cache_only or not art.redistributable:
            summary.skipped.append(f"{art.id}: cache-only or not redistributable")
            continue
        mirror = next((m for m in art.mirrors if m.backend == to_kind), None)
        if mirror is None:
            summary.failures.append(f"{art.id}: no {to_kind.value} mirror declared")
            continue
        try:
            validate_artifact_lock(art, validation_lock)
            files = []
            paths = remote_paths.setdefault((mirror.base, mirror.repo_type), set())
            for entry in locked_files(lock, art.id):
                local = data_root / art.dest / entry.path
                if inspect_file(local, entry.path) != entry:
                    raise ValueError(f"local differs from lock: {local}; rebuild and run data lock")
                remote = mirror.remote_for(entry.path)
                if remote in paths:
                    raise ValueError(f"multiple artifacts own remote path {remote}")
                paths.add(remote)
                files.append(UploadFile(local, FileRecord(remote, entry.size, entry.sha256)))
            selected.append(_ArtifactUpload(art, mirror, tuple(files)))
        except (LockError, ValueError, OSError) as exc:
            summary.failures.append(f"{art.id}: {exc}")
    if summary.failures:
        return summary

    if to_kind == BackendKind.GCS:
        backend = GcsBackend(billing_project)
        for upload in selected:
            art, mirror = upload.artifact, upload.mirror
            log(f"  {art.id} → {mirror.base}{mirror.prefix} ({len(upload.files)} files; copy only)")
            try:
                for file in upload.files:
                    log(f"      copy {file.local_path} → {mirror.base}{file.record.path}")
                    if not dry_run:
                        backend.upload(mirror, file.local_path, file.record.path)
                summary.artifacts += 1
                summary.files += len(upload.files)
            except BackendError as exc:
                summary.failures.append(f"{art.id}: {exc}")
                break
        return summary

    if not selected:
        return summary
    groups: dict[tuple[str, str], list[_ArtifactUpload]] = {}
    for upload in selected:
        mirror = upload.mirror
        groups.setdefault((mirror.base, mirror.repo_type), []).append(upload)
    try:
        if publisher is None:
            publisher = HfPublisher()
        # Plan every repository before making the first remote write.
        for (repo_id, repo_type), members in groups.items():
            plan = publisher.plan(
                repo_id=repo_id,
                repo_type=repo_type,
                files=[file for upload in members for file in upload.files],
                owned_prefixes=[upload.mirror.prefix for upload in members],
            )
            summary.plans.append(plan)
            log(f"  {repo_type}:{repo_id} at {plan.parent_commit}")
            for file in plan.files:
                log(f"      add/update {file.local_path} → {file.record.path}")
            for remote in plan.deletions:
                log(f"      delete {remote}")
    except (StorageError, ValueError, OSError) as exc:
        summary.failures.append(str(exc))
        return summary

    if dry_run:
        summary.artifacts = len(selected)
        summary.files = sum(len(upload.files) for upload in selected)
        return summary
    for plan in summary.plans:
        members = groups[(plan.repo_id, plan.repo_type)]
        try:
            receipt = publisher.commit(plan, message)
            summary.receipts.append(receipt)
            summary.artifacts += len(members)
            summary.files += len(plan.files)
            summary.updated_lock = with_hf_revision(
                summary.updated_lock, [upload.artifact for upload in members],
                plan.repo_id, plan.repo_type, receipt.revision,
            )
            log(f"  published {receipt.revision}: {receipt.url}")
            if on_commit is not None:
                on_commit(receipt, summary.updated_lock)
        except (StorageError, LockError, OSError) as exc:
            summary.failures.append(str(exc))
            break
    return summary
