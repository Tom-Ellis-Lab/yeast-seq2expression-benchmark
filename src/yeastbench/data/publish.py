"""Publish selected input artifacts using the shared snapshot publisher.

Validate the whole selection before any remote writes. Hugging Face receives
one commit per repository; GCS remains a per-file mirror without deletions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from yeastbench.data.backends import BackendError, GcsBackend
from yeastbench.data.lock import (
    LockError,
    locked_files,
    prune_sources,
    sha256_file,
    validate_artifact_lock,
    with_hf_revision,
)
from yeastbench.data.manifest import Artifact, BackendKind, Mirror
from yeastbench.storage.hf import HfStore, PublicationReceipt, SnapshotPlan, StorageError


@dataclass
class PublishSummary:
    artifacts: int = 0
    files: int = 0
    skipped: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    plans: list[SnapshotPlan] = field(default_factory=list)
    receipts: list[PublicationReceipt] = field(default_factory=list)
    updated_lock: dict | None = None

    @property
    def ok(self) -> bool:
        return not self.failures


def run_publish(
    artifacts: list[Artifact],
    lock: dict,
    to_kind: BackendKind,
    data_root: Path,
    message: str,
    dry_run: bool,
    log: Callable[[str], None] = lambda _: None,
    billing_project: str | None = None,
    *,
    store: HfStore | None = None,
    on_commit: Callable[[PublicationReceipt, dict], None] | None = None,
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

    selected: list[tuple[Artifact, Mirror, list[tuple[Path, str]]]] = []
    checksums: dict[tuple[str, str], dict[str, str]] = {}
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
            items = []
            hashes = checksums.setdefault((mirror.base, mirror.repo_type), {})
            for entry in locked_files(lock, art.id):
                local = data_root / art.dest / entry.relpath
                if not local.is_file():
                    raise ValueError(f"missing locally: {local}")
                if local.stat().st_size != entry.size or sha256_file(local) != entry.sha256:
                    raise ValueError(f"local differs from lock: {local}; rebuild and run data lock")
                remote = mirror.remote_for(entry.relpath)
                if remote in hashes:
                    raise ValueError(f"multiple artifacts own remote path {remote}")
                hashes[remote] = entry.sha256
                items.append((local, remote))
            selected.append((art, mirror, items))
        except (LockError, ValueError, OSError) as exc:
            summary.failures.append(f"{art.id}: {exc}")
    if summary.failures:
        return summary

    if to_kind == BackendKind.GCS:
        backend = GcsBackend(billing_project)
        for art, mirror, items in selected:
            log(f"  {art.id} → {mirror.base}{mirror.prefix} ({len(items)} files; copy only)")
            try:
                for local, remote in items:
                    log(f"      copy {local} → {mirror.base}{remote}")
                    if not dry_run:
                        backend.upload(mirror, local, remote)
                summary.artifacts += 1
                summary.files += len(items)
            except BackendError as exc:
                summary.failures.append(f"{art.id}: {exc}")
                break
        return summary

    if not selected:
        return summary
    groups: dict[tuple[str, str], list] = {}
    for art, mirror, items in selected:
        groups.setdefault((mirror.base, mirror.repo_type), []).append((art, mirror, items))
    try:
        hub = store if store is not None else HfStore()
        # Plan every repository before making the first remote write.
        for (repo_id, repo_type), members in groups.items():
            plan = hub.plan(
                repo_id=repo_id,
                repo_type=repo_type,
                items=[item for _, _, items in members for item in items],
                owned_prefixes=[mirror.prefix for _, mirror, _ in members],
                expected_checksums=checksums[(repo_id, repo_type)],
            )
            summary.plans.append(plan)
            log(f"  {repo_type}:{repo_id} at {plan.parent_commit}")
            for local, remote in plan.items:
                log(f"      add/update {local} → {remote}")
            for remote in plan.deletions:
                log(f"      delete {remote}")
    except (StorageError, ValueError, OSError) as exc:
        summary.failures.append(str(exc))
        return summary

    if dry_run:
        summary.artifacts = len(selected)
        summary.files = sum(len(items) for _, _, items in selected)
        return summary
    for plan in summary.plans:
        members = groups[(plan.repo_id, plan.repo_type)]
        try:
            receipt = hub.commit(plan, message)
            summary.receipts.append(receipt)
            summary.artifacts += len(members)
            summary.files += len(plan.items)
            summary.updated_lock = with_hf_revision(
                summary.updated_lock, [art for art, _, _ in members],
                plan.repo_id, plan.repo_type, receipt.revision,
            )
            log(f"  published {receipt.revision}: {receipt.url}")
            if on_commit is not None:
                on_commit(receipt, summary.updated_lock)
        except (StorageError, LockError, OSError) as exc:
            summary.failures.append(str(exc))
            break
    return summary
