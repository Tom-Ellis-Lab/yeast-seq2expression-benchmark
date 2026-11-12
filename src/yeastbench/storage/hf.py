"""Publish a verified file snapshot in one Hugging Face commit.

The caller owns selection and completeness rules. This module owns checksums,
remote prefix boundaries, and the atomic commit. Planning never creates a repo;
the destination must already exist.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from yeastbench.storage.files import FileRecord, UploadFile, inspect_file, validate_relative_path


class StorageError(RuntimeError):
    """A snapshot cannot be planned or published."""


def validate_hf_revision(value: str) -> str:
    """Require an immutable Hub revision, rather than a branch or tag name."""
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise ValueError(
            "Hugging Face revision must be a full 40-character lowercase commit SHA; "
            f"got {value!r}"
        )
    return value


def _commit_sha(value: str) -> str:
    try:
        return validate_hf_revision(value)
    except ValueError as exc:
        raise StorageError(str(exc)) from exc


def _prefixes(values: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    try:
        prefixes = tuple(sorted(validate_relative_path(v.removesuffix("/")) + "/" for v in values))
    except (ValueError, AttributeError) as exc:
        raise StorageError(f"invalid owned prefix: {exc}") from exc
    if not prefixes:
        raise StorageError("at least one owned prefix is required")
    for i, prefix in enumerate(prefixes):
        if any(other.startswith(prefix) for other in prefixes[i + 1:]):
            raise StorageError(f"duplicate or overlapping owned prefixes: {prefix}")
    return prefixes


@dataclass(frozen=True)
class HfCommitPlan:
    """Validated uploads and scoped deletions based on one remote parent commit."""

    repo_id: str
    repo_type: str
    parent_commit: str
    files: tuple[UploadFile, ...]
    deletions: tuple[str, ...]
    owned_prefixes: tuple[str, ...]

    @property
    def snapshot_sha256(self) -> str:
        payload = {
            "files": [item.record.to_dict() for item in self.files],
            "owned_prefixes": list(self.owned_prefixes),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


@dataclass(frozen=True)
class HfPublicationReceipt:
    """The verified remote commit and identity of its selected file snapshot."""

    repo_id: str
    repo_type: str
    revision: str
    url: str
    snapshot_sha256: str

    def to_dict(self) -> dict:
        result = asdict(self)
        # Keep existing receipt files compatible while using the more precise
        # snapshot name in Python. The digest covers files and owned prefixes.
        result["manifest_sha256"] = result.pop("snapshot_sha256")
        return result


def _inspect_upload_file(path: Path, remote: str) -> FileRecord:
    try:
        return inspect_file(path, remote)
    except (OSError, ValueError) as exc:
        raise StorageError(str(exc)) from exc


def _git_blob_sha1(path: Path, size: int) -> str:
    """Identity used by the Hub for files stored directly in Git."""
    digest = hashlib.sha1(f"blob {size}\0".encode())
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class HfPublisher:
    """Plan and publish complete file snapshots within declared remote prefixes."""

    def __init__(self, api: Any = None) -> None:
        self._api = api

    @property
    def api(self):
        if self._api is None:
            try:
                from huggingface_hub import HfApi
            except ImportError as exc:
                raise StorageError("huggingface_hub is required; install the data extra") from exc
            self._api = HfApi()
        return self._api

    def plan(
        self,
        repo_id: str,
        repo_type: str,
        files: list[UploadFile],
        owned_prefixes: list[str],
    ) -> HfCommitPlan:
        """Inspect an existing repository and prepare an exact scoped snapshot.

        Every local file must match its expected size and checksum before the
        Hub is contacted. Remote file discovery uses the resolved head SHA,
        not a second lookup of ``main``. Each owned prefix must contain at
        least one submitted file.
        """
        prefixes = _prefixes(owned_prefixes)
        if not files:
            raise StorageError("refusing to publish an empty snapshot")
        uploads: dict[str, UploadFile] = {}
        for upload in files:
            remote = upload.record.path
            if remote in uploads:
                raise StorageError(f"duplicate upload path: {remote}")
            if not any(remote.startswith(prefix) for prefix in prefixes):
                raise StorageError(f"upload path is outside owned prefixes: {remote}")
            local = Path(upload.local_path).absolute()
            if _inspect_upload_file(local, remote) != upload.record:
                raise StorageError(f"local file differs from expected file record: {remote}")
            uploads[remote] = UploadFile(local, upload.record)
        if any(not any(path.startswith(prefix) for path in uploads) for prefix in prefixes):
            raise StorageError("each owned prefix must contain at least one uploaded file")
        try:
            head = _commit_sha(self.api.repo_info(
                repo_id=repo_id, repo_type=repo_type, revision="main",
            ).sha)
            remote_paths = self.api.list_repo_files(
                repo_id=repo_id, repo_type=repo_type, revision=head,
            )
            deletions = tuple(sorted(
                validate_relative_path(path)
                for path in remote_paths
                if any(path.startswith(prefix) for prefix in prefixes) and path not in uploads
            ))
        except StorageError:
            raise
        except Exception as exc:
            raise StorageError(
                f"cannot inspect Hugging Face repository '{repo_id}'; "
                f"it must already exist and be accessible: {exc}"
            ) from exc
        return HfCommitPlan(
            repo_id, repo_type, head,
            tuple(uploads[path] for path in sorted(uploads)), deletions, prefixes,
        )

    def _verify_snapshot(
        self, plan: HfCommitPlan, revision: str, git_blobs: dict[str, str],
    ) -> None:
        """Verify the exact returned revision, including no-op commit responses.

        The Hub client can skip an unchanged commit and return the current
        head without enforcing ``parent_commit``. Checking the returned tree
        prevents recording that head as a pin if a concurrent change altered
        the selected files. Metadata provides content identities without
        downloading the uploaded payload a second time.
        """
        info = self.api.repo_info(
            repo_id=plan.repo_id, repo_type=plan.repo_type,
            revision=revision, files_metadata=True,
        )
        if info.sha != revision:
            raise StorageError("repository metadata did not resolve to the requested commit")
        actual = {}
        for sibling in info.siblings:
            path = sibling.rfilename
            if any(path.startswith(prefix) for prefix in plan.owned_prefixes):
                if path in actual:
                    raise StorageError(f"duplicate remote file metadata: {path}")
                actual[path] = sibling
        expected = {item.record.path: item.record for item in plan.files}
        if set(actual) != set(expected):
            missing = sorted(set(expected) - set(actual))
            extra = sorted(set(actual) - set(expected))
            raise StorageError(f"remote snapshot has missing files {missing} or unexpected files {extra}")
        for path, record in expected.items():
            remote = actual[path]
            if remote.size != record.size:
                raise StorageError(f"remote file size differs: {path}")
            if remote.lfs is not None:
                lfs_sha = (
                    remote.lfs.get("sha256") if isinstance(remote.lfs, Mapping)
                    else remote.lfs.sha256
                )
                if lfs_sha != record.sha256:
                    raise StorageError(f"remote LFS checksum differs: {path}")
            elif remote.blob_id != git_blobs[path]:
                raise StorageError(f"remote Git blob checksum differs: {path}")

    def commit(self, plan: HfCommitPlan, message: str) -> HfPublicationReceipt:
        """Freeze validated bytes and submit additions and deletions together."""
        if not plan.files:
            raise StorageError("refusing to publish an empty snapshot")
        prefixes = _prefixes(plan.owned_prefixes)
        _commit_sha(plan.parent_commit)
        for item in plan.files:
            if not any(item.record.path.startswith(prefix) for prefix in prefixes):
                raise StorageError(f"upload path is outside owned prefixes: {item.record.path}")
            if _inspect_upload_file(item.local_path, item.record.path) != item.record:
                raise StorageError(f"file changed after planning: {item.record.path}")
        for path in plan.deletions:
            validate_relative_path(path)
            if not any(path.startswith(prefix) for prefix in prefixes):
                raise StorageError(f"deletion is outside owned prefixes: {path}")
        try:
            from huggingface_hub import CommitOperationAdd, CommitOperationDelete

            # Hub clients read file bodies after constructing commit operations.
            # Stage private copies so edits during an upload cannot change the
            # bytes whose identities the receipt records.
            with tempfile.TemporaryDirectory(prefix="ybench-publish-") as staging:
                operations = []
                git_blobs = {}
                for i, item in enumerate(plan.files):
                    staged = Path(staging) / str(i)
                    shutil.copyfile(item.local_path, staged)
                    if _inspect_upload_file(staged, item.record.path) != item.record:
                        raise StorageError(f"file changed while staging: {item.record.path}")
                    git_blobs[item.record.path] = _git_blob_sha1(staged, item.record.size)
                    operations.append(CommitOperationAdd(
                        path_in_repo=item.record.path, path_or_fileobj=str(staged),
                    ))
                operations.extend(CommitOperationDelete(path_in_repo=path) for path in plan.deletions)
                result = self.api.create_commit(
                    repo_id=plan.repo_id,
                    repo_type=plan.repo_type,
                    revision="main",
                    parent_commit=plan.parent_commit,
                    operations=operations,
                    commit_message=message,
                )
                revision = _commit_sha(result.oid)
                try:
                    self._verify_snapshot(plan, revision, git_blobs)
                except Exception as exc:
                    raise StorageError(
                        f"Hugging Face returned commit {revision} ({result.commit_url}), "
                        f"but snapshot verification failed; no verified receipt was issued: {exc}"
                    ) from exc
            return HfPublicationReceipt(
                plan.repo_id, plan.repo_type, revision, str(result.commit_url),
                plan.snapshot_sha256,
            )
        except StorageError:
            raise
        except Exception as exc:
            raise StorageError(f"cannot publish Hugging Face snapshot: {exc}") from exc
