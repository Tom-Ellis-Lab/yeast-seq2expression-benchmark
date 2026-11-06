"""Resolve existing Hugging Face content to verified immutable input sources."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from yeastbench.data.lock import (
    LockError,
    locked_files,
    prune_sources,
    sha256_file,
    validate_artifact_lock,
    with_hf_revision,
)
from yeastbench.data.manifest import Artifact, BackendKind


def pin_inputs(
    artifacts: list[Artifact],
    lock: dict,
    revision: str = "main",
    *,
    api=None,
    download: Callable | None = None,
) -> dict:
    """Return an updated lock only after every selected artifact is verified.

For datasets, fetch the locked files at the resolved revision and compare the
bytes to the existing checksums. For runtime-loaded models, record the snapshot
identity; the model loader will use this exact snapshot. No remote writes.
    """
    from huggingface_hub import HfApi, hf_hub_download

    api = api if api is not None else HfApi()
    download = download if download is not None else hf_hub_download
    # Explicit migration may follow a renamed repository or artifact prefix.
    # Keep the expected bytes but require the new source to prove them again.
    lock = prune_sources(lock, artifacts)
    groups: dict[tuple[str, str], list] = {}
    for art in artifacts:
        validate_artifact_lock(art, lock)
        mirror = next((m for m in art.mirrors if m.backend == BackendKind.HF), None)
        if mirror is None:
            raise LockError(f"{art.id}: no Hugging Face source to pin")
        groups.setdefault((mirror.base, mirror.repo_type), []).append((art, mirror))

    updated = lock
    for (repo_id, repo_type), members in groups.items():
        try:
            resolved = api.repo_info(repo_id=repo_id, repo_type=repo_type, revision=revision).sha
            # Validate the returned SHA before downloading any files.
            candidate = with_hf_revision(
                updated, [art for art, _ in members], repo_id, repo_type, resolved,
            )
            for art, mirror in members:
                for entry in locked_files(lock, art.id):
                    cached = Path(download(
                        repo_id=repo_id, repo_type=repo_type, revision=resolved,
                        filename=mirror.remote_for(entry.relpath),
                    ))
                    if cached.stat().st_size != entry.size or sha256_file(cached) != entry.sha256:
                        raise LockError(
                            f"{art.id}/{entry.relpath}: revision {resolved} differs from lock; "
                            "choose a matching historical revision or publish the locked files"
                        )
            updated = candidate
        except LockError:
            raise
        except Exception as exc:
            raise LockError(f"could not pin {repo_id} at {revision}: {exc}") from exc
    return updated
