"""Resolve Hub model references before the model or tokenizer loads weights.

The input lock supplies the revision for the registered model repository.
An explicit ``hf_revision`` overrides it for experiments. Both paths resolve
to one local snapshot, so downstream loaders cannot follow a moving branch.
Legacy locks without a revision retain the original repository behaviour.
"""
from __future__ import annotations

import logging

from yeastbench.data.lock import read_lock, source_for, validate_revision
from yeastbench.data.manifest import BackendKind, Mirror

log = logging.getLogger(__name__)


def resolve_model_source(
    artifact_id: str,
    hf_repo: str,
    hf_revision: str | None = None,
    *,
    lock: dict | None = None,
    require_revision: bool = False,
) -> str:
    """Return a pinned local model snapshot, or a legacy unpinned repo ID.

    Only a full commit SHA is accepted as an explicit revision. Resolve a
    branch or tag with ``ybench data pin`` first. An override repository never
    inherits the default repository's revision. ``require_revision`` lets
    publication-oriented callers reject legacy unpinned references.
    """
    if hf_revision is not None:
        revision = validate_revision(hf_revision)
    else:
        mirror = Mirror(BackendKind.HF, base=hf_repo, repo_type="model")
        revision = source_for(
            read_lock() if lock is None else lock,
            artifact_id,
            mirror,
            require_revision=require_revision,
        ).revision

    if revision is None:
        log.warning(
            "%s: no pinned revision for %s; loading the current repository "
            "using the legacy input lock. Pin this model with `ybench data pin`.",
            artifact_id, hf_repo,
        )
        return hf_repo

    from huggingface_hub import snapshot_download

    return str(snapshot_download(repo_id=hf_repo, repo_type="model", revision=revision))
