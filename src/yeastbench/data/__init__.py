"""Registered benchmark inputs, version locks, and data workflows.

Artifacts (task data, model weights, shared references) are declared in
``manifest.py`` and frozen to a checksum lock (``manifest.lock.json``). The
``ybench data`` CLI fetches them from interchangeable backends (HF / GCS /
HTTP), verifying every file against the lock.
"""

from yeastbench.data.artifacts import (
    BackendKind,
    InputArtifact,
    InputKind,
    License,
    Mirror,
)
from yeastbench.data.manifest import (
    ARTIFACTS,
    artifact_by_id,
    artifacts_for,
)

__all__ = [
    "ARTIFACTS",
    "BackendKind",
    "InputArtifact",
    "InputKind",
    "License",
    "Mirror",
    "artifact_by_id",
    "artifacts_for",
]
