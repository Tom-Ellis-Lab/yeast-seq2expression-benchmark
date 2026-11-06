"""Checksums and immutable download sources for benchmark inputs.

Version 2 adds Hugging Face commit revisions to the version 1 file inventory.
Reading an old lock never invents a revision: its bytes remain verifiable, but
its Hub sources must be pinned before it can support a publishable run.
"""

from __future__ import annotations

import copy
import fnmatch
import json
import re
from dataclasses import dataclass, replace
from pathlib import Path

from yeastbench.artifacts import atomic_write_json, sha256_file, validate_relative_path
from yeastbench.data.manifest import Artifact, BackendKind, Mirror

LOCK_VERSION = 2
LOCK_PATH = Path(__file__).with_name("manifest.lock.json")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_REVISION = re.compile(r"[0-9a-f]{40}\Z")


class LockError(ValueError):
    """An input lock is malformed, incomplete, or inconsistent with its artifacts."""


def _relative(value: str, context: str) -> str:
    try:
        return validate_relative_path(value)
    except (TypeError, ValueError) as exc:
        raise LockError(f"{context}: invalid relative path {value!r}") from exc


def validate_revision(revision: str) -> str:
    """Require a full immutable Hub commit SHA, rather than a moving branch."""
    if not isinstance(revision, str) or _REVISION.fullmatch(revision) is None:
        raise LockError("Hugging Face revision must be a full 40-character lowercase commit SHA")
    return revision


def _source_key(source: dict) -> tuple[str, str, str, str]:
    return (source["backend"], source["repo_id"], source["repo_type"], source["prefix"])


def _validate_lock(lock: dict) -> None:
    if not isinstance(lock, dict):
        raise LockError("input lock must be a JSON object")
    version = lock.get("version")
    if type(version) is not int or version not in (1, LOCK_VERSION):
        raise LockError(f"unsupported input lock version: {version!r}")
    artifacts = lock.get("artifacts")
    if not isinstance(artifacts, dict):
        raise LockError("input lock must contain an 'artifacts' object")
    for aid, entry in artifacts.items():
        if not isinstance(aid, str) or not aid:
            raise LockError("artifact IDs must be nonempty strings")
        if not isinstance(entry, dict) or not isinstance(entry.get("files"), dict):
            raise LockError(f"{aid}: lock entry must contain a 'files' object")
        for rel, meta in entry["files"].items():
            _relative(rel, aid)
            if not isinstance(meta, dict):
                raise LockError(f"{aid}/{rel}: file record must be an object")
            checksum = meta.get("sha256")
            if not isinstance(checksum, str) or _SHA256.fullmatch(checksum) is None:
                raise LockError(f"{aid}/{rel}: invalid SHA-256 checksum")
            if type(meta.get("size")) is not int or meta["size"] < 0:
                raise LockError(f"{aid}/{rel}: size must be a nonnegative integer")
        sources = entry.get("sources", [])
        if not isinstance(sources, list):
            raise LockError(f"{aid}: 'sources' must be a list")
        seen = set()
        for source in sources:
            if not isinstance(source, dict) or source.get("backend") != "hf":
                raise LockError(f"{aid}: unsupported source record (expected backend 'hf')")
            repo_id = source.get("repo_id")
            if not isinstance(repo_id, str) or not repo_id:
                raise LockError(f"{aid}: source must name a Hugging Face repository")
            _relative(repo_id, f"{aid} source repository")
            if source.get("repo_type") not in ("dataset", "model"):
                raise LockError(f"{aid}: source repo_type must be 'dataset' or 'model'")
            prefix = source.get("prefix")
            if not isinstance(prefix, str):
                raise LockError(f"{aid}: source must contain a string prefix")
            if prefix:
                if not prefix.endswith("/"):
                    raise LockError(f"{aid}: source prefix must be empty or end with '/'")
                _relative(prefix[:-1], f"{aid} source prefix")
            validate_revision(source.get("revision"))
            key = _source_key(source)
            if key in seen:
                raise LockError(f"{aid}: duplicate source for {repo_id}/{prefix}")
            seen.add(key)


def _matches(relpath: str, pattern: str) -> bool:
    """Match Path.glob semantics, including zero directories for ``**/``."""
    parts, patterns = relpath.split("/"), pattern.split("/")

    def match(i: int, j: int) -> bool:
        if j == len(patterns):
            return i == len(parts)
        if patterns[j] == "**":
            return match(i, j + 1) or (i < len(parts) and match(i + 1, j))
        return (
            i < len(parts)
            and fnmatch.fnmatchcase(parts[i], patterns[j])
            and match(i + 1, j + 1)
        )

    return match(0, 0)


def _excluded(relpath: str, patterns: tuple[str, ...]) -> bool:
    parts = Path(relpath).parts
    for pat in patterns:
        token = pat.strip("*/")
        if token and token in parts:
            return True
        if fnmatch.fnmatchcase(relpath, pat):
            return True
    return False


def _validate_inventory(artifact: Artifact, files: dict) -> None:
    if not files:
        raise LockError(f"{artifact.id}: no locked files; build the data and run `ybench data lock`")
    for pattern in artifact.include:
        _relative(pattern, f"{artifact.id} include pattern")
        if not any(_matches(rel, pattern) for rel in files):
            raise LockError(f"{artifact.id}: required file or pattern {pattern!r} has no files")
    for rel in files:
        if _excluded(rel, artifact.exclude) or not any(
            _matches(rel, pattern) for pattern in artifact.include
        ):
            raise LockError(f"{artifact.id}/{rel}: locked file is outside the declared include list")


def local_files(artifact: Artifact, data_root: Path) -> list[str]:
    """Find included local files, sorted; cache-only artifacts return no files."""
    if artifact.cache_only or not artifact.dest:
        return []
    _relative(artifact.dest, f"{artifact.id} destination")
    base = data_root / artifact.dest
    if not base.exists():
        return []
    found: set[str] = set()
    for pattern in artifact.include:
        _relative(pattern, f"{artifact.id} include pattern")
        for path in base.glob(pattern):
            if not path.is_file():
                continue
            rel = path.relative_to(base).as_posix()
            _relative(rel, artifact.id)
            if _excluded(rel, artifact.exclude):
                continue
            if not path.resolve().is_relative_to(base.resolve()):
                raise LockError(f"{artifact.id}/{rel}: file resolves outside its artifact directory")
            found.add(rel)
    return sorted(found)


def build_lock(artifacts: list[Artifact], data_root: Path) -> dict[str, dict]:
    """Hash a complete local selection; fail rather than retain stale entries."""
    out: dict[str, dict] = {}
    for art in artifacts:
        if art.cache_only:
            continue
        files = {}
        for rel in local_files(art, data_root):
            path = data_root / art.dest / rel
            files[rel] = {"sha256": sha256_file(path), "size": path.stat().st_size}
        _validate_inventory(art, files)
        out[art.id] = {"files": files}
    return out


def merge_lock(existing: dict, fresh: dict[str, dict]) -> dict:
    """Replace selected inventories; retain old source pins only for equal bytes."""
    _validate_lock(existing)
    _validate_lock({"version": LOCK_VERSION, "artifacts": fresh})
    artifacts = copy.deepcopy(existing["artifacts"])
    for aid, entry in fresh.items():
        updated = copy.deepcopy(entry)
        old = artifacts.get(aid, {})
        if "sources" not in updated and old.get("files") == updated["files"]:
            if "sources" in old:
                updated["sources"] = old["sources"]
        artifacts[aid] = updated
    return {"version": LOCK_VERSION, "artifacts": artifacts}


def _object_without_duplicates(pairs: list[tuple[str, object]]) -> dict:
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise LockError(f"duplicate JSON key in input lock: {key!r}")
        obj[key] = value
    return obj


def read_lock(path: Path = LOCK_PATH) -> dict:
    if not path.exists():
        return {"version": LOCK_VERSION, "artifacts": {}}
    try:
        lock = json.loads(path.read_text(), object_pairs_hook=_object_without_duplicates)
    except json.JSONDecodeError as exc:
        raise LockError(f"invalid input lock JSON in {path}: {exc.msg}") from exc
    _validate_lock(lock)
    return lock


def write_lock(lock: dict, path: Path = LOCK_PATH) -> None:
    _validate_lock(lock)
    atomic_write_json(path, lock)


@dataclass(frozen=True)
class LockedFile:
    relpath: str
    sha256: str
    size: int


def locked_files(lock: dict, artifact_id: str) -> list[LockedFile]:
    _validate_lock(lock)
    entry = lock["artifacts"].get(artifact_id)
    if not entry:
        return []
    return [
        LockedFile(rel, meta["sha256"], meta["size"])
        for rel, meta in sorted(entry["files"].items())
    ]


def source_for(
    lock: dict, artifact_id: str, mirror: Mirror, require_revision: bool = False,
) -> Mirror:
    """Apply the pin for exactly this repository, type, and artifact prefix."""
    _validate_lock(lock)
    if mirror.backend != BackendKind.HF:
        return mirror
    if mirror.revision is not None:
        validate_revision(mirror.revision)
    wanted = ("hf", mirror.base, mirror.repo_type, mirror.prefix)
    entry = lock["artifacts"].get(artifact_id, {})
    for source in entry.get("sources", []):
        if _source_key(source) == wanted:
            revision = source["revision"]
            if mirror.revision is not None and mirror.revision != revision:
                raise LockError(f"{artifact_id}: configured revision differs from the input lock")
            return replace(mirror, revision=revision)
    if mirror.revision is not None:
        return mirror
    if require_revision:
        raise LockError(
            f"{artifact_id}: no pinned Hugging Face revision for "
            f"{mirror.base}/{mirror.prefix}; pin or publish this artifact first"
        )
    return mirror


def validate_artifact_lock(
    artifact: Artifact, lock: dict, require_pinned: bool = False,
) -> None:
    """Validate declared coverage and, optionally, an immutable Hub source."""
    _validate_lock(lock)
    entry = lock["artifacts"].get(artifact.id)
    if not artifact.cache_only:
        _validate_inventory(artifact, entry["files"] if entry else {})
    elif entry and entry["files"]:
        raise LockError(f"{artifact.id}: cache-only model must not declare local artifact files")
    declared_sources = {
        ("hf", mirror.base, mirror.repo_type, mirror.prefix)
        for mirror in artifact.mirrors if mirror.backend == BackendKind.HF
    }
    for source in (entry or {}).get("sources", []):
        if _source_key(source) not in declared_sources:
            raise LockError(f"{artifact.id}: locked source is not a declared mirror")
    if require_pinned:
        hf_mirrors = [m for m in artifact.mirrors if m.backend == BackendKind.HF]
        if artifact.cache_only and not hf_mirrors:
            raise LockError(f"{artifact.id}: cache-only model has no Hugging Face mirror")
        if hf_mirrors and not any(source_for(lock, artifact.id, m).revision for m in hf_mirrors):
            raise LockError(f"{artifact.id}: no pinned Hugging Face revision; pin or publish first")


def prune_sources(lock: dict, artifacts: list[Artifact]) -> dict:
    """Drop obsolete mirror pins during an explicit lock/pin/publish operation.

    Ordinary downloads still reject declaration drift. An explicit maintenance
    operation can replace a renamed repository or prefix without editing JSON.
    The input lock and all unselected artifact entries remain unchanged.
    """
    _validate_lock(lock)
    updated = copy.deepcopy(lock)
    for art in artifacts:
        entry = updated["artifacts"].get(art.id)
        if entry is None or "sources" not in entry:
            continue
        declared = {
            ("hf", mirror.base, mirror.repo_type, mirror.prefix)
            for mirror in art.mirrors if mirror.backend == BackendKind.HF
        }
        remaining = [source for source in entry["sources"] if _source_key(source) in declared]
        if remaining:
            entry["sources"] = remaining
        else:
            entry.pop("sources")
    return updated


def with_hf_revision(
    lock: dict, artifacts: list[Artifact], repo_id: str, repo_type: str, revision: str,
) -> dict:
    """Return a new lock recording a verified/published revision for a selection."""
    validate_revision(revision)
    updated = prune_sources(lock, artifacts)
    updated["version"] = LOCK_VERSION
    for art in artifacts:
        matching = [
            mirror for mirror in art.mirrors
            if mirror.backend == BackendKind.HF
            and mirror.base == repo_id and mirror.repo_type == repo_type
        ]
        if not matching:
            raise LockError(f"{art.id}: no declared {repo_type} mirror for {repo_id}")
        validate_artifact_lock(art, updated)
        entry = updated["artifacts"].setdefault(art.id, {"files": {}})
        sources = entry.setdefault("sources", [])
        for mirror in matching:
            record = {
                "backend": "hf", "repo_id": repo_id, "repo_type": repo_type,
                "revision": revision, "prefix": mirror.prefix,
            }
            sources[:] = [s for s in sources if _source_key(s) != _source_key(record)]
            sources.append(record)
        sources.sort(key=_source_key)
    _validate_lock(updated)
    return updated


def artifact_ids_in_lock(lock: dict) -> set[str]:
    _validate_lock(lock)
    return set(lock["artifacts"])
