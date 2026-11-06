"""File identities shared by input data and saved benchmark results."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


def validate_relative_path(value: str) -> str:
    """Require a canonical relative POSIX path, without normalising aliases."""
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or any(ord(c) < 32 or ord(c) == 127 for c in value)
        or any(part in ("", ".", "..") for part in value.split("/"))
        or value.split("/", 1)[0].endswith(":")
    ):
        raise ValueError(f"not a safe relative POSIX path: {value!r}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: object) -> None:
    """Replace JSON only after the complete new document has been written."""
    path = Path(path)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class FileRecord:
    path: str
    size: int
    sha256: str

    def __post_init__(self) -> None:
        validate_relative_path(self.path)
        if isinstance(self.size, bool) or not isinstance(self.size, int) or self.size < 0:
            raise ValueError(f"invalid file size for {self.path}: {self.size!r}")
        if (
            not isinstance(self.sha256, str)
            or len(self.sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.sha256)
        ):
            raise ValueError(f"invalid SHA-256 for {self.path}")

    def to_dict(self) -> dict:
        return asdict(self)


def _local_file(root: Path, relative: str) -> Path:
    validate_relative_path(relative)
    path = root / relative
    # Symlinks make a saved tree's meaning depend on files outside the tree.
    if any((root.joinpath(*Path(relative).parts[:i])).is_symlink()
           for i in range(1, len(Path(relative).parts) + 1)):
        raise ValueError(f"symlink is not an artifact file: {relative}")
    if not path.is_file():
        raise ValueError(f"missing artifact file: {relative}")
    return path


def inventory_files(
    root: Path, paths: Iterable[str] | None = None,
) -> tuple[FileRecord, ...]:
    """Hash an explicit selection, or all files in a directory, in path order."""
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"artifact directory does not exist: {root}")
    if paths is None:
        paths = (
            p.relative_to(root).as_posix()
            for p in root.rglob("*") if p.is_file() or p.is_symlink()
        )
    relative_paths = list(paths)
    if len(set(relative_paths)) != len(relative_paths):
        raise ValueError("duplicate paths in artifact inventory")
    records = []
    for relative in sorted(relative_paths):
        path = _local_file(root, relative)
        records.append(FileRecord(relative, path.stat().st_size, sha256_file(path)))
    return tuple(records)


def validate_files(
    root: Path, records: Iterable[FileRecord], *, exact: bool = False,
) -> None:
    """Reject missing or changed files; optionally reject unlisted files too."""
    records = tuple(records)
    expected = {record.path: record for record in records}
    if len(expected) != len(records):
        raise ValueError("duplicate paths in artifact inventory")
    actual = inventory_files(root, None if exact else expected)
    if {record.path for record in actual} != set(expected):
        raise ValueError("artifact files do not match the inventory")
    for record in actual:
        if record != expected[record.path]:
            raise ValueError(f"artifact file differs from inventory: {record.path}")
