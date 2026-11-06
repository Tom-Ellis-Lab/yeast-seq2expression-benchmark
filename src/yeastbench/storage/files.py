"""File identities and inventories shared by input data and saved results."""

from __future__ import annotations

import hashlib
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


@dataclass(frozen=True)
class FileRecord:
    """A file's relative path and content identity within an inventory."""

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


@dataclass(frozen=True)
class UploadFile:
    """A local file paired with its destination-relative path and identity."""

    local_path: Path
    record: FileRecord


def inspect_file(local_path: Path, relative_path: str) -> FileRecord:
    """Record a regular file, rejecting changes made while its bytes are hashed.

    ``relative_path`` identifies the file within its dataset or upload snapshot;
    ``local_path`` is the actual file to read. Directory inventories additionally
    reject symlinks in the path between their root and this file.
    """
    validate_relative_path(relative_path)
    path = Path(local_path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"missing or symlinked file: {path}")
    before = path.stat()
    digest = sha256_file(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (
        after.st_size, after.st_mtime_ns, after.st_ino,
    ):
        raise ValueError(f"file changed while hashing: {path}")
    return FileRecord(relative_path, after.st_size, digest)


def _local_file(root: Path, relative: str) -> Path:
    validate_relative_path(relative)
    path = root / relative
    # Symlinks make a saved tree's meaning depend on files outside the tree.
    if any((root.joinpath(*Path(relative).parts[:i])).is_symlink()
           for i in range(1, len(Path(relative).parts) + 1)):
        raise ValueError(f"symlink is not an inventory file: {relative}")
    if not path.is_file():
        raise ValueError(f"missing inventory file: {relative}")
    return path


def inventory_files(
    root: Path, paths: Iterable[str] | None = None,
) -> tuple[FileRecord, ...]:
    """Hash an explicit selection, or all files in a directory, in path order."""
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"inventory directory does not exist: {root}")
    if paths is None:
        paths = (
            p.relative_to(root).as_posix()
            for p in root.rglob("*") if p.is_file() or p.is_symlink()
        )
    relative_paths = list(paths)
    if len(set(relative_paths)) != len(relative_paths):
        raise ValueError("duplicate paths in file inventory")
    records = []
    for relative in sorted(relative_paths):
        path = _local_file(root, relative)
        records.append(inspect_file(path, relative))
    return tuple(records)


def validate_files(
    root: Path, records: Iterable[FileRecord], *, exact: bool = False,
) -> None:
    """Reject missing or changed files; optionally reject unlisted files too."""
    records = tuple(records)
    expected = {record.path: record for record in records}
    if len(expected) != len(records):
        raise ValueError("duplicate paths in file inventory")
    actual = inventory_files(root, None if exact else expected)
    if {record.path for record in actual} != set(expected):
        raise ValueError("files do not match the inventory")
    for record in actual:
        if record != expected[record.path]:
            raise ValueError(f"file differs from inventory: {record.path}")
