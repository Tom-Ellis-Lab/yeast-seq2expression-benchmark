"""Types for declared benchmark inputs and their download sources.

``InputArtifact`` describes a dataset, model, or shared reference needed by a
benchmark. Instances are registered in ``data.manifest``. Byte-level file
records and publication plans live in ``storage`` and do not depend on this
registry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping

# ──────────────────────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────────────────────


class InputKind(str, Enum):
    """The role of a registered benchmark input."""

    TASK_PROCESSED = "task-processed"
    MODEL_WEIGHTS = "model-weights"
    SHARED_REFERENCE = "shared-reference"


class BackendKind(str, Enum):
    HF = "hf"
    GCS = "gcs"
    HTTP = "http"


class License(str, Enum):
    OWN = "own"                # our data — mirror freely
    CC_BY_4_0 = "cc-by-4.0"    # mirror freely with attribution
    CC_BY_NC_4_0 = "cc-by-nc-4.0"  # mirror with attribution, non-commercial use
    PUBLIC = "public"          # publicly hosted by upstream; fetch from origin
    GEO_TERMS = "geo-terms"    # NCBI GEO — do not re-host
    UNKNOWN = "unknown"


# ──────────────────────────────────────────────────────────────
# Dataclasses
# ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Mirror:
    """One source for a declared input's files.

    The remote path of a file with local relative path ``rel`` is
    ``member_remote[rel]`` if present, else ``prefix + rel``. A
    ``member_remote`` value that starts with ``http`` is treated as an absolute
    URL (used for source mirrors with an irregular layout).
    """

    backend: BackendKind
    base: str = ""                     # HF repo id, or gs:// / https:// base (trailing slash)
    prefix: str = ""                   # remote dir prepended to each file's relpath
    repo_type: str = "dataset"         # HF only: "dataset" | "model"
    member_remote: Mapping[str, str] = field(default_factory=dict)
    revision: str | None = None        # HF full commit SHA, resolved from the input lock

    def remote_for(self, rel: str) -> str:
        if rel in self.member_remote:
            return self.member_remote[rel]
        return f"{self.prefix}{rel}"


@dataclass(frozen=True)
class InputArtifact:
    """A declared input: its role, required files, dependencies, and sources.

    File sizes and checksums belong to the input lock and use the shared
    ``storage.files.FileRecord`` representation.
    """

    id: str
    kind: InputKind
    dest: str                          # dir relative to the data root
    mirrors: tuple[Mirror, ...]
    needed_by: tuple[str, ...] = ()     # registry task/model names
    requires: tuple[str, ...] = ()      # other artifact ids (e.g. shared refs)
    include: tuple[str, ...] = ("**/*",)
    # Keep the published distribution to runtime files: drop the raw-download
    # cache, build logs, and raw source formats (spreadsheets, GenBank) that the
    # benchmarks don't read at run time.
    exclude: tuple[str, ...] = (
        "**/_cache/**", "*.log", "*.xlsx", "*.xls", "*.gb", "*.gbk",
    )
    license: License = License.UNKNOWN
    redistributable: bool = False
    cache_only: bool = False           # HF-hub model loaded at runtime; warm cache, nothing under dest
    raw: object | None = None          # reserved for future raw-data preparation metadata
