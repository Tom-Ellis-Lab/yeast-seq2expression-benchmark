"""Download backends — one per :class:`BackendKind`.

Each backend writes a single remote file to a local path (the caller supplies a
temp ``.part`` path, verifies the checksum, then moves it into place — so
backends don't need to worry about atomicity or verification themselves).

- HTTP  — stdlib ``urllib``; public URLs, no auth. Resolves the Shorkie
          authors' public bucket today.
- HF    — ``huggingface_hub``; free, no-auth public mirror. Also warms the HF
          cache for runtime-loaded models.
- GCS   — shells out to ``gcloud storage cp`` (fallback ``gsutil cp``), reusing
          the user's existing gcloud auth.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import Protocol

from yeastbench.data.manifest import BackendKind, Mirror


class BackendError(RuntimeError):
    """A fetch failed in a way the caller can report (missing file, no auth,
    mirror not published, …)."""


class Backend(Protocol):
    kind: BackendKind

    def available(self) -> bool:
        """Whether this backend can run here (tool installed / lib importable)."""

    def fetch(self, mirror: Mirror, remote: str, out_path: Path) -> None:
        """Download ``remote`` (resolved against ``mirror``) to ``out_path``."""


def _ensure_parent(out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────
# HTTP
# ──────────────────────────────────────────────────────────────


class HttpBackend:
    kind = BackendKind.HTTP

    def available(self) -> bool:
        return True

    def _url(self, mirror: Mirror, remote: str) -> str:
        if remote.startswith("http://") or remote.startswith("https://"):
            return remote
        return f"{mirror.base}{remote}"

    def fetch(self, mirror: Mirror, remote: str, out_path: Path) -> None:
        url = self._url(mirror, remote)
        _ensure_parent(out_path)
        req = urllib.request.Request(url, headers={"User-Agent": "yeastbench-data"})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp, open(
                out_path, "wb"
            ) as fh:
                shutil.copyfileobj(resp, fh, length=1 << 20)
        except urllib.error.HTTPError as e:
            raise BackendError(f"HTTP {e.code} fetching {url}") from e
        except urllib.error.URLError as e:
            raise BackendError(f"could not reach {url}: {e.reason}") from e


# ──────────────────────────────────────────────────────────────
# HuggingFace Hub
# ──────────────────────────────────────────────────────────────


class HfBackend:
    kind = BackendKind.HF

    def available(self) -> bool:
        try:
            import huggingface_hub  # noqa: F401
        except ImportError:
            return False
        return True

    def fetch(self, mirror: Mirror, remote: str, out_path: Path) -> None:
        try:
            from huggingface_hub import hf_hub_download
            from huggingface_hub.utils import (
                EntryNotFoundError,
                RepositoryNotFoundError,
            )
        except ImportError as e:
            raise BackendError(
                "huggingface_hub not installed; run `uv sync --extra data`"
            ) from e
        _ensure_parent(out_path)
        try:
            cached = hf_hub_download(
                repo_id=mirror.base,
                filename=remote,
                repo_type=mirror.repo_type,
                revision=mirror.revision,
            )
        except RepositoryNotFoundError as e:
            raise BackendError(
                f"HF repo '{mirror.base}' not found (not published yet?)"
            ) from e
        except EntryNotFoundError as e:
            raise BackendError(
                f"'{remote}' not found in HF repo '{mirror.base}'"
            ) from e
        except Exception as e:
            raise BackendError(f"HF download failed for {mirror.base}/{remote}: {e}") from e
        try:
            shutil.copyfile(cached, out_path)
        except OSError as e:
            raise BackendError(f"cannot save HF download to {out_path}: {e}") from e

    def warm(self, mirror: Mirror) -> None:
        """Pull a whole HF repo into the local cache (runtime-loaded models)."""
        try:
            from huggingface_hub import snapshot_download
            from huggingface_hub.utils import RepositoryNotFoundError
        except ImportError as e:
            raise BackendError(
                "huggingface_hub not installed; run `uv sync --extra data`"
            ) from e
        try:
            snapshot_download(
                repo_id=mirror.base, repo_type=mirror.repo_type, revision=mirror.revision,
            )
        except RepositoryNotFoundError as e:
            raise BackendError(f"HF repo '{mirror.base}' not found") from e
        except Exception as e:
            raise BackendError(f"HF cache download failed for {mirror.base}: {e}") from e


# ──────────────────────────────────────────────────────────────
# Google Cloud Storage (via gcloud / gsutil)
# ──────────────────────────────────────────────────────────────


_BILLING_ENV = "YBENCH_GCS_BILLING_PROJECT"


class GcsBackend:
    kind = BackendKind.GCS

    def __init__(self, billing_project: str | None = None) -> None:
        self._cli = shutil.which("gcloud") or shutil.which("gsutil")
        self._billing_project = billing_project

    def available(self) -> bool:
        return self._cli is not None

    def _resolve_billing(self) -> str | None:
        """Billing project for our requester-pays bucket. Flag wins, then the
        ``YBENCH_GCS_BILLING_PROJECT`` env var. No gcloud-config fallback — we
        never silently bill whatever project gcloud happens to be pointed at."""
        return self._billing_project or os.environ.get(_BILLING_ENV) or None

    def fetch(self, mirror: Mirror, remote: str, out_path: Path) -> None:
        src = f"{mirror.base}{remote}"
        _ensure_parent(out_path)
        self._cp(src, str(out_path))

    def upload(self, mirror: Mirror, local: Path, remote: str) -> None:
        """Publish a single file to ``mirror``. Maintainer-only."""
        self._cp(str(local), f"{mirror.base}{remote}")

    def _build_cmd(self, src: str, dst: str) -> list[str]:
        """The `cp` argv, with a `--billing-project`/`-u` for the requester-pays
        bucket. Raises if no billing project is set — our only GCS mirror is
        requester pays, so a copy without one would just fail server-side."""
        if self._cli is None:
            raise BackendError(
                "neither `gcloud` nor `gsutil` found on PATH; "
                "install the Google Cloud SDK or use `--from hf`"
            )
        billing = self._resolve_billing()
        if billing is None:
            raise BackendError(
                "the GCS mirror is requester pays — set a billing project via "
                f"`--billing-project <project>` or the {_BILLING_ENV} env var "
                "(or use `--from hf`, which is free)"
            )
        if self._cli.endswith("gcloud"):
            return [self._cli, "storage", "cp", f"--billing-project={billing}", src, dst]
        return [self._cli, "-u", billing, "cp", src, dst]

    def _cp(self, src: str, dst: str) -> None:
        cmd = self._build_cmd(src, dst)
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise BackendError(
                f"`{' '.join(cmd)}` failed: {proc.stderr.strip() or proc.stdout.strip()}"
            )


# ──────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────

_BACKENDS: dict[BackendKind, Backend] = {
    BackendKind.HTTP: HttpBackend(),
    BackendKind.HF: HfBackend(),
    BackendKind.GCS: GcsBackend(),
}


def get_backend(kind: BackendKind) -> Backend:
    return _BACKENDS[kind]
