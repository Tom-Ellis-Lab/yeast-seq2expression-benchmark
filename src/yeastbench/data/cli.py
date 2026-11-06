"""``ybench data`` — download, verify, and lock benchmark data & model weights.

Mounted onto the top-level ``ybench`` app in ``yeastbench.cli``.
"""

from __future__ import annotations

import json as _json
from pathlib import Path
from typing import Annotated, Optional

import typer

from yeastbench.data.artifacts import BackendKind
from yeastbench.data.fetch import (
    DownloadPlan,
    GetSummary,
    build_plans,
    default_data_root,
    format_bytes,
    run_get,
    run_status,
    run_verify,
    selection_from_config,
)
from yeastbench.data.lock import (
    LOCK_PATH,
    LockError,
    build_lock_entries,
    merge_lock,
    prune_sources,
    read_lock,
    write_lock,
)
from yeastbench.data.manifest import ARTIFACTS, artifact_by_id, artifacts_for
from yeastbench.data.pinning import pin_inputs
from yeastbench.data.publish import run_publish
from yeastbench.storage.json import atomic_write_json

app = typer.Typer(
    add_completion=False,
    help="Download and verify benchmark data & model weights.",
)


def _echo(msg: str) -> None:
    typer.echo(msg)


def _load_lock(path: Path) -> dict:
    try:
        return read_lock(path)
    except (LockError, OSError) as exc:
        _echo(f"invalid lock: {exc}")
        raise typer.Exit(code=1) from exc


def _split(csv: str | None) -> list[str] | None:
    if not csv:
        return None
    return [s.strip() for s in csv.split(",") if s.strip()]


def _backend_kind(value: str | None) -> BackendKind | None:
    if value is None:
        return None
    try:
        return BackendKind(value)
    except ValueError:
        raise typer.Exit(
            f"Unknown backend '{value}'. Choose from: "
            f"{', '.join(b.value for b in BackendKind)}"
        )


def _select(
    config: Path | None,
    task: str | None,
    model: str | None,
    tasks: str | None,
    models: str | None,
):
    """Resolve the artifact selection from CLI options. Precedence:
    --config (+ --task/--model filters) > explicit --tasks/--models > all."""
    if config is not None:
        sel_tasks, sel_models = selection_from_config(config, model, task)
        return artifacts_for(sel_tasks, sel_models)
    t, m = _split(tasks), _split(models)
    if t is None and m is None:
        return artifacts_for(None, None)
    return artifacts_for(t, m)


# ──────────────────────────────────────────────────────────────
# get
# ──────────────────────────────────────────────────────────────


@app.command("get")
def get_cmd(
    config: Annotated[Optional[Path], typer.Option("--config", "-c", help="Pull data for the (model, task) pairs in this run config")] = None,
    task: Annotated[Optional[str], typer.Option("--task", "-t", help="With --config: only this task")] = None,
    model: Annotated[Optional[str], typer.Option("--model", "-m", help="With --config: only this model")] = None,
    tasks: Annotated[Optional[str], typer.Option("--tasks", help="Comma-separated task names (no config)")] = None,
    models: Annotated[Optional[str], typer.Option("--models", help="Comma-separated model names (no config)")] = None,
    from_: Annotated[Optional[str], typer.Option("--from", help="Force a backend: hf | gcs | http")] = None,
    billing_project: Annotated[Optional[str], typer.Option("--billing-project", help="GCP project to bill for the requester-pays GCS bucket (or set YBENCH_GCS_BILLING_PROJECT); not needed for the HF default")] = None,
    data_root: Annotated[Optional[Path], typer.Option("--data-root", help="Where data/ lives (default: repo root)")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Print the plan and exit")] = False,
    lock_file: Annotated[Path, typer.Option("--lock-file", help="Input checksum/revision lock")] = LOCK_PATH,
    require_pinned: Annotated[bool, typer.Option("--require-pinned", help="Reject HF sources without immutable revisions")] = False,
) -> None:
    """Download benchmark data and model weights, verifying every file."""
    root = (data_root or default_data_root()).resolve()
    artifacts = _select(config, task, model, tasks, models)
    lock = _load_lock(lock_file)
    plans = build_plans(artifacts, lock, _backend_kind(from_), root, require_pinned=require_pinned)

    _echo(f"data root: {root}")
    _echo(f"artifacts: {len(plans)}")
    _print_plan(plans)

    if dry_run:
        raise typer.Exit(code=1 if any(p.note for p in plans) else 0)

    _echo("")
    summary = run_get(plans, log=_echo, billing_project=billing_project)
    _print_get_summary(summary)
    if not summary.ok:
        raise typer.Exit(code=1)


def _print_plan(plans: list[DownloadPlan]) -> None:
    for p in plans:
        mirror = (
            f"{p.mirror.backend.value}:{p.mirror.base}" if p.mirror else "—"
        )
        if p.mirror and p.mirror.backend == BackendKind.HF:
            mirror += f"@{p.mirror.revision or 'unpinned'}"
        avail = "" if (p.mirror_available or p.cache_only and p.mirror_available) else " (unavailable)"
        if p.cache_only:
            _echo(f"  {p.artifact.id}  [hf-cache]  {mirror}{avail}")
            if p.note:
                _echo(f"      note: {p.note}")
            continue
        n_fetch = len(p.to_fetch)
        n_ok = len(p.files) - n_fetch
        size = format_bytes(sum(f.record.size for f in p.to_fetch))
        _echo(
            f"  {p.artifact.id}  via {mirror}{avail}  "
            f"[{n_ok} ok, {n_fetch} to fetch, {size}]"
        )
        if p.note:
            _echo(f"      note: {p.note}")


def _print_get_summary(s: GetSummary) -> None:
    _echo("")
    _echo(
        f"fetched {s.fetched}, up-to-date {s.skipped_ok}, "
        f"hf-cache warmed {s.warmed}"
    )
    if s.skipped:
        _echo(f"skipped (mirror not reachable yet): {len(s.skipped)}")
        for m in s.skipped:
            _echo(f"  · {m}")
    if s.failures:
        _echo(f"FAILURES: {len(s.failures)}")
        for m in s.failures:
            _echo(f"  ! {m}")


# ──────────────────────────────────────────────────────────────
# verify
# ──────────────────────────────────────────────────────────────


@app.command("verify")
def verify_cmd(
    config: Annotated[Optional[Path], typer.Option("--config", "-c")] = None,
    task: Annotated[Optional[str], typer.Option("--task", "-t")] = None,
    model: Annotated[Optional[str], typer.Option("--model", "-m")] = None,
    tasks: Annotated[Optional[str], typer.Option("--tasks")] = None,
    models: Annotated[Optional[str], typer.Option("--models")] = None,
    data_root: Annotated[Optional[Path], typer.Option("--data-root")] = None,
    lock_file: Annotated[Path, typer.Option("--lock-file")] = LOCK_PATH,
    require_pinned: Annotated[bool, typer.Option("--require-pinned")] = False,
) -> None:
    """Checksum local files against the lock. Exits non-zero on any mismatch."""
    root = (data_root or default_data_root()).resolve()
    artifacts = _select(config, task, model, tasks, models)
    rows = run_verify(artifacts, _load_lock(lock_file), root, require_pinned=require_pinned)

    bad = [r for r in rows if r.status != "ok"]
    ok = len(rows) - len(bad)
    _echo(f"verified {ok}/{len(rows)} files ok")
    for r in bad:
        _echo(f"  ! {r.artifact_id}/{r.relpath}: {r.status}")
    if bad:
        raise typer.Exit(code=1)


# ──────────────────────────────────────────────────────────────
# status
# ──────────────────────────────────────────────────────────────


@app.command("status")
def status_cmd(
    config: Annotated[Optional[Path], typer.Option("--config", "-c")] = None,
    task: Annotated[Optional[str], typer.Option("--task", "-t")] = None,
    model: Annotated[Optional[str], typer.Option("--model", "-m")] = None,
    tasks: Annotated[Optional[str], typer.Option("--tasks")] = None,
    models: Annotated[Optional[str], typer.Option("--models")] = None,
    data_root: Annotated[Optional[Path], typer.Option("--data-root")] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Machine-readable output")] = False,
    lock_file: Annotated[Path, typer.Option("--lock-file")] = LOCK_PATH,
) -> None:
    """Show what's present locally vs declared."""
    root = (data_root or default_data_root()).resolve()
    artifacts = _select(config, task, model, tasks, models)
    rows = run_status(artifacts, _load_lock(lock_file), root)

    if as_json:
        _echo(_json.dumps([r.__dict__ for r in rows], indent=2))
        return
    _echo(f"data root: {root}")
    for r in rows:
        if r.note:
            _echo(f"  {r.artifact_id:28s} invalid: {r.note}")
            continue
        if r.cache_only:
            _echo(f"  {r.artifact_id:28s} [hf-cache, not tracked locally]")
            continue
        mark = "✓" if r.present == r.total and r.total else ("·" if not r.total else "partial")
        _echo(
            f"  {r.artifact_id:28s} {mark:8s} "
            f"{r.present}/{r.total} files  "
            f"{format_bytes(r.bytes_present)}/{format_bytes(r.bytes_total)}"
        )


# ──────────────────────────────────────────────────────────────
# list
# ──────────────────────────────────────────────────────────────


@app.command("list")
def list_cmd(
    as_json: Annotated[bool, typer.Option("--json", help="Machine-readable output")] = False,
) -> None:
    """List every declared artifact, its mirrors, and what needs it."""
    if as_json:
        out = [
            {
                "id": a.id,
                "kind": a.kind.value,
                "dest": a.dest,
                "needed_by": list(a.needed_by),
                "requires": list(a.requires),
                "license": a.license.value,
                "redistributable": a.redistributable,
                "cache_only": a.cache_only,
                "mirrors": [
                    {"backend": m.backend.value, "base": m.base, "prefix": m.prefix}
                    for m in a.mirrors
                ],
            }
            for a in ARTIFACTS
        ]
        _echo(_json.dumps(out, indent=2))
        return
    for a in ARTIFACTS:
        mirrors = ", ".join(f"{m.backend.value}:{m.base}" for m in a.mirrors)
        _echo(f"{a.id}  [{a.kind.value}]")
        _echo(f"    dest:      {a.dest or '(hf cache)'}")
        _echo(f"    needed_by: {', '.join(a.needed_by) or '—'}")
        _echo(f"    license:   {a.license.value}  redistributable={a.redistributable}")
        _echo(f"    mirrors:   {mirrors}")


# ──────────────────────────────────────────────────────────────
# lock (maintainer)
# ──────────────────────────────────────────────────────────────


@app.command("lock")
def lock_cmd(
    artifact_ids: Annotated[Optional[str], typer.Option("--artifacts", "--tasks", help="Only re-lock these artifact ids")] = None,
    data_root: Annotated[Optional[Path], typer.Option("--data-root")] = None,
    lock_file: Annotated[Path, typer.Option("--lock-file")] = LOCK_PATH,
) -> None:
    """Recompute per-file checksums from the local copy and write the lock.

    Maintainer tool: run after (re)building a distribution so the committed lock
    matches the bytes on the mirrors.
    """
    root = (data_root or default_data_root()).resolve()
    ids = _split(artifact_ids)
    artifacts = (
        [artifact_by_id(i) for i in ids] if ids else list(ARTIFACTS)
    )
    try:
        fresh = build_lock_entries(artifacts, root)
        merged = prune_sources(merge_lock(_load_lock(lock_file), fresh), artifacts)
        write_lock(merged, lock_file)
    except (LockError, OSError) as exc:
        _echo(f"lock failed: {exc}")
        raise typer.Exit(code=1) from exc

    n_files = sum(len(e["files"]) for e in fresh.values())
    _echo(f"locked {len(fresh)} artifact(s), {n_files} files → {lock_file}")
    for aid in sorted(fresh):
        _echo(f"  {aid}: {len(fresh[aid]['files'])} files")


@app.command("pin")
def pin_cmd(
    artifact_ids: Annotated[str, typer.Option("--artifacts", help="Comma-separated artifact ids with HF sources")],
    revision: Annotated[str, typer.Option("--revision", help="Branch, tag, or commit to resolve and verify")] = "main",
    lock_file: Annotated[Path, typer.Option("--lock-file")] = LOCK_PATH,
) -> None:
    """Pin existing HF data/model sources; verify dataset bytes before changing the lock."""
    selected = [artifact_by_id(i) for i in _split(artifact_ids) or []]
    if not selected:
        raise typer.BadParameter("select at least one artifact", param_hint="--artifacts")
    try:
        pinned = pin_inputs(selected, _load_lock(lock_file), revision)
        write_lock(pinned, lock_file)
    except (LockError, OSError, ImportError) as exc:
        _echo(f"pin failed: {exc}")
        raise typer.Exit(code=1) from exc
    for art in selected:
        for source in pinned["artifacts"][art.id]["sources"]:
            _echo(f"  {art.id}: {source['repo_id']}@{source['revision']}")
    _echo(f"pinned sources → {lock_file}")


# ──────────────────────────────────────────────────────────────
# publish (maintainer)
# ──────────────────────────────────────────────────────────────


@app.command("publish")
def publish_cmd(
    to: Annotated[str, typer.Option("--to", help="Mirror to publish to: hf | gcs")],
    artifact_ids: Annotated[Optional[str], typer.Option("--artifacts", "--tasks", help="Only these artifact ids (default: all redistributable)")] = None,
    billing_project: Annotated[Optional[str], typer.Option("--billing-project", help="GCP project to bill for the requester-pays GCS bucket (or set YBENCH_GCS_BILLING_PROJECT); ignored for --to hf")] = None,
    data_root: Annotated[Optional[Path], typer.Option("--data-root")] = None,
    message: Annotated[str, typer.Option("--message", "-M", help="Commit message (HF)")] = "publish benchmark data",
    yes: Annotated[bool, typer.Option("--yes", help="Actually upload (without this it's a dry run)")] = False,
    lock_file: Annotated[Path, typer.Option("--lock-file")] = LOCK_PATH,
    receipt_dir: Annotated[Path, typer.Option("--receipt-dir", help="Save HF commit receipts here")] = Path("data/publications"),
) -> None:
    """[maintainer] Upload redistributable artifacts to a mirror, exactly as
    locked. Re-checks every file against the lock before uploading; refuses to
    publish anything marked non-redistributable. Dry-runs unless --yes."""
    kind = _backend_kind(to)
    if kind == BackendKind.HTTP:
        raise typer.Exit("--to must be hf or gcs (http is read-only)")
    root = (data_root or default_data_root()).resolve()
    ids = _split(artifact_ids)
    artifacts = [artifact_by_id(i) for i in ids] if ids else list(ARTIFACTS)
    lock = _load_lock(lock_file)

    def committed(receipt, updated_lock):
        # Save the remote identity first so a failed local lock write is recoverable.
        receipt_path = receipt_dir / f"{receipt.repo_id.replace('/', '--')}--{receipt.revision}.json"
        atomic_write_json(receipt_path, receipt.to_dict())
        write_lock(updated_lock, lock_file)
        _echo(f"  receipt: {receipt_path}")
        _echo(f"  updated input lock: {lock_file}")

    dry_run = not yes
    _echo(f"{'DRY RUN — ' if dry_run else ''}publish to {kind.value}  (data root: {root})")
    summary = run_publish(
        artifacts, lock, kind, root, message, dry_run,
        log=_echo, billing_project=billing_project, on_commit=committed,
    )
    _echo("")
    verb = "would upload" if dry_run else "uploaded"
    _echo(f"{verb} {summary.files} files across {summary.artifacts} artifact(s)")
    if summary.skipped:
        _echo(f"skipped: {len(summary.skipped)}")
        for m in summary.skipped:
            _echo(f"  · {m}")
    if summary.failures:
        _echo(f"FAILURES: {len(summary.failures)}")
        for m in summary.failures:
            _echo(f"  ! {m}")
        raise typer.Exit(code=1)
    if dry_run:
        _echo("\nre-run with --yes to upload.")


# ──────────────────────────────────────────────────────────────
# build (v2 — not implemented)
# ──────────────────────────────────────────────────────────────


@app.command("build")
def build_cmd(
    task: Annotated[str, typer.Argument(help="Task to rebuild from raw data")],
) -> None:
    """[v2] Rebuild processed data from raw inputs and verify against the lock."""
    _echo(
        "`ybench data build` is part of benchmark-v2 (raw → build → reproduce) "
        "and isn't implemented yet."
    )
    raise typer.Exit(code=2)
