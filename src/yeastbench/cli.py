"""Unified CLI for yeastbench. Invoke via ``ybench`` (installed script) or
``python -m yeastbench.cli``.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Optional

import typer

from yeastbench.config import Config, load_config
from yeastbench.data.cli import app as data_app
from yeastbench.data.fetch import (
    RunDataCheck,
    _human,
    check_run_data,
    default_data_root,
)
from yeastbench.hardware import describe_device
from yeastbench.registry import MODELS, TASKS


app = typer.Typer(add_completion=False, help="yeast-seq2expression benchmark runner")
app.add_typer(data_app, name="data")

# Optional Modal GPU backend — `ybench modal …` (run the benchmark without a
# local NVIDIA GPU). Registered only when the `modal` extra is installed; `modal`
# itself is imported lazily per-command, so this adds no startup cost elsewhere.
if importlib.util.find_spec("modal") is not None:
    from yeastbench.modal.cli import app as modal_app

    app.add_typer(modal_app, name="modal")


def _echo(msg: str) -> None:
    typer.echo(msg)


def _git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _run_pair(cfg: Config, model_name: str, task_name: str, model_config: dict) -> None:
    if model_name not in MODELS:
        raise ValueError(f"Unknown model '{model_name}'. Known: {sorted(MODELS)}")
    if task_name not in TASKS:
        raise ValueError(f"Unknown task '{task_name}'. Known: {sorted(TASKS)}")

    task_config = cfg.tasks_config.get(task_name, {})
    out_dir = cfg.out_dir / f"{model_name}__{task_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    task = TASKS[task_name](**task_config)
    adapter = MODELS[model_name](task, device=cfg.device, **model_config)
    _echo(f"  ready in {time.time() - t0:.1f}s")

    t0 = time.time()
    results = task.evaluate(adapter)
    eval_s = time.time() - t0
    _echo(f"  evaluated in {eval_s:.1f}s")

    # Persist prediction arrays first so a plotting bug doesn't lose
    # the eval results (forcing a re-forward through the model).
    task.save_results(results, out_dir)

    t0 = time.time()
    task.plot(results, out_dir)
    _echo(f"  plots  written in {time.time() - t0:.1f}s")

    summary = {
        "model": model_name,
        "task": task_name,
        "task_version": task.info.version,
        **task.summary_dict(results),
    }
    prediction_metadata = getattr(adapter, "benchmark_metadata", None)
    if prediction_metadata is not None:
        diagnostics = getattr(adapter, "benchmark_diagnostics", None)
        if diagnostics is not None:
            prediction_metadata = {**prediction_metadata, **diagnostics()}
        summary["prediction_metadata"] = prediction_metadata
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    (out_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "model": model_name,
                "task": task_name,
                "task_version": task.info.version,
                "config_path": str(cfg.source_path),
                "config_hash": cfg.source_hash,
                "device": cfg.device,
                "model_config": model_config,
                **({"prediction_metadata": prediction_metadata}
                   if prediction_metadata is not None else {}),
                "task_config": task_config,
                "git_commit": _git_commit(),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_eval_s": eval_s,
            },
            indent=2,
        )
    )

    _echo(f"  {task.headline(results)}")


@app.command("run")
def run_cmd(
    config: Annotated[
        Path, typer.Option("--config", "-c", help="YAML run-spec path")
    ],
    model: Annotated[
        Optional[str],
        typer.Option("--model", "-m", help="Run only this model (filter)"),
    ] = None,
    task: Annotated[
        Optional[str],
        typer.Option("--task", "-t", help="Run only this task (filter)"),
    ] = None,
    device: Annotated[
        Optional[str],
        typer.Option("--device", help="Override the config device (e.g. cuda:2, cpu)"),
    ] = None,
    gpu: Annotated[
        Optional[int],
        typer.Option("--gpu", help="Shorthand for --device cuda:N (pick one GPU)"),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="List planned runs, check data presence, and exit"),
    ] = False,
    no_data_check: Annotated[
        bool,
        typer.Option("--no-data-check", help="Skip the pre-flight data/weights check"),
    ] = False,
) -> None:
    """Execute (model, task) pairs defined by the config, with optional filters."""
    cfg = load_config(config).filtered(model, task)
    if not cfg.runs:
        raise typer.Exit(
            f"No runs match filters (model={model!r}, task={task!r}) in {config}"
        )

    resolved_device = device or (f"cuda:{gpu}" if gpu is not None else cfg.device)
    if resolved_device != cfg.device:
        cfg = replace(cfg, device=resolved_device)

    pairs = [(r, t) for r in cfg.runs for t in r.tasks]
    _echo(f"config:        {cfg.source_path}  [hash {cfg.source_hash}]")
    _echo(f"out_dir:       {cfg.out_dir}")
    for line in describe_device(cfg.device).lines():
        _echo(line)

    check = None
    if not no_data_check:
        sel_tasks = sorted({t for r in cfg.runs for t in r.tasks})
        sel_models = sorted({r.model for r in cfg.runs})
        check = check_run_data(sel_tasks, sel_models, default_data_root())
        _print_data_check(check)

    _echo(f"runs:          {len(pairs)} pair(s)")
    for r, t in pairs:
        _echo(f"  - {r.model} × {t}")

    if dry_run:
        raise typer.Exit(code=0)

    if check is not None and not check.ready:
        _echo(
            "\nrequired data/weights are missing or stale. fetch them with:\n"
            f"  uv run ybench data get --config {config}\n"
            "or re-run with --no-data-check to proceed anyway."
        )
        raise typer.Exit(code=1)

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    total = len(pairs)
    durations: list[float] = []
    started = time.time()
    for i, (r, t) in enumerate(pairs, 1):
        out_dir = cfg.out_dir / f"{r.model}__{t}"
        _echo(f"\n[{i:>2}/{total}] {r.model} × {t} → {out_dir}")
        t0 = time.time()
        _run_pair(cfg, r.model, t, r.model_config)
        dt = time.time() - t0
        durations.append(dt)
        _echo(_progress_line(i, total, dt, durations, time.time() - started))

    _echo(f"\ndone {total}/{total} in {_fmt_dur(time.time() - started)}  ·  {cfg.out_dir}")

    # Auto-trigger the cross-model comparison. Walks the FULL config's
    # `out_dir` (i.e. ignores --model / --task filters when looking for
    # peer results), so a run that just produced one of two models still
    # gets paired against the prior model's results on disk.
    full_cfg = load_config(config)
    _run_compare(full_cfg)


def _progress_line(
    done: int, total: int, last_dt: float, durations: list[float], elapsed: float
) -> str:
    msg = f"  ✓ done in {_fmt_dur(last_dt)} · elapsed {_fmt_dur(elapsed)}"
    if done < total:
        mean = sum(durations) / len(durations)
        msg += f" · mean {_fmt_dur(mean)}/pair · ETA ~{_fmt_dur(mean * (total - done))}"
    return msg


def _fmt_dur(seconds: float) -> str:
    s = int(round(seconds))
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


def _print_data_check(c: RunDataCheck) -> None:
    mark = "✓ ready" if c.ready else "✗ incomplete"
    extra = f", {len(c.cache_only)} hf-cache model(s)" if c.cache_only else ""
    _echo(
        f"data:          {mark}  "
        f"{c.files_present}/{c.files_total} files "
        f"({_human(c.bytes_present)}/{_human(c.bytes_total)}){extra}"
    )
    for aid, rel in c.missing:
        _echo(f"  missing: {aid}/{rel}")
    for aid, rel in c.stale:
        _echo(f"  stale:   {aid}/{rel}")


def _run_compare(cfg: Config) -> None:
    """Auto-triggered at the end of `ybench run`. Silent no-op when
    there's nothing to compare."""
    from yeastbench.compare import compare

    result = compare(cfg)
    if result.empty:
        return
    _echo(f"\nCross-model comparison → {result.out_dir}")
    _echo(f"  tasks compared: {', '.join(result.tasks_compared)}")
    if result.tasks_skipped:
        _echo(
            "  tasks skipped (< 2 models): "
            + ", ".join(result.tasks_skipped)
        )
    if result.summary_csv:
        _echo(f"  summary.csv:    {result.summary_csv}")
    if result.summary_md:
        _echo(f"  summary.md:     {result.summary_md}")


@app.command("list")
def list_cmd() -> None:
    """List registered models and tasks."""
    _echo("models:")
    for name in sorted(MODELS):
        _echo(f"  - {name}")
    _echo("tasks:")
    for name in sorted(TASKS):
        _echo(f"  - {name}")


if __name__ == "__main__":
    app()
