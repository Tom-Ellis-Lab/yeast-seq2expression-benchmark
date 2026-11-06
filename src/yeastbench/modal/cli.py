"""``ybench modal`` — run the benchmark on Modal GPUs (no local NVIDIA GPU needed).

A thin transport over ``yeastbench.modal.app``: read the config locally, ship its
bytes to a remote container that runs the *unmodified* ``ybench`` CLI, and bring
the results tree back. The run/scoring logic is never duplicated.

``modal`` is imported lazily inside each command, so ``ybench modal --help`` and
the rest of the CLI work even when the ``modal`` extra isn't fully importable.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Annotated, Optional

import typer

app = typer.Typer(
    add_completion=False, help="Run ybench on Modal GPUs (no local GPU needed)."
)
data_app = typer.Typer(
    add_completion=False, help="Seed the remote data/weights volume."
)
app.add_typer(data_app, name="data")

_CONFIG_OPT = typer.Option("--config", "-c", help="YAML run-spec path")
_MODEL_OPT = typer.Option("--model", "-m", help="Run only this model (filter)")
_TASK_OPT = typer.Option("--task", "-t", help="Run only this task (filter)")


def _echo(msg: str) -> None:
    typer.echo(msg)


def _validate_gpu(gpu: str) -> str:
    """Reject a bare device index. On `ybench run` `--gpu` is an integer GPU
    index; here it's a Modal GPU *type*, so `--gpu 0` would otherwise be shipped
    verbatim and only rejected remotely, after the seed already ran.
    Valid GPU types: https://modal.com/docs/guide/gpu"""
    if gpu.strip().isdigit():
        raise typer.BadParameter(
            f"--gpu takes a Modal GPU type (A10, L4, T4, A100, H100, …), not a "
            f"device index like {gpu!r}. For a local GPU index use "
            f"`ybench run --gpu {gpu}`. Valid types: "
            "https://modal.com/docs/guide/gpu"
        )
    return gpu


def _modal_executable() -> str:
    """Find the Modal CLI, including beside an explicitly-invoked venv Python."""
    on_path = shutil.which("modal")
    if on_path is not None:
        return on_path
    beside_python = Path(sys.executable).with_name("modal")
    if beside_python.is_file():
        return str(beside_python)
    return "modal"


def _download_results(out: Path, remote: str = "/") -> None:
    """Download ``remote`` from the results volume into ``out``.

    Two ``modal volume get`` behaviours to keep in mind: it APPENDS the remote
    path's basename to the destination (so pull ``/<hash>/default`` into
    ``results/`` to land ``results/default/…``), and the destination must already
    be a directory or every file collapses onto the single ``out`` path
    (modal/cli/_download.py: ``output_path = dest / rel`` only when ``dest`` is a
    dir) — hence the mkdir.
    """
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            _modal_executable(),
            "volume",
            "get",
            "--force",
            "ybench-results",
            remote,
            str(out),
        ],
        check=True,
    )


def _download_config_results(plan, out: Path) -> None:
    """Pull one config's namespace so it lands at ``out/<out_dir-relative>``,
    matching the layout a local run produces."""
    rel = plan.out_dir_rel
    dest = out if str(rel.parent) == "." else out / rel.parent
    _download_results(dest, f"/{plan.source_hash}/{rel}")


@data_app.command("get")
def data_get(
    config: Annotated[Path, _CONFIG_OPT],
    model: Annotated[Optional[str], _MODEL_OPT] = None,
    task: Annotated[Optional[str], _TASK_OPT] = None,
) -> None:
    """One-time CPU seed: fetch this config's data + weights into the volume."""
    import modal

    from yeastbench.modal.app import app as modal_app, seed
    from yeastbench.modal.plan import build_plan

    plan = build_plan(config, model, task)
    _echo(f"seeding 'ybench-data' volume for {len(plan.pairs)} pair(s) from {config} …")
    with modal.enable_output(), modal_app.run():
        seed.remote(plan.config_bytes, plan.config_name, model=model, task=task)
    _echo("done — data + weights committed to the 'ybench-data' volume.")


@app.command("run")
def run(
    config: Annotated[Path, _CONFIG_OPT],
    model: Annotated[Optional[str], _MODEL_OPT] = None,
    task: Annotated[Optional[str], _TASK_OPT] = None,
    gpu: Annotated[
        str, typer.Option("--gpu", help="Modal GPU type, e.g. A10, L4, T4, A100, H100")
    ] = "A10",
    out: Annotated[
        Path, typer.Option("--out", help="local dir to download results into")
    ] = Path("results"),
    detach: Annotated[
        bool, typer.Option("--detach", help="keep running if the client disconnects")
    ] = False,
) -> None:
    """Seed (CPU), then run the benchmark on a Modal GPU and pull results back.

    The GPU *type* is chosen with ``--gpu`` (A10/L4/T4/…). The container's torch
    device is always ``cuda`` (its single GPU), independent of the config's
    ``device:`` field — so a config written for a local multi-GPU box still runs.
    """
    import modal

    from yeastbench.modal.app import app as modal_app, run_benchmark, seed
    from yeastbench.modal.plan import build_plan

    gpu = _validate_gpu(gpu)
    plan = build_plan(config, model, task)
    _echo(f"config:   {config}  [hash {plan.source_hash}]")
    _echo(f"gpu:      {gpu}")
    _echo(f"runs:     {len(plan.pairs)} pair(s) → {', '.join(plan.pair_dirs())}")

    with modal.enable_output(), modal_app.run(detach=detach):
        seed.remote(plan.config_bytes, plan.config_name, model=model, task=task)
        fn = run_benchmark.with_options(gpu=gpu)
        produced = fn.remote(
            plan.config_bytes,
            plan.config_name,
            plan.source_hash,
            out_dir=plan.out_dir,
            expected_pairs=plan.pair_dirs(),
            model=model,
            task=task,
        )

    _echo(f"\nremote run produced {len(produced)} pair dir(s): {', '.join(produced)}")
    _echo(f"downloading results → {out}/ …")
    # Results live under this config's hash namespace on the volume; pulling that
    # subtree recreates the local `<out>/<out_dir>/…` layout.
    _download_config_results(plan, out)
    _echo(f"done — results under {out}/")


@app.command("pull")
def pull(
    config: Annotated[
        Optional[Path],
        typer.Option("--config", "-c", help="pull only this config's results"),
    ] = None,
    out: Annotated[
        Path, typer.Option("--out", help="local dir to download results into")
    ] = Path("results"),
) -> None:
    """Download results from the volume (no run needed).

    With ``--config``, pulls just that config's hash namespace, recreating the
    local ``<out>/<out_dir>/…`` layout. Without it, pulls every config's
    namespace, so you get ``<out>/<config_hash>/…``.
    """
    from yeastbench.modal.plan import build_plan

    if config is not None:
        plan = build_plan(config)
        _echo(f"downloading results for {config} [hash {plan.source_hash}] …")
        _download_config_results(plan, out)
    else:
        _echo("downloading every config's results from 'ybench-results' …")
        _download_results(out)
    _echo(f"results written under {out}/")


@app.command("status")
def status() -> None:
    """List what's currently on the data and results volumes."""
    for vol in ("ybench-data", "ybench-results"):
        _echo(f"\n# {vol}")
        subprocess.run(
            [_modal_executable(), "volume", "ls", vol], check=False
        )
