"""Modal GPU compute backend — the engine behind ``ybench modal``.

Lift-and-shift: a CPU ``seed`` function fetches data/weights into a Volume, then
a GPU ``run_benchmark`` function runs the *unmodified* ``ybench`` CLI against
that Volume and returns the results tree. The eval/scoring code is never
reimplemented — the same bytes run remotely.

Importing this module is cheap and offline: Modal Image/Volume/Function objects
are lazy specs; nothing builds or connects until the CLI enters ``app.run()``.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path, PurePosixPath

import modal

from yeastbench.modal.plan import RESULTS_ROOT

APP_NAME = "ybench"
GPU_DEFAULT = "A10"

# Anchor the tree we ship to the PACKAGE, not the process cwd: this file lives at
# <repo>/src/yeastbench/modal/app.py, so parents[3] is the checkout root. Deriving
# it from cwd (e.g. default_data_root()) would ship whatever directory `ybench`
# happened to be invoked from — $HOME for a global install — and fail obscurely.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if not (_REPO_ROOT / "pyproject.toml").exists():  # pragma: no cover - misconfiguration
    raise RuntimeError(
        f"expected a source checkout at {_REPO_ROOT} (no pyproject.toml there). "
        "The Modal backend ships this tree to the container, so it must run from "
        "a git checkout, not an installed wheel."
    )
_IMAGE_IGNORE = [
    "data/**",
    "results/**",
    "tmp-ytk-implementation/**",
    "**/__pycache__",
    ".venv/**",
]
# .git is intentionally NOT ignored: `ybench run` records git_commit in
# run_metadata.json, so the checkout needs to be baked into the image.


app = modal.App(APP_NAME)

# ── Volumes (two) ───────────────────────────────────────────────────────────
# The HF cache rides inside the data volume (HF_HOME=/repo/data/.hf), so there's
# no separate cache volume: one "inputs" volume, one "results" volume.
data_vol = modal.Volume.from_name("ybench-data", create_if_missing=True)
results_vol = modal.Volume.from_name("ybench-results", create_if_missing=True)
RESULTS_MOUNT = Path("/repo") / RESULTS_ROOT
VOLUMES = {"/repo/data": data_vol, str(RESULTS_MOUNT): results_vol}
# HF cache lives inside the data volume; no HF_HUB_ENABLE_HF_TRANSFER — newer
# huggingface_hub deprecated it (hf_transfer is unused) and warns on every run.
HF_ENV = {"HF_HOME": "/repo/data/.hf"}

# ── Images ──────────────────────────────────────────────────────────────────
# Seed runs on a torch-free CPU container: just the data backend.
cpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .add_local_dir(str(_REPO_ROOT), "/repo", copy=True, ignore=_IMAGE_IGNORE)
    .uv_pip_install("/repo[data]", extra_options="-e")  # editable, mirrors local `uv sync`
    .env(HF_ENV)
)

# GPU run: a plain image + the model extras. yorzoi 0.2.1 no longer needs
# flash-attn, so torch comes in as an ordinary dependency of the extras (a
# standard CUDA-bundled wheel that runs on Modal's GPUs) — no CUDA base image,
# cu130 pin, or prebuilt wheel required.
gpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    # So `git rev-parse` works on the baked /repo checkout (run by `ybench run`
    # to record git_commit in run_metadata.json) despite root/dubious-ownership.
    .run_commands("git config --global --add safe.directory /repo")
    .add_local_dir(str(_REPO_ROOT), "/repo", copy=True, ignore=_IMAGE_IGNORE)
    # Install the model extras + yorzoi + CodonTransformer.
    # EDITABLE (like the local `uv sync`), so the package lives at /repo/src and
    # `import yeastbench` resolves there: several adapters locate their frozen
    # default data files via `Path(__file__).resolve().parents[3]`, which only
    # equals the repo root in the src/ checkout layout. A non-editable install
    # (site-packages) would make parents[3] wrong — e.g. hong's cassette would
    # resolve to /data/tasks/... and 404. `include_source=False` on the functions
    # keeps this editable install the only copy of the package on the container's
    # path (no auto-mounted /root/yeastbench to shadow it).
    #
    # `codon_transformer` extra carries CodonTransformer's *light* deps
    # (python_codon_tables/biopython/CAI); CodonTransformer itself is installed
    # --no-deps because it pins numpy<2 / pandas<3 and drags in onnxruntime /
    # pytorch-lightning, none of which the chen adapter uses — those caps are
    # conservative and it runs fine on numpy 2 / pandas 3 (verified).
    .uv_pip_install("/repo[shorkie,dream_rnn,data,codon_transformer]", extra_options="-e")
    .uv_pip_install("yorzoi==0.2.1")
    .uv_pip_install("codontransformer", extra_options="--no-deps")
    .env(HF_ENV)
)


def _write_config(config_bytes: bytes, config_name: str) -> str:
    """Write the user's config bytes into the container's configs/ dir under its
    original name. Preserves config_hash (same bytes) and the config basename;
    config_path in run_metadata.json is the container path (/repo/configs/<name>),
    which is environment-relative by design and has no effect on scores."""
    rel = f"configs/{config_name}"
    dest = Path("/repo") / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(config_bytes)
    return rel


@app.function(image=cpu_image, volumes=VOLUMES, timeout=2 * 3600, include_source=False)
def seed(
    config_bytes: bytes,
    config_name: str,
    model: str | None = None,
    task: str | None = None,
) -> None:
    """Fetch this config's data + weights into the volumes on a CPU container, so
    the ~1 GB download is never billed at GPU rates. Idempotent: the canonical
    ``ybench data get`` skips checksum-OK files, so re-seeding is near-free.
    ``model``/``task`` mirror the run's filters so a filtered run seeds only what
    it needs."""
    rel = _write_config(config_bytes, config_name)
    cmd = ["ybench", "data", "get", "--config", rel]
    if model:
        cmd += ["--model", model]
    if task:
        cmd += ["--task", task]
    subprocess.run(cmd, cwd="/repo", check=True)
    data_vol.commit()  # persists locked data/ AND the /repo/data/.hf cache


@app.function(
    image=gpu_image, gpu=GPU_DEFAULT, volumes=VOLUMES, timeout=8 * 3600,
    include_source=False,
)
def run_benchmark(
    config_bytes: bytes,
    config_name: str,
    config_hash: str,
    out_dir: str = "results/default",
    expected_pairs: list[str] | None = None,
    model: str | None = None,
    task: str | None = None,
) -> list[str]:
    """Run the unmodified ``ybench`` CLI on the GPU against the seeded volume,
    commit the results, and return the pair dirs THIS run produced (a small
    manifest — the bulk results are downloaded by the caller via
    ``modal volume get``, which has no return-value size limit).

    Results are namespaced per ``config_hash`` on the volume, so two different
    config versions never share an out_dir — otherwise ``ybench run``'s
    auto-compare would silently table results from different configs (and
    different git commits) against each other."""
    data_vol.reload()  # see what seed committed
    results_vol.reload()
    rel = _write_config(config_bytes, config_name)

    live = Path("/repo") / out_dir  # where the config tells `ybench run` to write
    # This config's own subtree, preserving out_dir's shape under the results
    # root (results/default → <hash>/default, results/a/b → <hash>/a/b).
    rel_out = PurePosixPath(out_dir).relative_to(RESULTS_ROOT)
    bucket = RESULTS_MOUNT / config_hash / rel_out

    # Restore this config's history into the live path so the auto-compare can
    # still pair models across separate invocations of the SAME config. Anything
    # already at the shared live path belongs to another config (or a crashed
    # run), so it goes first. Copy rather than move: the bucket stays intact if
    # this run dies partway.
    if live.exists():
        shutil.rmtree(live)
    if bucket.exists():
        shutil.copytree(bucket, live)

    # Force torch device "cuda": a Modal GPU container exposes exactly one GPU as
    # cuda:0. We intentionally override the config's `device:` field (meant for
    # local multi-GPU boxes, e.g. "cuda:1", which would crash here). The Modal GPU
    # *type* (A10/L4/…) is a separate choice, set via the function's `gpu=`.
    # `--no-data-check` because the CPU seed already verified every checksum —
    # re-hashing the whole dataset here would do it at GPU billing rates.
    cmd = ["ybench", "run", "--config", rel, "--device", "cuda", "--no-data-check"]
    if model:
        cmd += ["--model", model]
    if task:
        cmd += ["--task", task]
    # The real run, including the auto-triggered cross-model compare.
    subprocess.run(cmd, cwd="/repo", check=True)

    # Stash back under the hash namespace and leave the shared path clean for the
    # next (possibly different) config.
    bucket.parent.mkdir(parents=True, exist_ok=True)
    if bucket.exists():
        shutil.rmtree(bucket)
    shutil.move(str(live), str(bucket))
    results_vol.commit()

    # Report only what THIS run was asked to produce — the bucket also holds the
    # restored history, so listing it wholesale would over-report.
    return sorted(d for d in (expected_pairs or []) if (bucket / d).exists())
