# CLI reference

The repo ships a single CLI, `ybench`, driven by a YAML run-spec. The committed
`configs/default.yaml` is the canonical run — it captures which `(model, task)`
pairs to evaluate with what per-run settings, and is the single source of truth
for the numbers we report.

- [Install](#install)
- [Running benchmarks](#running-benchmarks)
- [Progress, hardware, and GPU selection](#progress-hardware-and-gpu-selection)
- [Output layout](#output-layout)
- [Getting the data](#getting-the-data)
- [Running without a local NVIDIA GPU (`ybench modal`)](#running-without-a-local-nvidia-gpu-ybench-modal)

## Install

The framework uses [`uv`](https://docs.astral.sh/uv/). Model-specific
dependencies (PyTorch, Shorkie weights, Yorzoi) are isolated behind extras:

```bash
# Minimal install: core benchmark framework + data loaders (no model deps)
uv sync

# Add specific model dependencies as needed
uv sync --extra shorkie   # PyTorch + h5py, for the Shorkie adapter
uv sync --extra yorzoi    # yorzoi, for the Yorzoi adapter
uv sync --extra dream_rnn # PyTorch, for the DREAM-RNN supervised baseline
uv sync --extra data      # huggingface_hub, for the `ybench data` backend
uv sync --extra all       # all models + data backend (everything default.yaml needs)

# CodonTransformer baseline needs one extra step: its package pins numpy<2 /
# pandas<3 (conservative — it runs fine on this project's numpy 2 / pandas 3), so
# install it --no-deps after syncing its light deps:
uv sync --extra codon_transformer
uv pip install --no-deps codontransformer
```

## Running benchmarks

```bash
# List registered models and tasks
uv run ybench list

# Preview the planned runs without executing
uv run ybench run --config configs/default.yaml --dry-run

# Execute every (model, task) pair in the config
uv run ybench run --config configs/default.yaml

# Filter to a single model or task
uv run ybench run --config configs/default.yaml --model shorkie
uv run ybench run --config configs/default.yaml --task  caudal_eqtl
```

`run` flags:

- `--config <path>` — the YAML run-spec (required).
- `--dry-run` — print the planned pairs and exit without scoring.
- `--model <name>` / `--task <name>` — restrict the run to one model or task.
- `--gpu <n>` — pick a GPU by index (shorthand for `--device cuda:<n>`).
- `--device <cuda:N|cpu>` — choose the device explicitly.
- `--no-data-check` — skip the pre-flight data-readiness check.

## Progress, hardware, and GPU selection

`run` opens with a banner — config hash, resolved device (GPU name + free
VRAM + `CUDA_VISIBLE_DEVICES` when CUDA is available), a data-readiness check,
and the planned pairs — then prints a `[i/N]` line per pair with a running
mean-pair-time **ETA** so you can see at a glance what's done and how long is
left:

```
hardware:      cuda:0  NVIDIA A100-80GB  (79.2/80.0 GB free)
data:          ✓ ready  50/50 files (562.6MB/562.6MB), 2 hf-cache model(s)
runs:          21 pair(s)

[ 1/21] shorkie × caudal_eqtl → results/default/shorkie__caudal_eqtl
  …
  ✓ done in 41s · elapsed 41s · mean 41s/pair · ETA ~13m40s
```

Before executing, `run` pre-flights the data: a real run **stops** if any
required file is missing or stale (with the `ybench data get` command to fix
it); `--no-data-check` skips that. Pick the GPU with `--gpu 2` (shorthand for
`--device cuda:2`) or `--device cpu`. Runs are sequential — one GPU at a time;
to isolate a physical GPU on a shared box use `CUDA_VISIBLE_DEVICES`.

## Output layout

One directory per `(model, task)` pair, under the config's `out_dir`:

```
results/default/
  shorkie__caudal_eqtl/
    negset_{1..4}_scores.npy   # per-iteration raw scores
    negset_{1..4}_labels.npy   # per-iteration labels
    negset_{1..4}_pairs.tsv    # per-pair metadata (pair_id, distances)
    summary.json               # per-iter + aggregate AUROC / AUPRC
    run_metadata.json          # config hash, git commit, timestamp
    primary_roc_pr.png         # ROC + PR on full set (|score|, mean ± SEM, baselines)
    close_only_roc_pr.png      # same, filtered to pos_distance_to_tss ≤ 2 kb
    distance_stratified.png    # AUROC / AUPRC per distance-to-TSS bin
  yorzoi__caudal_eqtl/
    …
```

`run_metadata.json` captures everything needed to reproduce that directory's
numbers — the config hash, repo git commit, resolved model/task configs,
and timestamp. Raw scores + labels + pair metadata are persisted so
post-hoc analyses (distance-stratification, signed-vs-absolute comparison,
etc.) don't require re-running the model.

## Getting the data

Task data and model weights are **not** in git. `ybench data` downloads them
from a mirror (HuggingFace or GCS) and checksum-verifies every file against a
committed lock (`src/yeastbench/data/manifest.lock.json`).

```bash
uv sync --extra data            # adds huggingface_hub (HF backend)

# Pull everything a config's runs need (recommended — same selection as `run`)
uv run ybench data get --config configs/default.yaml

# Or pull explicit subsets / everything
uv run ybench data get --tasks cuperus_utr,caudal_eqtl --models shorkie
uv run ybench data get                     # all artifacts
uv run ybench data get --dry-run           # show the plan, fetch nothing

# Inspect and check
uv run ybench data list                    # every artifact, its mirrors, license
uv run ybench data status                  # what's present locally vs declared
uv run ybench data verify                  # checksum local files against the lock
```

`get` is idempotent (skips files already present and valid), picks the first
reachable mirror (HF first, then GCS; override with `--from hf|gcs`), and writes
atomically. Shorkie weights, Yorzoi, and CodonTransformer resolve from their
public homes; the per-task processed data comes from the project mirror. Add
`--json` to `list`/`status` for machine-readable output.

The HF mirror is free and needs no account — it's the default. The GCS mirror
(`gs://yeast-seq2expression-benchmark`) is **requester pays**, so `--from gcs`
needs your own GCP project to bill: pass `--billing-project <project>` or set
`YBENCH_GCS_BILLING_PROJECT`. If you don't have a project, just use the HF
default.

**How much you'll download.** The full set is **~690 MB**: Shorkie weights
(~440 MB), all task data (~215 MB), and the R64 reference genomes (~34 MB). A
single task is much smaller — from ~11 KB (`hong`) to ~132 MB (`brooks_scramble`)
— so scope your `get` to the config or `--tasks`/`--models` you actually need.
Yorzoi and CodonTransformer aren't in that figure: they're pulled into the
HuggingFace cache the first time you run those models (additional, model-sized).
Run `ybench data status` for an exact present-vs-total byte count, or
`ybench data get --dry-run` to see the size before fetching.

> Maintainers: `ybench data lock` re-freezes the checksum lock from a local
> copy, and `ybench data publish --to hf|gcs` uploads the redistributable
> artifacts to a mirror (dry-run unless `--yes`). Publishing to GCS needs a
> billing project (`--billing-project` / `YBENCH_GCS_BILLING_PROJECT`).

## Running without a local NVIDIA GPU (`ybench modal`)

No local NVIDIA GPU? Run the benchmark on a [Modal](https://modal.com) GPU
instead. The remote container runs the *unmodified* `ybench` CLI, so results
match a local run — only the transport differs.

One-time setup (the `modal` client installs anywhere; no local GPU needed):

```bash
uv sync --extra modal      # installs the modal client
uv run modal setup         # browser login; writes ~/.modal.toml
```

Then seed the data once and run:

```bash
uv run ybench modal data get --config configs/default.yaml   # one-time CPU seed of the data volume
uv run ybench modal run      --config configs/default.yaml   # full run on a GPU; results land in ./results/default
```

`run` takes the same `--model`/`--task` filters as `ybench run`, plus `--gpu`
(a Modal GPU *type* — `A10`, `L4`, `T4`, … — not a device index; see
[Modal's GPU docs](https://modal.com/docs/guide/gpu)), `--out` (where results are
downloaded, default `results`), and `--detach`.

Results are namespaced per config hash on the volume, so two config versions
never mix (and the auto-compare never tables results from different configs
together). `ybench modal pull -c <config>` re-downloads just that config's
results; without `-c` it pulls every config's namespace. `ybench modal status`
lists what's on the data/results volumes. The default data path is public and
free (HuggingFace mirror), so no extra secrets are required. A full run is
~$1–4 of GPU time (≈1–2 GPU-hours on an A10), covered by Modal's free monthly
credit.
