# Add `shorkie_random_init` to ybench

## Context

Shorkie is a masked DNA language model pretrained on 165 fungal genomes, then
fine-tuned on yeast RNA-seq tracks. The authors also trained the same
architecture supervised **from scratch**, skipping the pretraining step, and
called it `Shorkie_Random_Init`. Their own summary of the difference: "the only
difference between finetuned and random-init is the `--restore` flag + learning
rate."

Scoring it across all ten ybench tasks turns the benchmark into a controlled
measurement of what language-model pretraining actually buys, task by task. The
paper reports that comparison for eQTLs and MPRAs; ybench would extend it to
reporter insertions, synonymous codons, SCRaMBLE rearrangements and foreign DNA.

Your instinct was right — it is a different set of weights, same architecture,
same adapters. There is exactly one wrinkle, described below.

## What I verified

**The weights are published and need no auth.** All eight folds return HTTP 200
at `https://storage.googleapis.com/seqnn-share/shorkie_models/shorkie_random_init/f{0..7}/model_best.h5`
(56,870,796 bytes each), with `params.json` at that prefix's root. I downloaded
fold 0 and opened it.

**The architecture is identical.** Diffing the two `params.json` files, the only
differences sit in the `train` block: `task: "fine-tune"→"supervised"` and
`learning_rate: 2e-05→0.0005`. The `model` block — sequence length 16,384, trunk
of conv_dna/res_tower/transformer_tower/3× unet_conv/Cropping1D, 5,215-unit
softplus head — is byte-identical. All 192 layer names in the two checkpoints
match.

**The one real difference is the input width.** Pretrained Shorkie's first conv
kernel is `(11, 170, 96)`; the random-init one is `(11, 4, 96)`. 170 = 4
nucleotides + 1 reserved + 165 species channels. The random-init model was
trained on plain DNA one-hot with no species encoding. The size gap between the
two checkpoints is 701,184 bytes = 11 × 166 × 96 × 4 exactly, which is that conv
layer and nothing else. Upstream's own scoring script says the same thing:
`shorkie-paper/from_kuanhao/eQTL/inference/caudal_etal_Shorkie_Random_Init/positive/1_score_variants_Shorkie.py:68`
— *"Do NOT set num_features (unlike Shorkie which needs 170 features)."*

Our PyTorch port hard-codes 170 in three places (`src/yeastbench/models/shorkie/nn.py`),
so it cannot load these checkpoints today. That is the whole code change.

## Where it runs: Modal

All four local A6000s are pinned at 98–100% utilisation by other users
(`es1219`, `xz9118`), with host load average 80 and 6 GB of 250 GB RAM free.
Only VRAM is free (~37 GB per card), not compute. Run on Modal: the backend is
merged on `dev` (`src/yeastbench/modal/`), `modal` is installed in `.venv`, and
`~/.modal.toml` already holds a token. An A10 run of ten tasks is roughly
$1–3.

## Steps

### 1. Branch, and take the URL fix with it

```bash
git checkout dev && git checkout -b feat/shorkie-random-init
git cherry-pick d16b299
```

`d16b299` ("Fix the Shorkie weights URL — upstream moved the bucket path") lives
only on `mytk-v1`. Without it `dev` cannot fetch Shorkie at all — the manifest
still points at `seqnn-share/shorkie/`, which 404s. Both variants now live under
`seqnn-share/shorkie_models/`, so this branch needs it. Cherry-pick rather than
retype, so the two branches don't conflict later.

### 2. Pull the existing Modal results first — before editing any config

`results/default/` on this machine holds only four pair directories (`cai`,
`codon_transformer`, `dream_rnn`, `yorzoi__hong_igr`). Every pretrained-Shorkie
number lives on the Modal results volume, namespaced by config hash
(`src/yeastbench/modal/app.py:159-160`). Editing `configs/default.yaml` changes
that hash, and `ybench modal pull -c configs/default.yaml` would then look in an
empty namespace.

```bash
uv run ybench modal pull -c configs/default.yaml --out results
uv run ybench modal pull -c configs/brooks.yaml  --out results
uv run ybench modal pull -c configs/meneu.yaml   --out results
```

Do this before step 5. (Recovery if it is missed: `ybench modal pull` with no
`-c` dumps every namespace under `results/<hash>/`.)

### 3. `src/yeastbench/models/shorkie/nn.py` — read the width from the checkpoint

Rather than add a flag and thread it through nine adapters, let the checkpoint
describe itself. `from_tf_checkpoint` already opens the H5; have it read the
conv kernel's shape first and build the module to match.

- Add `KNOWN_IN_CHANNELS = (DNA_CHANNELS, SPECIES_OFFSET + NUM_SPECIES)` next to
  the existing constants (`nn.py:12-17`).
- Add a small `_tf_conv_dna_in_channels(h5_path)` that reads
  `model_weights/conv1d/conv1d/kernel:0`'s `.shape[1]` without materialising the
  kernel, reusing the Keras double-nesting unwrap from `_get_tf_layer`
  (`nn.py:643-660`).
- Add `_with_conv_dna_in_channels(config, n)` returning a **copy** of the config
  whose `trunk[0]` carries `in_channels: n`. A copy, not a mutation —
  `wrapper.py:110-115` loads the config once and hands the same dict to all
  eight folds. `_build` already splats `trunk[0]` into `ConvDNA(**cfg)`
  (`nn.py:517-519`), so nothing else changes there.
- In `_build`, set `self._expects_species_encoding = self.conv_dna.conv.in_channels != DNA_CHANNELS`.
- In `forward` (`nn.py:596`), guard the expansion:
  `if self._expects_species_encoding: x = self._expand_species_encoding(x)`.
- In `from_tf_checkpoint` (`nn.py:626-635`), read the width, reject anything
  outside `KNOWN_IN_CHANNELS` with a message naming both valid widths, then
  build from the injected config.
- In `_load_tf_weights` (`nn.py:702-707`), compare against
  `model.conv_dna.conv.in_channels` instead of the hard-coded 170. Keep the
  raise — it still guards anyone hand-building a module.
- Refresh the two docstrings that assert 170 (`nn.py:33-38`, `nn.py:577-590`).

Keep the allowlist. Accepting any width would turn today's loud "these are
extracted weights, not a real checkpoint" failure into quietly wrong numbers.

### 4. `src/yeastbench/adapters/shorkie_hong.py` — resolve `targets.txt` per variant

The fallback at `shorkie_hong.py:117-121` hard-codes
`data/models/shorkie/targets.txt`. Seed a fresh Modal volume with only
`--model shorkie_random_init` and that file is absent, at which point Hong
silently degrades to primary-only (`shorkie_hong.py:213-215`) and the two models
get scored on different metric sets — Hong reports the IntTrain-fitted ρ as a
co-primary metric.

Add a `_default_targets_path(params_path)` helper that prefers `targets.txt`
beside the checkpoint's `params.json` and falls back to the existing
repo-anchored path, and call it from `from_checkpoints` when `targets_path` is
None. For the pretrained model both branches name the same file, so nothing
moves. Don't route this through `model_config` — that dict goes to every adapter
in the run and the other eight signatures don't take it.

### 5. Registry, manifest, lock, configs

**`src/yeastbench/registry.py:328`** — one line:

```python
"shorkie_random_init": _build_shorkie,
```

`_build_shorkie` dispatches on the task's protocol, never on the model name, so
all nine adapters are reused as they stand. The name has no `__`, so
`compare._DIR_RE` (`compare.py:41`) splits the result directories correctly.

**`src/yeastbench/data/manifest.py`** — a new `InputArtifact` after the `shorkie`
block (`:258-289`), `dest="data/models/shorkie_random_init"`, mirror base
`.../shorkie_models/shorkie_random_init/`, the same `f{i}/model_best.h5`
remapping. Upstream ships no `targets.txt` for this variant and both models were
trained on the same 5,215-track sheet, so point that member at the pretrained
copy by absolute URL (absolute URLs bypass the base — `data/backends.py:58-61`).
Do not use `requires=("shorkie",)`; that drags 460 MB along for a 781 KB file.

**Lock it before running anything.** Until the lock entry exists, `build_plans`
skips the artifact (`fetch.py:137-140`), the preflight reports "ready", and the
run dies at model load.

```bash
# fetch once so `data lock` has local files to hash
mkdir -p data/models/shorkie_random_init/checkpoints
BASE=https://storage.googleapis.com/seqnn-share/shorkie_models
for i in 0 1 2 3 4 5 6 7; do
  curl -fL "$BASE/shorkie_random_init/f$i/model_best.h5" \
       -o "data/models/shorkie_random_init/checkpoints/f$i.h5"
done
curl -fL "$BASE/shorkie_random_init/params.json" -o data/models/shorkie_random_init/params.json
curl -fL "$BASE/shorkie/targets.txt"             -o data/models/shorkie_random_init/targets.txt

uv run ybench data lock   --tasks  shorkie_random_init
uv run ybench data verify --models shorkie_random_init
```

**Configs** — add a `shorkie_random_init` run block to `configs/default.yaml`
(after the Shorkie block at `:72-96`, same eight tasks, `batch_size: 32`,
`use_rc: true`), `configs/brooks.yaml` (`:25-38`) and `configs/meneu.yaml`
(`:30-43`), each pointing at `data/models/shorkie_random_init/`.

Adding it to the canonical config does not re-run the other models. `--model
shorkie_random_init` selects only the new pairs, and the auto-compare afterwards
deliberately walks the **unfiltered** config's `out_dir`, so it pairs the new
results against whatever peers are already on disk (`cli.py:197-201`). There is
no skip-if-exists logic in the runner, but with the filter there is nothing to
skip.

### 6. Tests (all CPU-only)

- `tests/test_registry.py` — the new name is in `MODELS` and maps to the same
  factory as `shorkie`.
- `tests/test_data_manifest.py` — the artifact is declared and locked; the
  registry-coverage gate at `:44` stays green. Add a `@pytest.mark.network`
  fetch of the 2.7 KB `params.json` against the lock, mirroring `:184-198`.
- `tests/test_shorkie_input_channels.py` (new) — default config builds a
  170-wide conv with expansion on; an injected 4 builds a 4-wide conv with
  expansion off; the injection leaves the caller's dict untouched; an unknown
  width raises and names both valid widths.
- `tests/test_hong_igr.py` — `_default_targets_path` prefers the sheet beside
  `params.json` and falls back when absent.

### 7. Docs

A row in the README Models table, and a `Shorkie_Random_Init` column in the
per-benchmark tables in `docs/results.md`. House style per `CLAUDE.md`: say what
changed, don't editorialise.

## Verification

**Prove the pretrained path is bit-for-bit unchanged.** This is the gate. The
change adds one `if` and pins a value that was already the default, so it should
hold exactly — but check it rather than argue it:

```bash
git worktree add /tmp/ybench-dev dev
# same probe script in each tree: torch.manual_seed(0), torch.set_num_threads(1),
# load data/models/shorkie/checkpoints/f0.h5 via from_tf_checkpoint,
# forward one fixed 16,384 bp window, np.save the output
python -c "import numpy as np; a=np.load('/tmp/before.npy'); b=np.load('/tmp/after.npy'); \
           print('bit-identical:', np.array_equal(a, b))"
```

`np.array_equal`, not `allclose`. Then commit a small regression fixture — the
T0-track mean per bin is 896 floats (~3.6 KB), where the full output would be
18.7 MB — and assert it with `atol=0`, skipping when the checkpoint is absent.

**Smoke-test the new model on CPU** before spending GPU time. Load one fold,
confirm `conv_dna.conv.in_channels == 4` and that species expansion is off, then
run the cheapest real pair:

```bash
uv run ybench run -c configs/default.yaml -m shorkie_random_init -t hong_igr --device cpu
```

Hong is the right smoke task: fewest loci of the eight, and the only one that
reads `targets.txt`, so it exercises step 4. Check that
`results/default/shorkie_random_init__hong_igr/summary.json` carries the
IntTrain-fitted metric rather than the degraded primary-only shape.

**Run the ten tasks on Modal.**

```bash
uv run ybench modal data get -c configs/default.yaml -m shorkie_random_init
uv run ybench modal run      -c configs/default.yaml -m shorkie_random_init --gpu A10 --detach
uv run ybench modal run      -c configs/brooks.yaml  -m shorkie_random_init --gpu A10 --detach
uv run ybench modal run      -c configs/meneu.yaml   -m shorkie_random_init --gpu A10 --detach
uv run ybench modal pull     -c configs/default.yaml --out results
uv run ybench modal pull     -c configs/brooks.yaml  --out results
uv run ybench modal pull     -c configs/meneu.yaml   --out results
```

The image bakes the working tree, so uncommitted edits ship and it rebuilds
itself. Expect roughly 1–2 GPU-hours; `rafi_mpra_marginalized` (~1.56M windows)
and `cuperus_utr` (~501k) dominate, everything else is minutes.

**Sanity-check against the paper.** The random-init model should lose to
pretrained Shorkie on the eQTL tasks — that is the published result, and it is
the cheapest check that the port is right. If it wins, suspect the port before
believing it.

## Risks

- **Brooks headline can move.** `BrooksScrambleBenchmark.compare_plot`
  (`benchmarks/brooks.py:978-981`) intersects `sample_ids` across every model
  present and reports on that shared cohort. A third model can shrink the
  intersection. Both Shorkie variants share window geometry so the sets ought to
  be identical — diff `shared_cohort["n"]` before and after, and if it shrinks,
  send the random-init Brooks run to its own `out_dir` rather than restating the
  published Shorkie-vs-Yorzoi numbers.
- **Plot colours shift.** `model_color` indexes `MODEL_COLORS` by position in
  the sorted model list (`benchmarks/base.py:106-124`). Sorted order becomes
  `[shorkie, shorkie_random_init, yorzoi]`, so Yorzoi changes colour in every
  regenerated compare figure. Cosmetic, but it will show up in docs diffs.
- **Merge conflict with `exoshorkie/distill`.** That branch rewrites
  `_expand_species_encoding`, `forward`'s tail and `from_tf_checkpoint` — the
  same lines. Keeping this edit config-shaped keeps the conflict mechanical.
  Worth a line in the PR description.
- **Fold correspondence is unproven.** The random-init `f0..f7` are assumed to be
  the same eight CV folds as the pretrained model's. The naming and the shared
  experiment root say so, but nothing in the bucket proves it. Compare ensembles,
  not individual folds, and don't claim per-fold correspondence in the writeup.
