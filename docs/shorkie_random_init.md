# Benchmark Shorkie trained from random initialization

Reviewed against `0e37d14` on 2026-11-06. This branch was fast-forwarded from
`7b090ee` to include the updated input-storage workflow and default config.

## Scope

Add the authors' supervised, eight-fold `Shorkie_Random_Init` ensemble and score
all eleven tasks with the same adapters, inputs, track selection, and reverse-
complement averaging as pretrained Shorkie. This is a model trained without
language-model pretraining, not an untrained network. Compare ensembles; matching
fold numbers alone does not establish corresponding train/test splits.

The published weights are at
`https://storage.googleapis.com/seqnn-share/shorkie_models/shorkie_random_init/`.
Eight checkpoints total 454,966,368 bytes. Upstream's manifest lists their MD5s:
<https://github.com/calico/shorkie-paper/blob/main/data/manifest.json>.

## Changes from the original plan

- `configs/default.yaml` now covers all eleven tasks, including YTK, Brooks,
  and Meneu. Coverage runs and Chen use batch size 4; the other eight use 32. Keep the
  standalone Brooks/Meneu configs consistent, but do not run these tasks twice.
- Inputs now use `InputArtifact`/`InputKind` from `data.artifacts`, with selection
  in `data.manifest`. The lock workflow supports immutable HF revisions and
  rejects missing lock entries. Use `data lock --artifacts`, not the old
  description of silently skipping unregistered checkpoints.
- Use the cluster GPUs as requested. On this host GPUs are visible outside the
  sandbox; four A6000s are shared with active jobs. Limit CPU threads and retain
  separate task logs. Do not stop or change other jobs.
- Local sibling checkouts contain weights, task inputs, and some previous
  results. Copy inputs into this checkout and verify their committed checksums;
  do not modify shared inputs. Retain old result metadata when importing peers.
- Preserve original config bytes before edits for optional recovery of Modal
  results by their old hashes. No cloud compute is required for the new runs.
- The pretrained manifest still uses the old bucket path. Correct it in this
  change; no branch switching or historical cherry-pick is needed.
- Initial measured throughput is about 3 Rafi sequences/s and 17 Cuperus
  sequences/s on the shared GPUs: roughly 6 and 8 hours respectively, excluding
  initialization and contention changes. The old 1–2 GPU-hour estimate is withdrawn.
- Chen's batch size counts variants, each expanded over 20 host contexts. The
  inherited batch size 32 exhausted A6000 memory. Both Shorkie variants now use
  the adapter default of 4 for this task.
- Cuperus needs `construct.json`, which the previous manifest omitted. Rebuild
  it with `scripts/cuperus/build_construct.py` from the locked reference, include
  its checksum in the input lock, and commit the small reporter definition.
  Its bytes match the existing local reference exactly. It is not present on
  the current HF mirror, so source checkouts carry this file directly.
- Rafi's adapter already defaults to bfloat16 autocast and compilation. Other
  Shorkie adapters use float32. Preserve these existing per-task defaults;
  the CPU regression check exercises float32.

## Implementation

1. Inspect real checkpoint kernels and both parameter files. The previous audit
   found first-convolution shapes `(11, 170, 96)` and `(11, 4, 96)`, despite equal
   model config blocks. Preserve a deterministic pretrained CPU output before
   modifying the loader.
2. Read input width from the H5 kernel shape before building `ShorkieModule`.
   Accept only 4 or 170 channels. Copy the config before inserting `in_channels`
   since the wrapper reuses it across folds. Expand species channels only for
   the 170-channel model; keep weight-shape validation against the built module.
   A width allowlist cannot distinguish a genuine four-channel checkpoint from
   an incorrectly extracted one: use the published, checksummed checkpoints.
3. Resolve Hong's `targets.txt` beside `params.json`, retaining the existing
   pretrained fallback. Fetch the same upstream 5,215-track metadata into the
   new model directory without depending on all pretrained checkpoints.
4. Register `shorkie_random_init` with `_build_shorkie`. Add its HTTP input
   artifact, ten-file checksum inventory, and config entries. Reuse all ten
   adapters (eleven tasks share ten protocols/adapters).
5. Test channel inference across supported H5 layouts, unknown widths,
   nonmutation of config, conditional species expansion, registry/config
   coverage, metadata resolution, and manifest/lock completeness.
6. Compare pretrained outputs before/after with `np.array_equal` on the same
   CPU/runtime. Smoke-test random-init on Hong (including its fitted metrics)
   and YTK, then execute every remaining full task on cluster GPUs.

## Execution and evidence

Use this checkout's source (`PYTHONPATH=src`) when using a shared Python
installation. Example commands after acquiring and locking the new weights:

```bash
PYTHONPATH=src .venv/bin/python -m yeastbench.cli data lock --artifacts shorkie_random_init
PYTHONPATH=src .venv/bin/python -m yeastbench.cli data verify -c configs/default.yaml -m shorkie_random_init
PYTHONPATH=src .venv/bin/python -m yeastbench.cli run -c configs/default.yaml -m shorkie_random_init --dry-run
PYTHONPATH=src .venv/bin/python -m yeastbench.cli run -c configs/default.yaml -m shorkie_random_init --gpu 0
```

For parallel execution, select a distinct task per worker and write individual
logs. Record source changes, input lock, config hash, device, runtime, and full
sample counts. Preserve predictions before comparing or plotting.

Update README model support and report only measured results in `docs/results.md`.
Compare Brooks on identical sample IDs and report any cohort change. Existing
plot colours depend on sorted model position and may change when adding a model.
Unexpected performance differences warrant checking predictions and sample
selection; agreement with the expected model ranking is not a correctness test.

## Status

Implementation and all eleven full benchmark runs completed.

- All eight random-init checkpoints match upstream MD5s; all have four input
  channels. Their model config equals pretrained Shorkie's model config.
- The entire pretrained fold-0 output on the deterministic CPU probe is
  bit-identical before/after (shape `1 × 896 × 5215`). A compact T0-track mean
  regression fixture is committed with the test.
- Input verification passed (54 files after adding Cuperus's reporter definition).
- Full CPU suite: 499 passed, 9 skipped, 6 deselected. Both real parameter-download/checksum tests also pass.
- The initial broad suite had 497 passes, 9 skips, and one failure due to a
  missing YTK build-provenance sidecar. Geometry itself passed. The test now
  checks that optional provenance only when present, so fresh installations
  of the declared runtime inputs pass the same geometry checks.
- Run logs and execution provenance are under `tmp/random-init/`; completed
  results are under `results/default/shorkie_random_init__<task>/`.
- `tmp/random-init/finish.py` monitors active jobs and retries failed tasks when
  GPU memory is available. It updates `tmp/random-init/status.json` and the
  results appendix as tasks finish. `tmp/random-init/completion.json` is written
  only after all eleven tasks have finished, their raw outputs reproduce the
  saved reconstructable metrics (Brooks does not persist its secondary shape
  diagnostic arrays), and the final comparison has been regenerated.
- Regenerate the report with
  `PYTHONPATH=src .venv/bin/python scripts/docs/report_shorkie_random_init.py`.
  Pending tasks stay explicit; missing pretrained raw results remain blank.

All eleven completed-task summaries and evaluation timings are recorded in `docs/results.md` and `docs/results/shorkie_random_init.json`. Raw predictions, plots, and run metadata remain under `results/default/`.
