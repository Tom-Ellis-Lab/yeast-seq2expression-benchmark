# Plan: publish results to a HuggingFace dataset

Status: proposal, for review. Nothing here is implemented yet.

## Why

Input data is already mirrored to HuggingFace and pulled with `ybench data get`,
checksum-verified against a committed lock. Results have no such path. They are
gitignored (`.gitignore:212`; `git ls-files results` returns nothing), so the
only way to see a number is to own a GPU and rerun the suite.

What reaches the public today is copied by hand: rounded figures typed into
`docs/benchmarks/*.md`, a scoreboard maintained by hand in `docs/results.md`, and
17 figures copied into the committed `img/results/` by
`scripts/docs/collect_result_figures.py` — a script whose paths have already
drifted out of date.

This closes the open item at `docs/ROADMAP.md:344-347`: *"Host the precomputed
results for download, so the suite is usable without re-running on GPU."*

Results go to their own repo, `tom-ellis-lab/ybench-results`, separate from
`tom-ellis-lab/yeast-seq2expression-data`, so nobody pulling inputs drags down
prediction arrays, and nobody after the numbers pulls 690MB of genomes.

## Decisions taken

- Repo: `tom-ellis-lab/ybench-results`, `repo_type="dataset"`. HuggingFace only.
  The GCS bucket is requester-pays, the wrong toll for an archive whose point is
  to save people a GPU. The `Mirror` type leaves the door open.
- Payload: everything a run writes — `summary.json`, `run_metadata.json`, `.npy`,
  `.npz`, `.tsv`, `.png`, `.svg`, and `compare/`.
- One folder per run, plus a curated `published/` folder holding what the docs
  cite. Promotion into `published/` is opt-in; the default leaves it alone.
- The CLI works both directions: push a run, pull results into `results/`.

## Where this departs from the data path

The shape follows `ybench data` on purpose — same backend, same
dry-run-until-`--yes`, same ambient auth. Three differences are deliberate, and
the module docstring should say so, because otherwise they read as oversights.

**Checksums live in the repo, not in the wheel.** Data pins integrity with
`manifest.lock.json` shipped inside the package (`data/lock.py:31`), which works
because the input set is fixed at release time. The results set grows on every
publish, so a wheel-shipped lock would be stale the moment it was cut. Results
checksums belong in `runs/<run_id>/manifest.json` on the hub, next to the bytes
they describe.

**No GCS mirror.** See above.

**A generated dataset card and `.gitattributes`.** The data repo has neither:
`HfBackend.upload` pushes only locked artifact files, and `create_repo`
(`backends.py:147`) is called without `license=`. Do not repeat that here.
Backporting a card to the data repo is a separate one-line change.

## Remote layout

```
tom-ellis-lab/ybench-results               (repo_type="dataset")
├── README.md                  generated card: frontmatter + prose + run index
├── .gitattributes             LFS rules
├── index.json                 every run and every published entry
├── index.csv                  the same, flattened, for the hub viewer
├── runs/
│   └── 20260731T1242Z-meneu-b5d4f5287172/
│       ├── manifest.json      file list, sha256, size, provenance
│       ├── config/meneu.yaml  the exact config bytes
│       ├── shorkie__meneu_foreign_dna/…    verbatim
│       ├── yorzoi__meneu_foreign_dna/…     verbatim
│       └── compare/…                       verbatim
└── published/
    ├── manifest.json          task -> {run_id, models, files, promoted_at}
    ├── leaderboard.csv        task,model,metric,value across every promoted task
    └── meneu_foreign_dna/
        ├── shorkie__meneu_foreign_dna/…
        ├── yorzoi__meneu_foreign_dna/…
        └── compare/per_task/meneu_foreign_dna/…
```

Each choice earns its place:

- **`runs/<id>/` mirrors the local `out_dir` one-for-one**, so pulling a run is a
  straight tree copy into `results/<group>/` and every path shape the docs
  already use survives. Nothing under `runs/` is ever rewritten. That
  append-only property is what makes a citation stable.
- **`runs/` is flat, not `runs/<group>/<id>/`.** The group is already in the id,
  and a flat directory sorts chronologically in the hub's file browser, which is
  how a person scans it.
- **`published/` is keyed by task, not by group.** The curated set is a per-task
  mix across `results/{default,brooks,meneu}` — that is the observed state, not a
  hypothetical. Task keying is also exactly the granularity promotion needs, so
  replacing one task is deleting one folder rather than diffing a group.
- **`index.json` sits at the root** so a consumer resolves the whole archive with
  one `hf_hub_download("index.json")` and no repo listing.

### Run id

The publishable unit is an `out_dir` snapshot, not a single `(model, task)` pair.
`compare/` is written per `out_dir` and the docs cite it, so a per-pair id would
split the compare outputs from the pairs they compare.

```
<YYYYMMDD>T<hhmm>Z-<group>-<config_hash>

20260731T1242Z-meneu-b5d4f5287172
20260731T1220Z-default-a4892ae9fa49
```

- **Timestamp**: the largest `timestamp_utc` across every `run_metadata.json` in
  the snapshot, truncated to the minute — not the publish time, which is not
  provenance. Deriving it from run-time facts makes the id a function of the
  bytes on disk.
- **Group**: `out_dir` relative to `results/`. Reuse the guard at
  `modal/plan.py:71-79`, which already requires `out_dir` to live under
  `results/`.
- **Config hash**: all 12 characters, matching what `cli.py:155` prints and what
  `modal/app.py:160` namespaces the Modal volume by. Consistency with what the
  maintainer already sees beats four saved characters.

Compute it at publish time, in `run_id_for(snapshot)`. Not at run time: an
`out_dir` is filled incrementally, so no single `ybench run` knows the final
contents; an id written at run time would need a new file in every result dir;
and the six runs already on disk have no id but must still be publishable.

Two awkward cases fall out cleanly. Adding `yorzoi` to a folder that held only
`shorkie` moves the largest timestamp forward, so the second publish gets a new
id and the first snapshot stays in the archive untouched — correct, because they
are two real snapshots of the same group. Publishing the same config twice in one
day gives different minutes, so different ids.

Minute resolution is not injective, so compute a content digest alongside:

```
content_sha = sha256 over sorted "<relpath>\t<sha256>\t<size>\n" lines
```

reusing `sha256_file` (`data/lock.py:41`). If `runs/<id>/manifest.json` already
exists remotely, an identical `content_sha` makes the publish a no-op — print
"already published" and exit 0. A different one appends `-2`, `-3` and warns
loudly, because a collision there usually means someone edited files in place.

`--run-id <slug>` overrides for a memorable archive name (`paper-v1-brooks`),
validated against `^[a-z0-9][a-z0-9._-]{2,63}$` and rejected if it contains `/`
or `..`. If no `run_metadata.json` carries a parseable timestamp, refuse and
demand `--run-id`. Do not fall back to file mtimes: `modal volume get` does not
reliably preserve them, so an mtime-derived id would be quietly wrong.

### How `published/` works

Copy the bytes and record the pointer; do not write pointer files alone.
HuggingFace deduplicates LFS blobs by content, so a 16MB `.npz` at a second path
costs one pointer in git rather than another 16MB. Pointer-only would force every
consumer to resolve indirection, and a reader clicking `published/brooks_scramble/`
on the hub would find an empty folder.

`published/manifest.json` carries the traceability the bytes lack:

```json
{
  "version": 1,
  "tasks": {
    "brooks_scramble": {
      "run_id": "20260731T1243Z-brooks-1b616322c394",
      "models": ["shorkie", "yorzoi"],
      "promoted_at": "2026-07-31T14:02:11Z",
      "ybench_version": "0.1.0",
      "cited_by": ["docs/benchmarks/brooks_scramble.md"],
      "files": {"shorkie__brooks_scramble/summary.json": {"sha256": "…", "size": 412}}
    }
  }
}
```

A verifier can then prove `published/<task>/x` is byte-identical to
`runs/<run_id>/x`.

Promotion replaces one task inside one commit:
`CommitOperationDelete(path_in_repo=f"published/{task}", is_folder=True)`, then
the adds. Other tasks sit under sibling prefixes and are untouched — the
"update one task without clobbering the others" requirement is satisfied by the
layout rather than by careful diffing.

| Flag | Effect |
| --- | --- |
| *(none)* | Upload to `runs/<id>/` only. Never touches `published/`. **Default.** |
| `--promote` | Also promote every task in the snapshot |
| `--promote-task a,b` | Promote only those tasks; implies `--promote` |

Four guards, each earning its keep:

- Refuse `--promote-task X` when `X` is not in the snapshot. Typo protection.
- Refuse to promote a task whose pair dirs lack `summary.json`. Partial run.
- Refuse when the snapshot has fewer models for a task than
  `published/manifest.json` records, unless `--force`. Replace-semantics invite
  exactly one mistake: run one model, promote, silently lose the other.
- Assert task names match `^[a-z0-9_]+$` and reject `compare`, which would
  collide with the directory of the same name.

Rebuild `leaderboard.csv` on every promotion by re-reading the promoted
`summary.json` files through `compare._flat_scalar_metrics` (`compare.py:94`) and
writing the header `compare.py:131` already writes. Reuse that, rather than
implementing "which metrics are scalar" twice.

Promoting a run that is already archived (`ybench results promote --run-id …`)
cannot copy server-side: `CommitOperationCopy` is LFS-only and does not work
across repos, so the JSON, CSV and SVG half of the payload would need downloading
anyway. Make `--from-local <dir>` the primary path — verify each sha256 against
`runs/<id>/manifest.json`, then upload — and fall back to download-then-reupload.
The payloads are a few MB per task; this is not worth being clever about.

## Provenance

The test for inclusion: **can a reader tell what code produced these numbers, on
what inputs?** Two gaps break that. The rest is comfort.

### Fix first — these block a trustworthy archive

**`git_commit` is `"unknown"` in four of the six runs on disk.** Root cause,
verified: `.git` in this checkout is an 84-byte gitfile reading
`gitdir: /home/tds122/yeast-seq2epxression-benchmark/.git/worktrees/yeast-chen-rerun`
— a path outside the tree that `modal/app.py` ships. So `add_local_dir` copies a
pointer to nothing, `git rev-parse HEAD` fails in the container, and
`cli.py:52` returns `"unknown"`. The `safe.directory` setting cannot help.

Resolve the commit where git actually works — on the host — and carry it in:

- `modal/plan.py:23-47` — add `git_commit: str` to `RemotePlan`, resolved in
  `build_plan`.
- `modal/app.py:134-142` — accept it on `run_benchmark`; pass it to the
  subprocess at `:178-184` as `YBENCH_GIT_COMMIT`. `modal/cli.py:129-137` threads
  it through.
- `cli.py:45-52` — `_git_commit()` reads `YBENCH_GIT_COMMIT` first, then shells
  out with `cwd` set to the package root rather than the process cwd, then falls
  back to `"unknown"`.

Then enforce it at publish: refuse a snapshot containing `"unknown"` unless
`--allow-unknown-commit`. Without the guard the fix rots. The six runs on disk
today need either a rerun or the escape hatch — say which in the PR.

Record **all** distinct commits as `code_versions` in the run manifest and warn
when there is more than one. `results/default` is a live example: the
`cai__chen_synonymous` pair ran at `06290f7b`, the two `kita_eqtl` pairs ten days
later. Incremental fill across commits is legitimate, so record it rather than
reject it.

**Archive the config bytes, and check them.** Make `--config` the primary
selector for publish, mirroring `ybench data get --config`. That buys three
things at once: `load_config` resolves `out_dir`; the exact YAML bytes get
archived at `runs/<id>/config/<name>.yaml`; and the publisher can assert
`sha256(bytes)[:12] == config_hash` for every `run_metadata.json` in the
snapshot. The assertion is worth more than the archived bytes — it proves the
snapshot is internally consistent.

### Take at the same time — two lines each

- **`ybench` version.** `src/yeastbench/__init__.py` is empty. Add `__version__`
  from `importlib.metadata`, and add `"ybench_version"` to the dict at
  `cli.py:91-108`. Note the distribution name is misspelled at
  `pyproject.toml:2` (`yeast-seq2epxression-benchmark`); keeping it in one place
  stops the typo spreading.
- **`config_path` leaks container paths** — it currently records
  `/repo/configs/default.yaml`. Store it relative to the repo root at
  `cli.py:97`. Nothing in the repo reads the field, so this breaks nothing.
- **Hardware.** `describe_device()` (`hardware.py:37`) returns a dataclass and is
  already called at `cli.py:157`; add `asdict(...)` to the metadata. Be honest
  about why — it does not make the numbers more trustworthy, it answers "why did
  this take four hours" and "did it quietly fall back to CPU".

### Leave alone

- **Seeds.** Evaluation is deterministic: forward passes, no sampling, no
  augmentation. Threading a seed through `Config` into every adapter is a real
  change with a real surface and buys nothing until something stochastic exists.
  Record `"seed": null` and state in the card that v1 evaluation is
  deterministic. If it turns out not to be, that is a bug to fix, not a field to
  add.
- **`scripts/docs/collect_result_figures.py`.** Its list is stale in three places
  (expects `shared_tier1.png`, disk has `shared_lfc.svg`; expects
  `per_task/chen_tdh3/`, Chen is now one task; expects a doubled
  `meneu_foreign_dna_shorkie` suffix). Do not repair the list. Once `published/`
  exists the script's whole job is `ybench results get --published`. Retarget or
  delete it alongside the docs edits.

## Modules

New package `src/yeastbench/results/`, parallel to `data/`:

| New file | Analogue | Holds |
| --- | --- | --- |
| `results/snapshot.py` | `lock.py` | `Snapshot`, `scan(out_dir)`, `run_id_for()`, `content_sha()`, `problems()`. Pure, no network. |
| `results/store.py` | `fetch.py` + `manifest.py` | `RESULTS_REPO`, mirror construction, index and manifest IO, `plan_publish()` / `apply_publish()`, `plan_get()` / `apply_get()` |
| `results/card.py` | *(none — this is the gap)* | dataset-card template and `.gitattributes` text |
| `results/cli.py` | `data/cli.py` | the Typer app |

The plan/apply split mirrors `build_plans` (`fetch.py:118`) against `run_get`
(`fetch.py:188`) — same shape, same offline testability.

**Reuse directly:** `sha256_file` (`lock.py:41`); `_DIR_RE` (`compare.py:41`) for
pair-dir parsing, imported rather than copied, so there is one definition of how
a pair dir is named; `Mirror` (`manifest.py:55-74`) built with
`prefix=f"runs/{run_id}/"`, whose `remote_for()` then does the path joining;
`default_data_root()` (`fetch.py:45`) for the repo-root anchor; `_human()`
(`fetch.py:444`) for byte counts; `load_config` (`config.py:52`);
`_flat_scalar_metrics` (`compare.py:94`); and the `.part`-then-`os.replace`
sequence at `fetch.py:224-235` for the pull direction.

**Add to `HfBackend`** (`backends.py:83-157`):

1. `snapshot(mirror, out_dir, allow_patterns=None)` wrapping `snapshot_download`.
   A new method is needed: `warm()` (`:121-133`) is cache-only, passes no
   patterns and no `local_dir`, and its one caller wants exactly that. Do not
   widen `warm`.
2. `deletes: list[str] | None = None` on `upload()` — a keyword, backward
   compatible with the call at `fetch.py:322`, so promotion's delete and its adds
   land in one commit.
3. `list_files(mirror)` over `HfApi().list_repo_files`, for collision detection
   and delete planning.

**On the Protocol asymmetry.** `Backend` (`backends.py:33-40`) declares only
`available()` and `fetch()`; `upload` is not on it, and `GcsBackend.upload` has a
different signature, which is why `fetch.py:320-327` branches on backend kind. Do
not fix that globally — it is a refactor of a working path. Narrow instead: the
results store is HuggingFace-only by construction, so `results/store.py` types
its backend as `HfBackend` concretely and never calls `get_backend()`. One
comment explains the narrowing. If a second results mirror is ever wanted, that
is when the Protocol earns an `Uploader` split.

One `create_commit` per run is right. The largest snapshot here is 47 files;
`UPLOAD_BATCH_MAX_NUM_FILES = 256` only chunks the preupload mode query, and the
commit itself is atomic.

## CLI

Mount at `ybench results`, in `cli.py:30` right after the `data` sub-app.
`ybench data` is inputs, `ybench results` is outputs — symmetric, and it keeps
`publish` unambiguous against the existing `ybench data publish`. A top-level
`ybench publish` would read as "publish the data".

```
ybench results get  [--published | --run-id ID | --latest]
                    [--task/-t NAME] [--out results] [--force] [--dry-run]

ybench results publish  --config configs/default.yaml
                        [--out-dir DIR] [--run-id ID]
                        [--promote | --promote-task a,b]
                        [--allow-unknown-commit] [--allow-partial] [--force]
                        [--message/-M "..."] [--yes]

ybench results promote  --run-id ID [--task NAME] [--from-local DIR]
                        [--force] [--message/-M "..."] [--yes]

ybench results list    [--json]
ybench results status  [--out results] [--json]
```

`publish` follows `data/cli.py:298-337` line for line: dry run unless `--yes`,
`-M` for the message, `typer.Exit(code=1)` on failure, the module-level `_echo`,
and the closing `"re-run with --yes to upload."`. `--promote` defaulting to
false is the requested "do not overwrite by default", and the flag says what it
does rather than what it withholds.

`get` has no `--yes` — pulling is not destructive — but it must refuse to
overwrite a file whose sha256 differs from the manifest unless `--force`,
reusing the state logic at `fetch.py:98-101`. `--run-id X` lands at
`results/<group>/…` so the tree is drop-in identical to a local run.

Mount the sub-app unconditionally and fail at call time with the existing
message, "huggingface_hub not installed; run `uv sync --extra data`"
(`backends.py:101`), never at import. Keep every `huggingface_hub` import lazy
and inside the method, as `HfBackend` already does.

## HuggingFace specifics

### `.gitattributes`

Ship one in the first commit. The hub's server-side default already tracks
`*.npy` and `*.npz` as LFS, and the preupload endpoint forces LFS for anything
over 10MB regardless — so the two 16.3MB `stitched_pred.npz` files are handled
either way. The reason to ship a file is the opposite direction: opting small
text **out** of LFS so the hub renders it in the browser.

```gitattributes
*.npy filter=lfs diff=lfs merge=lfs -text
*.npz filter=lfs diff=lfs merge=lfs -text
*.png filter=lfs diff=lfs merge=lfs -text
# Keep the readable surface as plain blobs so the Hub renders it in-browser.
*.json -filter -diff -merge text
*.csv  -filter -diff -merge text
*.tsv  -filter -diff -merge text
*.md   -filter -diff -merge text
*.svg  -filter -diff -merge text
```

Last match wins, so the negations must come after the positives, and never write
a blanket `* filter=lfs`. Keeping SVG plain is the line that pays: the compare
plots are SVG (`shared_per_sample.svg` is 295KB) and rendering them in the file
viewer is worth real money. Note the 10MB server rule overrides `-filter`, so a
`summary.json` that grew past 10MB would silently become LFS — plausible for the
Brooks `compare/per_task/*/summary.json`, which embeds a full `sample_ids` list.

### Dataset card

Declaring `configs:` is not a nicety. It is what stops the viewer choking.

```yaml
---
license: cc-by-4.0
pretty_name: yeast-seq2expression benchmark results
tags: [yeast, gene-expression, benchmark, results, zero-shot]
configs:
  - config_name: leaderboard
    data_files: published/leaderboard.csv
    default: true
  - config_name: runs
    data_files: index.csv
---
```

The `.npy` and `.npz` files carry no risk by their mere presence — the viewer
never attempts an unsupported format unless a config's `data_files` glob names
one. The real risk is having no `configs:` key: the viewer then auto-detects and
tries to read every `.json` in the repo as JSON-Lines. Our `summary.json` files
are pretty-printed multi-line objects whose keys are entirely task-specific
(`cli.py:83-88` splats `task.summary_dict(results)`), and `run_metadata.json` has
another schema again. Auto-detection across thousands of those produces a schema
mismatch and a red banner on the dataset page. Two explicit configs fix it, and
they render the same leaderboard `README.md:48-59` and `docs/results.md` keep by
hand. Never glob `published/**` or `runs/**` into `data_files`.

### Repo creation, auth, revisions

`create_commit` handles LFS on its own — it asks the hub's preupload endpoint per
file and uploads LFS blobs out of band. No `git-lfs` binary, nothing for us to
do. It also skips no-op commits, which reinforces publish idempotency for free.

On a first publish, seed `README.md`, `.gitattributes`, `index.json` and
`index.csv` **in the same commit as the run**, so the repo is never live in an
undocumented state.

Keep the ambient-auth model — no `token=` anywhere, consistent with `data`, and
it fails with HuggingFace's own clear message when `HF_TOKEN` is unset. Document
`huggingface-cli login` in the maintainer note.

Do not add `revision=` pinning to the download path. `runs/<id>/` is append-only
and every download is checksum-verified against the run manifest, which is
strictly stronger than a revision pin. `published/` is meant to move.

## Files to touch

- New: `src/yeastbench/results/{__init__,snapshot,store,card,cli}.py`
- `src/yeastbench/cli.py` — mount the sub-app (`:30`); `_git_commit` (`:45-52`);
  the metadata block (`:91-108`)
- `src/yeastbench/data/backends.py` — `snapshot()`, `list_files()`, `deletes=`
- `src/yeastbench/modal/{plan,app,cli}.py` — carry the host git commit
- `src/yeastbench/__init__.py` — `__version__`
- `docs/cli.md` — a `ybench results` section beside "Getting the data" (`:114-158`)
- `docs/results.md:14-17` — already tells readers the per-metric tables "live next
  to the runs in `results/<group>/compare/summary.md`", a path they cannot reach
  today. Add the fetch command there; it is the best hook in the docs. Fix the
  stale `results/brooks/compare__shared/` at `:163`.
- `docs/benchmarks/*.md` — repoint the nine `Artifacts:` lines at
  `published/<task>/` plus the run id
- `docs/ROADMAP.md:344-347` — tick the item
- New: `tests/test_results_{snapshot,plan,backend,cli,card}.py`

## Verification

All of the following runs offline. None of it gets the `network` marker.

- **`test_results_snapshot.py`** — fabricate `<model>__<task>/{summary,run_metadata}.json`
  in `tmp_path` as `tests/test_compare.py:36` already does. Assert the id format;
  that the id is stable across two calls on the same bytes; that it moves forward
  when a later-timestamped pair is added; that `content_sha` changes when one byte
  changes; that `problems()` catches a missing `summary.json`, a mixed
  `config_hash`, and `git_commit == "unknown"`; that several distinct commits
  warn rather than fail; and that the `--run-id` validator rejects `../escape`.
- **`test_results_plan.py`** — `plan_publish()` against a fabricated index. A new
  run gives N adds; a run already in the index with a matching `content_sha`
  gives zero operations; `--promote-task brooks_scramble` emits a delete for
  `published/brooks_scramble` and never for `published/meneu_foreign_dna`;
  promoting a task absent from the snapshot records a failure. Plain dict and
  list assertions, following `tests/test_data_backends.py`.
- **`test_results_backend.py`** — monkeypatch `HfApi` with a recorder; assert
  `upload(..., deletes=[…])` emits `CommitOperationDelete(is_folder=True)` before
  the adds in one `create_commit`, and that `snapshot()` passes the right
  `allow_patterns`.
- **`test_results_cli.py`** — `CliRunner`, following `tests/test_modal.py:90-99`.
  `ybench results --help` lists the subcommands; `publish` without `--yes` prints
  a plan and uploads nothing, with the backend monkeypatched to raise if called.
- **`test_results_card.py`** — assert the `.gitattributes` text keeps `*.json`,
  `*.csv` and `*.svg` out of LFS and puts `*.npy`/`*.npz` in; assert the README
  frontmatter parses as YAML and names `published/leaderboard.csv`. A typo there
  breaks the hub's front page silently.

One real round trip, maintainer-run, marked `network` and `integration`. Use
`meneu`: only two pairs, and its 16.3MB `.npz` files exercise the LFS path.

```bash
uv run ybench results publish -c configs/meneu.yaml            # dry run
uv run ybench results publish -c configs/meneu.yaml --yes
uv run ybench results publish -c configs/meneu.yaml --yes      # "already published"
uv run ybench results publish -c configs/meneu.yaml --promote --yes
uv run ybench results publish -c configs/brooks.yaml --promote --yes
uv run ybench results get --run-id <id> --out /tmp/rt
diff -r results/meneu /tmp/rt/meneu                            # byte-identical
```

Check by eye on the hub: the card renders; `compare/summary.md` renders as
markdown; `stitched_pred.npz` shows as LFS; `plot.svg` renders inline; the
leaderboard table appears in the viewer; and after the brooks promote,
`published/meneu_foreign_dna/` is untouched. That last one is the only behaviour
worth watching on a real server — everything else is covered offline.

## Risks

**Growth is unbounded and hard to undo.** `runs/` is append-only and LFS objects
are not garbage-collected without a history rewrite. A full ten-task, four-model
rerun is 100–200MB, and every rerun adds that again. State a retention rule in
the card up front, and publish deliberately rather than on every local run.

**Sparse files understate the upload by roughly eight times.** The two
`stitched_pred.npz` files are 16,321,672 bytes apparent but a couple of MB on
disk, so `du -sh results` says 6.2MB while the real tree is 34MB. The uploader
reads the file, so the upload is the apparent size. Use `st_size` everywhere, as
`lock.py:95` does, and print the honest number in the dry-run plan. Separately,
those files are stored uncompressed; `np.savez_compressed` would cut 16.3MB to
about 4MB, but it changes bytes already on disk. Flag it, do not bundle it.

**Partial runs are structurally invisible.** `save_results` runs at `cli.py:77`
and `summary.json` is written at `cli.py:89`, so a crash in `task.plot()`
(`cli.py:80`) leaves `.npy` files with no summary, and `compare.py:88` silently
skips them. Refuse by default, allow with `--allow-partial`, record
`"partial": true` in the manifest, and never let a partial run be promoted.
`results/default` today holds 3 of the 12 pairs its config declares.

**One `out_dir` can hold two configs.** `configs/test.yaml` and
`configs/default.yaml` both declare `out_dir: results/default`, and `ybench run`
writes there regardless of config version — so editing a config and rerunning
fills one folder from two configs, and `compare.py:184` restricts by
`(model, task)` pair rather than by hash, so it will table them together. The
Modal path dodges this by namespacing the volume (`modal/app.py:156-160`); the
local path does not. The publisher must assert a single `config_hash` across the
snapshot and refuse otherwise. The bug predates this work; publishing only makes
it visible.

**Atomicity cuts one way.** One commit per run means the run appears whole or not
at all, and a crash between the LFS upload and the commit leaves orphaned blobs
that are invisible and harmless. The danger is the temptation to split a run
across several commits for progress reporting, which would let a reader see half
a run. Do not.

**Pulling into a dirty `results/`.** The tree is gitignored and people leave junk
in it. `get` must never delete anything, must write only files named in the
manifest, must refuse on a sha mismatch without `--force`, and must use the
`.part`-then-`os.replace` sequence.

**Single-model runs have no `compare/`** — `compare()` skips tasks with fewer
than two models. The index must not imply otherwise.
