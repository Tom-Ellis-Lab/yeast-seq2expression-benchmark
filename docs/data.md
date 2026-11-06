# Register, pin, and publish benchmark inputs

Benchmark inputs are registered in `src/yeastbench/data/manifest.py` using the
`InputArtifact` type from `src/yeastbench/data/artifacts.py`. Their filenames,
SHA-256 checksums, sizes, and published Hugging Face revisions live
in `src/yeastbench/data/manifest.lock.json`. Generated predictions belong to the
results workflow; `ybench data publish` only publishes input artifacts.

## Add or change data

1. Build the processed files with the task's preparation script under `scripts/`.
2. Register an `InputArtifact` in `data/manifest.py`, including its local
   destination, exact runtime filenames, task dependencies, and redistribution
   status.
3. Freeze the local files:

   ```bash
   uv run ybench data lock --artifacts ytk_promoter
   ```

   Missing required files and symlinks fail the command before the lock is changed.
   Locked distributions contain regular files whose contents can be inventoried.
   Re-locking unchanged bytes preserves their revision; changed bytes clear the
   old revision.
   Changing a declared repository or artifact prefix also clears its obsolete
   pin. Explicit `pin` and `publish` commands can establish the new source;
   ordinary downloads reject a stale declaration.

4. Preview the publication:

   ```bash
   uv run ybench data publish --to hf --artifacts ytk_promoter
   ```

   This reads the existing remote repository and prints additions/updates and
   deletions. It does not upload or change the local lock. The destination must
   already exist. A preview checks remote readability, not write permission.

5. Publish the reviewed files:

   ```bash
   uv run ybench data publish --to hf --artifacts ytk_promoter \
     --message "Update YTK benchmark inputs" --yes
   ```

   The publisher validates the entire selected set before any remote writes.
   Each repository receives one commit containing the selected artifacts. Files
   absent from the new inventories are removed only inside the selected artifact
   prefixes. Unselected artifacts and repository-level files remain outside that
   scope. Removing an entire artifact is a separate operation; empty snapshots
   are rejected.

   The expected parent commit protects against concurrent updates. After upload,
   the returned revision is checked against the intended file inventory and
   content identities before it is recorded in the local lock. This also handles
   unchanged uploads, where the Hub client can return an existing commit.

6. Commit the updated manifest, checksum/revision lock, preparation code, and
   corresponding benchmark changes to Git.

`--artifacts` takes artifact IDs, such as `hong`. The old `--tasks` spelling is
still accepted for `lock` and `publish`. In `data get` and `data verify`,
`--tasks` instead takes registered task names, such as `hong_igr`.

## Pin already-published inputs

Existing version 1 locks remain readable. They describe expected bytes, but
contain no Hugging Face revisions. Downloads identify these sources as
`@unpinned`; runtime model loading also reports unpinned sources.

To establish a revision without uploading anything:

```bash
uv run ybench data pin --artifacts ytk_promoter --revision main
uv run ybench data pin --artifacts yorzoi,codon_transformer --revision main
```

`pin` resolves the branch or tag to a full commit SHA. For processed data, it
downloads the files at that revision and checks every file against the existing
lock. A mismatch leaves the lock unchanged. Choose a matching historical
revision or publish the locked files instead; do not assign the latest revision
to older bytes without checking them.

For runtime-loaded models, `pin` records the resolved repository revision without
downloading weights. Subsequent fetching and loading use that exact snapshot.
Yorzoi loads from the resolved local snapshot; CodonTransformer loads both its
tokenizer and weights from the same snapshot. A failed pinned download does not
fall back to the current branch.

Explicit model configuration can supply `hf_revision: <full-40-character-SHA>`;
it takes precedence over the registered model's lock. An alternate `hf_repo`
never inherits a revision from a different repository. Include an explicit
`hf_revision` when experimenting with an alternate repository.

## Download and verify

```bash
uv run ybench data get --config configs/default.yaml
uv run ybench data verify --config configs/default.yaml

# Require revision pins for the selected registered Hugging Face sources.
uv run ybench data get --config configs/default.yaml --require-pinned
uv run ybench data verify --config configs/default.yaml --require-pinned
```

Pinned Hugging Face downloads request the recorded SHA, even after the remote
default branch changes. Files are still checked against their checksums.
Missing lock entries, incomplete inventories, and failed downloads produce a
failure rather than a successful zero-file result.

`--require-pinned` checks sources declared by the artifact manifest. It does not
yet validate arbitrary data paths or model overrides in a run config; that
binding belongs to the isolated-run workflow. HTTP/GCS files retain checksum
verification, but this flag does not add immutable remote versions to them.

Data commands accept `--lock-file <path>` for maintaining or inspecting another
lock. This option affects those commands only. The current benchmark runner and
model loaders read the project's default lock unless a model has an explicit
`hf_revision`; downloading through another lock does not switch later runs to it.

## Lock and receipt formats

A version 2 entry extends the existing file inventory with resolved sources.
This is an illustrative fixture, not a published dataset revision:

```json
{
  "version": 2,
  "artifacts": {
    "example_task": {
      "files": {
        "measurements.tsv": {
          "size": 4,
          "sha256": "9493985885f1acd67f91eb1c725fe4c30a6d46aff62b1e80d42dfb490bb84d4d"
        }
      },
      "sources": [{
        "backend": "hf",
        "repo_id": "example/benchmark-data",
        "repo_type": "dataset",
        "prefix": "example_task/",
        "revision": "1111111111111111111111111111111111111111"
      }]
    }
  }
}
```

Pins match the repository ID, repository type, and artifact prefix. Cache-only
model entries have an empty `files` object and a pinned model source. The schema
version alone does not imply that every artifact has been pinned.

After successful HF publication, the command prints the full revision and URL,
saves a receipt under `data/publications/`, and atomically updates the selected
local lock entries. `--receipt-dir` changes that receipt directory. The receipt
contains the repository, revision, URL, and selected snapshot inventory digest.
The commit's own SHA is stored locally after publication, not embedded in the
content whose commit hash it identifies.

If a local lock write fails after an upload, the printed revision and receipt
identify what was published; `data pin --revision <SHA>` can recover the local
reference after verifying it. If a network response is lost, inspect the remote
revision before retrying. Publication retries never run model inference.

## Google Cloud mirror

```bash
uv run ybench data publish --to gcs --artifacts ytk_promoter \
  --billing-project YOUR_PROJECT --yes
```

GCS validates the complete selected local set before copying files, but still
copies one file at a time and leaves obsolete remote files in place. It does not
create a Git commit or update Hugging Face pins. Historical retrieval from GCS
needs a separate immutable-path or object-generation scheme.

## Code organization

Input declarations describe what a benchmark needs. Shared storage types describe
the bytes to save or upload:

| Entity | Module | Meaning |
| --- | --- | --- |
| `InputArtifact`, `InputKind`, `Mirror` | `data/artifacts.py` | A registered dataset, model, or shared reference, its role, and its sources. |
| `ARTIFACTS`, `artifact_by_id`, `artifacts_for` | `data/manifest.py` | The input registry and dependency selection. |
| `FileRecord` | `storage/files.py` | One relative file path, its size, and its SHA-256 checksum. Used by input locks and uploads. |
| `DownloadFile`, `DownloadPlan` | `data/fetch.py` | A file record with its local destination/state, grouped by input and download source. |
| `UploadFile` | `storage/files.py` | A local file paired with its destination path and expected content identity for upload. |
| `HfCommitPlan` | `storage/hf.py` | The exact uploads and scoped deletions proposed for one HF repository commit. |
| `HfPublicationReceipt` | `storage/hf.py` | The verified repository revision, URL, and inventory digest returned by publication. |

`storage/files.py` owns file inspection, inventory creation, and verification;
`storage/json.py` owns atomic JSON writes. `data/lock.py` uses these shared file
records and translates them to the existing lock format. `build_lock_entries`
builds the selected per-input entries; merging and writing the full lock are
separate operations.

`data/publish.py` applies input selection and redistribution rules, then uses
`storage/hf.py`'s `HfPublisher` to plan and publish the selected files. The shared
storage modules do not import the input registry, so a results publisher can use
the same file records, commit plans, and receipts.

Scientific result objects and each benchmark's `save_results`/`load_results`
remain task-specific. Their existing JSON serialization also remains unchanged:
some metrics contain NaN, while the shared metadata writer requires finite JSON
values. A shared results publication layer can inventory those saved files
without changing their scientific format.

## Checkpoint 1 review

The tests in `tests/test_data_workflow.py` demonstrate an upload preview, one
commit for two artifacts, receipts, and verified historical downloads using a
fake Hub.

The committed legacy lock is retained until its real remote revisions are
verified. Run isolation, full-run result publication, and Modal run changes are
the next checkpoints.
