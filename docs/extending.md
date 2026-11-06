# Extending the benchmark

The codebase is built around two orthogonal abstractions, wired together
by a registry:

- **Protocols** (`src/yeastbench/adapters/protocols.py`) — small Python
  `Protocol`s describing what a model must implement to run a given
  *type* of benchmark. Current protocols:
  - `VariantEffectScorer.score_variants(variants) -> np.ndarray`
    — for eQTL-style benchmarks.
  - `SequenceExpressionScorer.predict_expression_scores(seqs) -> np.ndarray`
    — for native-position marginalized MPRA scoring.
- **Benchmarks** (`src/yeastbench/benchmarks/`) — a `Benchmark` subclass
  per task type. Each declares `adapter_protocol` (which protocol it
  consumes) and implements `evaluate`, `plot`, `save_results`,
  `load_results`, `summary_dict`, and `headline`.
- **Adapters** (`src/yeastbench/adapters/`) — one class per
  `(model, protocol)` pair. Implements the protocol by wrapping the
  model's forward pass, tokenization, and post-processing.

The CLI's `_run_pair` is task-agnostic: `task = TASKS[name](...)`,
`adapter = MODELS[name](task, device, ...)`, then
`task.evaluate(adapter) → task.plot → task.save_results`.

## Adding a new benchmark

Most new benchmarks reuse an existing protocol. The workflow:

1. **Pick or add a protocol.** Can one of the existing protocols score
   your task? If yes, reuse it. If no — the task needs a
   semantically-different operation — add a new `@runtime_checkable`
   `Protocol` in `adapters/protocols.py`.
2. **Write the Benchmark class** in `src/yeastbench/benchmarks/<name>.py`:
   - Subclass `Benchmark[AdapterT, ResultT]` with your adapter protocol
     and results dataclass.
   - Set `adapter_protocol: ClassVar[type] = YourProtocol`.
   - Implement `__init__(<task_config_fields>, info)`,
     `evaluate(adapter) -> Results`, `plot`, `save_results`,
     `load_results`, `summary_dict`, `headline`.
3. **Register** the task in `src/yeastbench/registry.py`:
   ```python
   def _build_my_task(path_a, path_b) -> Benchmark:
       return MyBenchmark(..., info=BenchmarkInfo(name="my_task", ...))

   TASKS["my_task"] = _build_my_task
   ```
4. **If you added a new protocol**, extend each model's adapter map
   (see "Adding a new model" below) with an implementation for that
   protocol.
5. **Reference the task in `configs/default.yaml`** under both
   `tasks_config:` (its constructor kwargs) and any `runs:` that should
   include it.
6. **Write a spec** in `docs/benchmarks/<name>.md` and add tests in
   `tests/test_<name>.py`.

## Adding a new model

1. **Implement one adapter class per protocol the model should support**,
   in `src/yeastbench/adapters/<model>_<task_type>.py`. Each adapter
   wraps the model's forward pass + any pre/post-processing, and
   exposes the single method required by its protocol.
2. **Register the model** in `src/yeastbench/registry.py` by adding a
   protocol → builder dict:
   ```python
   def _mymodel_eqtl(device, fasta_path, gtf_path, **cfg):
       from yeastbench.adapters.mymodel_eqtl import MyModelScorer
       return MyModelScorer(fasta_path, gtf_path, device, **cfg)

   MYMODEL_ADAPTERS: dict[type, tuple[Callable, bool]] = {
       VariantEffectScorer: (_mymodel_eqtl, True),  # True = needs FASTA/GTF
       # add more protocol entries as you add adapters
   }

   def _build_mymodel(task, device, **cfg):
       return _dispatch(MYMODEL_ADAPTERS, task, device, **cfg)

   MODELS["mymodel"] = _build_mymodel
   ```
   The `needs_refs` flag controls whether the dispatcher passes the
   task's `fasta_path`/`gtf_path` to the adapter (true for
   genomic-context tasks; false for protocol-only adapters that need
   neither reference, e.g. the Brooks coverage track).
3. **Reference the model in `configs/default.yaml`** under `runs:`
   with the model-specific kwargs it accepts (checkpoint paths, batch
   size, `use_rc`, etc.).
4. **Optional**: add a `[project.optional-dependencies]` entry for any
   model-specific packages (e.g., a HuggingFace wheel), so users can
   install just the adapter they need with `uv sync --extra mymodel`.

No new runner scripts, no new CLI wiring — both tasks and models are
fully plug-in.
