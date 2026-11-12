"""Registered benchmark inputs and dependency selection.

Task data, model weights, and shared references are declared here as
``InputArtifact`` instances. Their types are defined in ``data.artifacts``;
per-file checksums and pinned source revisions live in ``manifest.lock.json``.

Adding a task or model to ``registry.py`` should include a corresponding input
here. ``validate_against_registry`` checks that every registered task/model has
its required inputs declared.
"""

from __future__ import annotations

from yeastbench.data.artifacts import BackendKind, InputArtifact, InputKind, License, Mirror


# ──────────────────────────────────────────────────────────────
# Mirror helpers
# ──────────────────────────────────────────────────────────────

# Mirrors for our own processed task data — these repos/buckets are the publish
# targets. HF is the free, auth-free default; the GCS bucket is **requester
# pays** (project clex-415415, US multi-region), so every read/write must name a
# billing project (`ybench data get/publish --billing-project …` or the
# YBENCH_GCS_BILLING_PROJECT env var).
_HF_DATA_REPO = "tom-ellis-lab/yeast-seq2expression-benchmark-data"
_GCS_DATA_BUCKET = "gs://yeast-seq2expression-benchmark/"


def _task_mirrors(artifact_id: str) -> tuple[Mirror, ...]:
    """HF-first, GCS-secondary mirrors for one task's processed data."""
    return (
        Mirror(BackendKind.HF, base=_HF_DATA_REPO, prefix=f"{artifact_id}/"),
        Mirror(BackendKind.GCS, base=_GCS_DATA_BUCKET, prefix=f"{artifact_id}_v1/"),
    )


def _task(
    artifact_id: str,
    *,
    dest: str,
    needed_by: tuple[str, ...],
    requires: tuple[str, ...] = (),
    license: License = License.OWN,
    include: tuple[str, ...] = ("**/*",),
) -> InputArtifact:
    return InputArtifact(
        id=artifact_id,
        kind=InputKind.TASK_PROCESSED,
        dest=dest,
        mirrors=_task_mirrors(artifact_id),
        needed_by=needed_by,
        requires=requires,
        include=include,
        license=license,
        redistributable=(
            license in (License.OWN, License.CC_BY_4_0, License.CC_BY_NC_4_0)
        ),
    )


# ──────────────────────────────────────────────────────────────
# Declarations
# ──────────────────────────────────────────────────────────────

_REFS = "refs"

ARTIFACTS: tuple[InputArtifact, ...] = (
    # Shared genome references — declared once, required by many tasks.
    InputArtifact(
        id=_REFS,
        kind=InputKind.SHARED_REFERENCE,
        dest="data/tasks",
        include=(
            "R64-1-1.fa", "R64-1-1.fa.fai", "R64-1-1.115.gtf",
            "R64-5-1.fa", "R64-5-1.fa.fai",
        ),
        mirrors=_task_mirrors(_REFS),
        license=License.PUBLIC,  # SGD reference genome
        redistributable=True,
    ),

    # ── Task-processed data ────────────────────────────────────
    # `include` is an allowlist of exactly the files each benchmark reads at run
    # time (verified against the benchmark/adapter code) — NOT the whole task
    # dir. Raw inputs, build intermediates (_intermediates_*), and provenance
    # (README, build manifests) stay out of the v1 distribution; they belong to
    # the v2 raw tier.
    _task(
        "cuperus_mpra_5utr",
        dest="data/tasks/cuperus_mpra_5utr",
        needed_by=("cuperus_utr",),
        requires=(_REFS,),
        include=("random_utrs.tsv", "native_utrs.tsv", "construct.json"),
    ),
    _task(
        "caudal_eqtl",
        dest="data/tasks/caudal_eqtl",
        needed_by=("caudal_eqtl",),
        requires=(_REFS,),
        include=("negset_*.tsv",),  # benchmark globs negset_*.tsv (root only)
    ),
    _task(
        "kita_eqtl",
        dest="data/tasks/kita_eqtl",
        needed_by=("kita_eqtl",),
        requires=(_REFS,),
        include=("negset_*.tsv",),
    ),
    _task(
        "rafi_mpra",
        dest="data/tasks/rafi_mpra",
        needed_by=("rafi_mpra_marginalized",),
        requires=(_REFS,),
        include=(
            "filtered_test_data_with_MAUDE_expression.txt",
            "test_subset_ids/*.csv",
        ),
    ),
    _task(
        "shalem_mpra_terminator",
        dest="data/tasks/shalem_mpra_terminator",
        needed_by=("shalem_mpra_marginalized",),
        requires=(_REFS,),
        include=("segal_2015.tsv", "host_genes.json"),
    ),
    _task(
        "wu_rfpins",
        dest="data/tasks/wu_rfpins",
        needed_by=("wu_rfpins",),
        requires=(_REFS,),
        include=(
            "table_s2_fluorescence_1044_loci.csv",
            "expression_cassette.fasta",
            "barcodes.tsv",
        ),
    ),
    _task(
        "hong",
        dest="data/tasks/hong",
        needed_by=("hong_igr",),
        requires=(_REFS,),
        include=("hong_igr_v1.tsv", "expression_cassette.fasta"),
    ),
    _task(
        "chen_synonymous",
        dest="data/tasks/chen_synonymous",
        needed_by=("chen_synonymous",),
        requires=(_REFS,),
        # gfp_r1/gfp_r2 read for cassette synthesis even by the tdh3 run; hosts
        # json read by the adapter. construct_*.fa/.gtf + library_loci are build.
        include=("gfp_r1.tsv", "gfp_r2.tsv", "tdh3.tsv", "marginalized_hosts.json"),
    ),
    _task(
        "brooks_scramble",
        dest="data/tasks/brooks_scramble",
        needed_by=("brooks_scramble",),
        license=License.OWN,  # Brooks lab Nanopore data, our processing
        # Window-agnostic artifact: per-construct index + generous gene-centred
        # sequence slices + per-base coverage. The benchmark re-cuts each model's
        # window at run time, so no per-window TSV.
        include=("brooks_index.tsv", "brooks_constructs.fasta", "brooks_cov.npz"),
    ),
    _task(
        "meneu_foreign_dna",
        dest="data/tasks/meneu_foreign_dna",
        needed_by=("meneu_foreign_dna",),
        license=License.CC_BY_4_0,  # ExoShorkie figshare
        # One window-agnostic sidecar per contig (seq + fwd/rev); the benchmark
        # tiles to the model's receptive field at run time, so no per-window TSV.
        include=("meneu_cov_Mpneumo.npz", "meneu_cov_Mmmyco.npz"),
    ),
    _task(
        "ytk_promoter",
        dest="data/tasks/ytk_promoter",
        needed_by=("ytk_promoter",),
        requires=(_REFS,),
        license=License.CC_BY_NC_4_0,
        include=("figure3a.tsv", "constructs.tsv", "constructs.fasta"),
    ),

    # ── Model weights ──────────────────────────────────────────
    # Shorkie: authors' PUBLIC bucket — fetchable today, no re-hosting.
    # Local layout is f{i}.h5 + params.json; the bucket stores f{i}/model_best.h5
    # and params.json lives in the shorkie-pytorch repo.
    InputArtifact(
        id="shorkie",
        kind=InputKind.MODEL_WEIGHTS,
        dest="data/models/shorkie",
        needed_by=("shorkie",),
        include=("checkpoints/f0.h5", "checkpoints/f1.h5", "checkpoints/f2.h5",
                 "checkpoints/f3.h5", "checkpoints/f4.h5", "checkpoints/f5.h5",
                 "checkpoints/f6.h5", "checkpoints/f7.h5", "params.json",
                 "targets.txt"),  # read at runtime by shorkie_hong
        license=License.PUBLIC,
        redistributable=False,  # upstream's to host; we point at their bucket
        mirrors=(
            Mirror(
                BackendKind.HTTP,
                base="https://storage.googleapis.com/seqnn-share/shorkie_models/shorkie/",
                member_remote={
                    "checkpoints/f0.h5": "f0/model_best.h5",
                    "checkpoints/f1.h5": "f1/model_best.h5",
                    "checkpoints/f2.h5": "f2/model_best.h5",
                    "checkpoints/f3.h5": "f3/model_best.h5",
                    "checkpoints/f4.h5": "f4/model_best.h5",
                    "checkpoints/f5.h5": "f5/model_best.h5",
                    "checkpoints/f6.h5": "f6/model_best.h5",
                    "checkpoints/f7.h5": "f7/model_best.h5",
                    "params.json": (
                        "https://raw.githubusercontent.com/tdsone/"
                        "shorkie-pytorch/main/data/shorkie_params.json"
                    ),
                },
            ),
        ),
    ),
    InputArtifact(
        id="shorkie_random_init",
        kind=InputKind.MODEL_WEIGHTS,
        dest="data/models/shorkie_random_init",
        needed_by=("shorkie_random_init",),
        include=("checkpoints/f0.h5", "checkpoints/f1.h5", "checkpoints/f2.h5",
                 "checkpoints/f3.h5", "checkpoints/f4.h5", "checkpoints/f5.h5",
                 "checkpoints/f6.h5", "checkpoints/f7.h5", "params.json",
                 "targets.txt"),  # read at runtime by shorkie_hong
        license=License.PUBLIC,
        redistributable=False,  # upstream's to host; we point at their bucket
        mirrors=(
            Mirror(
                BackendKind.HTTP,
                base="https://storage.googleapis.com/seqnn-share/shorkie_models/shorkie_random_init/",
                member_remote={
                    "checkpoints/f0.h5": "f0/model_best.h5",
                    "checkpoints/f1.h5": "f1/model_best.h5",
                    "checkpoints/f2.h5": "f2/model_best.h5",
                    "checkpoints/f3.h5": "f3/model_best.h5",
                    "checkpoints/f4.h5": "f4/model_best.h5",
                    "checkpoints/f5.h5": "f5/model_best.h5",
                    "checkpoints/f6.h5": "f6/model_best.h5",
                    "checkpoints/f7.h5": "f7/model_best.h5",
                    "targets.txt": (
                        "https://storage.googleapis.com/seqnn-share/"
                        "shorkie_models/shorkie/targets.txt"
                    ),
                },
            ),
        ),
    ),
    # Yorzoi + CodonTransformer: HF-hub models the adapters load via
    # from_pretrained at runtime. `get` warms the HF cache; nothing lands
    # under data/. The lock records their repository revision without a local
    # file inventory; the loaders request that exact snapshot.
    InputArtifact(
        id="yorzoi",
        kind=InputKind.MODEL_WEIGHTS,
        dest="",
        needed_by=("yorzoi",),
        cache_only=True,
        include=(),
        license=License.PUBLIC,
        mirrors=(Mirror(BackendKind.HF, base="tom-ellis-lab/yorzoi", repo_type="model"),),
    ),
    InputArtifact(
        id="codon_transformer",
        kind=InputKind.MODEL_WEIGHTS,
        dest="",
        needed_by=("codon_transformer",),
        cache_only=True,
        include=(),
        license=License.PUBLIC,
        mirrors=(Mirror(BackendKind.HF, base="adibvafa/CodonTransformer", repo_type="model"),),
    ),
    # DREAM-RNN supervised baseline: a single small net (~8 MB) + the reporter
    # plasmid. Weights come from Zenodo 10633252 (dir 0_1_1_0/model_best.pth, the
    # Bi-LSTM-core composite — NOT 0_1_0_0, which is DREAM-CNN); plasmid.json from
    # the de-Boer-Lab repo (MIT). Both are small + redistributable, so they're
    # re-hosted on our data mirrors rather than fetched from the 2.3 GB tarball.
    # Lock the two files locally, then publish them to the HF/GCS mirrors.
    InputArtifact(
        id="dream_rnn",
        kind=InputKind.MODEL_WEIGHTS,
        dest="data/models/dream_rnn",
        needed_by=("dream_rnn",),
        include=("model_best.pth", "plasmid.json"),
        license=License.CC_BY_4_0,
        redistributable=True,
        mirrors=_task_mirrors("dream_rnn"),
    ),
)


# Models that legitimately have no downloadable artifact (pure baselines that
# derive everything from task data already covered elsewhere).
MODELS_WITHOUT_ARTIFACTS: frozenset[str] = frozenset({"cai"})


# ──────────────────────────────────────────────────────────────
# Lookups & selection
# ──────────────────────────────────────────────────────────────

_BY_ID = {a.id: a for a in ARTIFACTS}


def artifact_by_id(artifact_id: str) -> InputArtifact:
    try:
        return _BY_ID[artifact_id]
    except KeyError:
        raise KeyError(
            f"No artifact '{artifact_id}'. Known: {sorted(_BY_ID)}"
        ) from None


def artifacts_for(
    tasks: list[str] | None = None,
    models: list[str] | None = None,
) -> list[InputArtifact]:
    """Resolve the artifacts needed by the given task/model names, closing
    over ``requires`` (shared references). With both ``None``, returns every
    artifact. Order is stable and deduplicated.

    Unknown names raise — a typo shouldn't silently download nothing.
    """
    if tasks is None and models is None:
        return list(ARTIFACTS)

    wanted = set(tasks or []) | set(models or [])
    selected: dict[str, InputArtifact] = {}

    def _add(art: InputArtifact) -> None:
        if art.id in selected:
            return
        selected[art.id] = art
        for dep_id in art.requires:
            _add(artifact_by_id(dep_id))

    matched_names: set[str] = set()
    for art in ARTIFACTS:
        hit = wanted.intersection(art.needed_by)
        if hit:
            matched_names |= hit
            _add(art)

    unmatched = wanted - matched_names - MODELS_WITHOUT_ARTIFACTS
    if unmatched:
        raise KeyError(
            f"No artifacts declared for: {sorted(unmatched)}. "
            f"Check the names against `ybench list`."
        )

    # Preserve ARTIFACTS declaration order.
    return [a for a in ARTIFACTS if a.id in selected]


def validate_against_registry() -> list[str]:
    """Return a list of problems: registered tasks/models with no artifact, and
    artifacts pointing at unknown task/model names. Empty list == all good.
    """
    from yeastbench.registry import MODELS, TASKS

    problems: list[str] = []

    covered_tasks = {n for a in ARTIFACTS for n in a.needed_by}
    for task_name in TASKS:
        if task_name not in covered_tasks:
            problems.append(f"task '{task_name}' has no declared artifact")

    covered_models = {n for a in ARTIFACTS for n in a.needed_by}
    for model_name in MODELS:
        if model_name in MODELS_WITHOUT_ARTIFACTS:
            continue
        if model_name not in covered_models:
            problems.append(f"model '{model_name}' has no declared artifact")

    known = set(TASKS) | set(MODELS)
    for art in ARTIFACTS:
        for name in art.needed_by:
            if name not in known:
                problems.append(
                    f"artifact '{art.id}' references unknown name '{name}'"
                )
    return problems
