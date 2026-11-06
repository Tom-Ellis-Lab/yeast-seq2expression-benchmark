# Roadmap

## Contents

**[v1 release](#v1-release)**

- [Infrastructure](#infrastructure)
- [eQTL](#eqtl) — Caudal, Kita
- [MPRA](#mpra) — Rafi / de Boer, Shalem, Chen, Wu, Hong, Cuperus
- [Structural rearrangements](#structural-rearrangements) — Brooks
- [Foreign-DNA integration](#foreign-dna-integration) — Meneu
- [Code-structure refactor](#code-structure-refactor)
- [Documentation](#documentation)
- [Reproducibility](#reproducibility)
- [Pre-release cleanup](#pre-release-cleanup)

**[v2 release](#v2-release)**

- [Roadmap](#roadmap)
  - [Contents](#contents)
  - [v1 release](#v1-release)
    - [Infrastructure](#infrastructure)
    - [eQTL](#eqtl)
      - [Caudal et al. cis-eQTL classification](#caudal-et-al-cis-eqtl-classification)
      - [Kita et al. eQTL](#kita-et-al-eqtl)
    - [MPRA](#mpra)
      - [Rafi / de Boer (promoter, DREAM)](#rafi--de-boer-promoter-dream)
      - [Shalem / Segal (terminator)](#shalem--segal-terminator)
      - [Chen et al. (synonymous CDS)](#chen-et-al-synonymous-cds)
      - [Wu et al. (RFP insertions)](#wu-et-al-rfp-insertions)
      - [Hong et al. (mCherry IGR insertions)](#hong-et-al-mcherry-igr-insertions)
      - [Cuperus et al. (5′ UTR)](#cuperus-et-al-5-utr)
      - [Lee et al. (YTK promoter panel)](#lee-et-al-ytk-promoter-panel)
    - [Structural rearrangements](#structural-rearrangements)
    - [Foreign-DNA integration](#foreign-dna-integration)
    - [Code-structure refactor](#code-structure-refactor)
      - [Correctness sweep](#correctness-sweep)
    - [Documentation](#documentation)
    - [Reproducibility](#reproducibility)
    - [Pre-release cleanup](#pre-release-cleanup)
    - [Timon Cleaup Notes](#timon-cleaup-notes)
  - [v2 release](#v2-release)
    - [ExoShorkie integration](#exoshorkie-integration)
    - [Species LM](#species-lm)
    - [Condition coherence](#condition-coherence)
    - [Caudal eQTL effect-size calibration](#caudal-eqtl-effect-size-calibration)
    - [Caudal eQTL positive-set cleanup](#caudal-eqtl-positive-set-cleanup)
    - [Per-stratum reporting and bootstrap CIs](#per-stratum-reporting-and-bootstrap-cis)
    - [Hong et al. extensions](#hong-et-al-extensions)
    - [Brooks et al. extensions](#brooks-et-al-extensions)
    - [Meneu et al. extensions](#meneu-et-al-extensions)
    - [Bin-boundary sensitivity](#bin-boundary-sensitivity)
    - [Baselines as first-class models](#baselines-as-first-class-models)
    - [CI smoke-test](#ci-smoke-test)

Per-benchmark detail (scoring design, sources, results) lives in the spec
files under `docs/benchmarks/`. This roadmap tracks status only.

## v1 release

### Infrastructure

- [x] Protocol-based adapter dispatch — 10 protocols
  (`VariantEffectScorer`, `SequenceExpressionScorer`,
  `TerminatorMarginalizedExpressionPredictor`, `CassetteExpressionPredictor`,
  `IGRInsertionExpressionPredictor`, `CoverageTrackPredictor`,
  `TiledCoverageTrackPredictor`, `LocalCodingVariantPredictor`,
  `FivePrimeUtrReporterExpressionPredictor`,
  `IntegratedPromoterPanelPredictor`)
- [x] Benchmark ABC (`evaluate` / `plot` / `save_results` / `load_results` /
  `summary_dict` / `headline`)
- [x] Registry (`SHORKIE_ADAPTERS` / `YORZOI_ADAPTERS` keyed by protocol;
  tasks in `TASKS`)
- [x] YAML run-spec CLI (`ybench run|list` + `ybench data` sub-app;
  `configs/default.yaml`)
- [x] Run UX (PR #23): hardware banner (`describe_device`), `--gpu` / `--device`
  selection, per-pair ETA (`hardware.py`, `cli.py`, `tests/test_run_progress.py`)
- [x] 331-test pytest suite
- [x] Cross-model comparison runner (`compare.py`): per-task
  `compare/per_task/<task>/` plots via each benchmark's `compare_plot` hook,
  plus cross-task `summary.csv` / `summary.md`; auto-runs after every
  `ybench run` (no standalone command)
- [x] CLI simplification: removed the `compare` and `replot` commands —
  `ybench run` alone produces per-model results, plots, and the cross-model
  comparison. (Reusing persisted scores so a re-run regenerates outputs without
  re-scoring is deferred to the result-caching follow-up; see Reproducibility.)
- [x] Fix the auto-comparison scope: `compare()` is restricted to the config's
  `(model, task)` pairs, so stale on-disk results from other runs are ignored
  (was: walked the whole `out_dir` and pulled in models not named in the config).
- [x] Unify split tasks: one logical benchmark = one registry task. The
  benchmark re-cuts each construct to the adapter's receptive field at run time
  from a single window-agnostic artifact; `<task>_shorkie` twins and the
  `compare_task_name` overrides are gone.
  - [x] Meneu: `meneu_foreign_dna` only; per-contig `meneu_cov_<contig>.npz`
    (seq + fwd/rev), `tile_contig` at eval time (`benchmarks/meneu.py`)
  - [x] Brooks: `brooks_scramble` only; window-agnostic `brooks_index.tsv` +
    `brooks_constructs.fasta` + `brooks_cov.npz`, `window_slice` + membership/
    dedup replay at eval time (bit-identical 698/1055; `benchmarks/brooks.py`)
  - [x] Chen: `chen_synonymous` only (library-split, not window) — one task
    scores all three libraries, results stratified per library + aggregate
    (`benchmarks/chen.py`)
  - [x] `compare_task_name` / `_group_by_compare_task` deleted; `compare.py`
    indexes directly on the registry task name

### eQTL

#### Caudal et al. cis-eQTL classification

- [x] Benchmark class + per-iteration `negset_*.tsv` eval (|score| AUROC/AUPRC,
  close-only subset, distance-to-TSS strata)
- [x] Shorkie adapter (`ShorkieVariantScorer`, 8-fold ensemble)
- [x] Yorzoi adapter (`YorzoiVariantScorer`, strand-swap RC averaging)
- [x] Verified vs the Shorkie paper's Caudal numbers — reproduces Fig 7E
  (≤ 8 kb) within readout precision on current per-base code; the earlier
  apparent divergence was stale pre-migration results
- [x] Audit discovery methodology — positives called from the 1011-strain
  panel with varying ploidy / per-variant VAFs. Decided: no AF/VAF filter;
  positive-set cleanup deferred to v2 (confound real but small on Yorzoi).
  Audit + notebook in `scripts/eqtl/audit/`

#### Kita et al. eQTL

- [x] Spec (`docs/benchmarks/kita_eqtl.md`)
- [x] Generalized eQTL data-prep handles Kita (`scripts/eqtl/0_data_generation/*`
  with `--dataset kita`)
- [x] Distribution committed (`data/tasks/kita_eqtl/`, negset_{1..4}.tsv,
  619 rows each; `scripts/eqtl/build_kita_v1_distribution.py`)
- [x] Wired into `TASKS` + `configs/default.yaml` (reuses
  `EQTLClassificationBenchmark`)
- [ ] Shorkie + Yorzoi end-to-end run on Kita (adapters already wired)

### MPRA

#### Rafi / de Boer (promoter, DREAM)

- [x] Deleted the fixed-context variant (2026-05-21): `MPRARegressionBenchmark`,
  its adapters, the `rafi_mpra_promoter` task, and the
  `SequenceExpressionPredictor` protocol are gone. Marginalized / native-position
  is the honest eval for Shorkie / Yorzoi.
- [x] Marginalized benchmark (`MPRAMarginalizedBenchmark`; 22 host genes,
  180 bp upstream of TSS; logSED_agg over host-gene exon bins)
- [x] Shorkie marginalized adapter (T0 tracks, 8-fold ensemble)
- [x] Yorzoi marginalized adapter (strand-matched tracks)
- [x] Pair-difference Pearson for SNV / motif pair strata (`per_pair_stratum`
  with `diff_pearson_r` in both summaries)
- [x] DREAM-RNN supervised baseline — single in-distribution model, reported
  separately from zero-shot (spec: `docs/benchmarks/rafi_mpra_promoter.md`); weights
  pending publish to mirrors

#### Shalem / Segal (terminator)

- [x] Spec (`docs/benchmarks/shalem_mpra_terminator.md`): marginalized-only, 150 bp
  insert + 300 bp CYC1 no-term filler, 450 bp replacement at stop + 1
- [x] cycl-512 mutant UTR reconstruction (Guo 1995)
- [x] DEE2 per-gene median TPM table (`data/tasks/dee2_gene_median_tpm.tsv`)
- [x] Host-gene selection (`scripts/shalem/select_host_genes.py`; 22 genes at
  `data/tasks/shalem_mpra_terminator/host_genes.json`)
- [x] `ShalemMPRAMarginalizedBenchmark` + `_shalem_scaffold.py`
- [x] Shorkie adapter (`ShorkieShalemPredictor`)
- [x] Yorzoi adapter (`YorzoiShalemPredictor`)

#### Chen et al. (synonymous CDS)

- [x] Spec (`docs/benchmarks/chen_synonymous.md`): marginalized over 20 YPD hosts,
  per-replicate labels, per-library replicate ceilings
- [x] `LocalCodingVariantPredictor` protocol (`predict_local_variants`)
- [x] `ChenSynonymousBenchmark` (`benchmarks/chen.py`); one task `chen_synonymous`
  in `TASKS` + `configs/default.yaml`, results stratified per library + aggregate
- [x] WT avGFP CDS backbone fix (PR #24 / issue #8; `_chen_gfp_reference.py`)
- [x] Shorkie + Yorzoi adapters (`shorkie_chen_marginalized.py`,
  `yorzoi_chen_marginalized.py`; shared `_chen_marginalized.py`)
- [x] Tests (`tests/test_chen.py`, `tests/test_chen_gfp_reference.py`)
- [x] GPU runs (Shorkie / Yorzoi) — per-library r / ρ recorded
- [x] CAI baseline (`cai`, `adapters/baselines/cai.py`)
- [x] CodonTransformer baseline (`codon_transformer`,
  `adapters/baselines/codon_transformer.py`)

#### Wu et al. (RFP insertions)

- [x] Spec (`docs/benchmarks/wu_rfpins.md`)
- [x] Labels (`table_s2_fluorescence_1044_loci.csv`, 1044 loci)
- [x] Cassette verified + frozen (`scripts/wu/verify_cassette.py`,
  `expression_cassette.fasta`, `scripts/wu/build_cassette_fasta.py`)
- [x] Benchmark `RFPInsertionBenchmark` + `_wu_scaffold.py`
  (`tests/test_rfpins.py`)
- [x] Shorkie + Yorzoi adapters (`ShorkieWuPredictor`, `YorzoiWuPredictor`);
  mCherry stop codon at the downstream crop edge (max upstream context),
  strand-aware
- [x] Metrics: two binary tail tasks (extreme-low < 5, extreme-high ≥ 8) with
  AUROC / AUPRC + ROC/PR plots
- [x] GPU run (Shorkie / Yorzoi) — headline negative result
- [x] Diagnostic dump `scripts/wu/dump_yorzoi_tracks.py`
- [x] Per-ORF UPTAG / DNTAG barcodes injected (`barcodes.tsv`,
  `scripts/wu/build_barcodes.py`, `inject_barcodes`) — replaces OOD 20×N
  slots with real SGTC tags (1013 both / 31 synthetic DNTAG)
- [ ] Barcode robustness re-run (Shorkie / Yorzoi) — confirm N→real-tag
  swap leaves the headline negative result unchanged

#### Hong et al. (mCherry IGR insertions)

- [x] Spec (`docs/benchmarks/hong_igr.md`): 98 IntTrain + 52 IntProp = 150 loci,
  mCherry CDS-sum readout, two co-primary metrics (Primary RNA-seq × cassette
  CDS; IntTrain-fitted IntProp ρ)
- [x] Published ceiling SPCC = 0.847 surfaced in `summary.json`; caveats in
  `docs/benchmarks/model_failures.md`
- [x] Reference R64-5-1 (`data/tasks/R64-5-1.fa`)
- [x] Distribution built (`scripts/hong/build_hong_distribution.py`,
  `data/tasks/hong/hong_igr_v1.tsv`, 150 rows; coords from Table S2 gRNAs →
  R64-5-1; S5 ignored, documented in spec)
- [x] Cassette FASTA frozen (`data/tasks/hong/expression_cassette.fasta`,
  1595 bp, mCherry CDS at offset 686; always + strand)
- [x] Shared insertion scaffold `_cassette_scaffold.py` (used by Wu + Hong;
  window anchor parameterized)
- [x] `HongIGRInsertionBenchmark` + `_hong_scaffold.py` +
  `IGRInsertionExpressionPredictor` protocol
- [x] Two-metric output (Primary + IntTrain-fitted IntProp ρ); adapters opt in
  via optional `predict_diagnostic_readouts`
- [x] Shorkie adapter `ShorkieHongPredictor` (8-fold; T0 RNA-seq + histone /
  nucleosome track groups)
- [x] Yorzoi adapter `YorzoiHongPredictor` (RNA-seq only)
- [x] Tests (`tests/test_hong_igr.py`)
- [x] GPU runs (Shorkie / Yorzoi) recorded

#### Cuperus et al. (5′ UTR)

- [x] Spec (`docs/benchmarks/cuperus_mpra_5utr.md`): HIS3 reporter, no marginalization;
  two metrics (direct corr + Kozak partial-corr)
- [x] Source data vendored (`archive/cuperus/`; random 489,348 + native 11,856
  variable-length, both primary evals)
- [x] Benchmark `CuperusUTRBenchmark` + `_cuperus_scaffold.py` (CYC1pr–HIS3–CYC1term
  reporter, HIS3 readout); `FivePrimeUtrReporterExpressionPredictor` protocol;
  Kozak partial-corr in the eval layer
- [x] Shorkie + Yorzoi adapters (+ registry + `configs/default.yaml`)
- [x] GPU run (Shorkie / Yorzoi) recorded; scored single-HIS3 (the `backgrounds=`
  machinery was removed after a divergence check, 2026-06-03)

#### Lee et al. (YTK promoter panel)

- [x] Figure 3A digitised for 19 promoters × mRuby2/Venus
- [x] YTK constructs rebuilt from the official GenBank parts and URA3 assembly
  rules; exact reporter CDS coordinates frozen
- [x] Raw, untransformed, unbinned per-base CDS-sum adapters for Shorkie and
  Yorzoi
- [x] Consensus max/min dynamic-range recovery and per-reporter metrics
- [x] GPU run (Shorkie / Yorzoi) on Modal A10, 2026-08-25; Shorkie recovers
  0.165 of the consensus log-range, Yorzoi 0.109
- [ ] v2: ask the authors for original per-replicate Figure 3A values and final
  promoter-test plasmid sequences; replace both digitisation and reconstruction

### Structural rearrangements

Brooks et al. SCRaMBLE chromosome 9 — spec `docs/benchmarks/brooks_scramble.md`.

- [x] Spec locked (2026-05-19): per-copy sampling, total-native-reads size
  factor, locked sample rule (alt differs from native in the receptive field,
  intact CDS, deletions excluded), gene-centred alt-vs-native constructs
- [x] Data from `gs://brooks-nanopore/` — each strain's `JS<n>_1` contig already
  encodes the rearranged synIXR; construct = gene-centred window vs parental
  `JS96_1` (no junction-walking)
- [x] Leakage characterized: SCRaMBLEd input sequences treated zero-shot (user
  2026-05-19), but Yorzoi's training targets include the Brooks Nanopore tracks
  (verified 2026-05-20) → its headline is partly a leakage measurement, not
  zero-shot. Shorkie is clean; remediation deferred to v2 (see Brooks extensions)
- [x] Distribution built (`scripts/brooks/build_brooks_distribution.py` →
  window-agnostic `brooks_index.tsv` + `brooks_constructs.fasta` +
  `brooks_cov.npz`): 1786 candidate constructs / 56 strains (→ 698 @ 4992,
  1055 @ 16384 after the run-time window/dedup); JS94 deep-WT replicates;
  per-copy sampling; per-replicate raw + normalized JS94 coverages in the schema
- [x] `CoverageTrackPredictor` protocol — batched `predict_coverage_batch(seqs,
  strands, strains) → (B, out_len)`, per-base raw counts; adapters expose
  `batch_size`
- [x] Yorzoi adapter `YorzoiBrooksPredictor` (`yorzoi_brooks.py`): RC averaging +
  strand swap, inverse Borzoi transform, per-base unbin (10 bp);
  `track_mode = all | nanopore_all | matched` (default matched)
- [x] Shorkie adapter `ShorkieBrooksPredictor`: 8-fold, T0 tracks, softplus raw
  counts, 16 bp unbin, `varies_by_strain = False`
- [x] `BrooksScrambleBenchmark`: per-replicate LFC design (0–3 true + 0–3
  predicted LFCs per sample); LFC metrics (scored: r / ρ / dir-acc;
  calibration: within-range + mean |z|); shape metrics (per-base r + JS
  divergence); LOO noise ceiling. JS94 replicate aliases in `_yorzoi_constants.py`
- [x] LFC metrics reported **per JS94 replicate**, never averaged across the
  three (2026-09-02). Each replicate scores a different, unequally sized gene
  set — the read-depth cutoff is applied per gene per run — against its own
  ceiling, so a mean over replicates weighted unlike cohorts equally. One
  shared `per_rep_lfc_metrics` now serves `evaluate` / `load_results` /
  compare, and `n_scored_per_rep` ships with every number
- [x] Cross-model shared-cohort convention: headline on the intersection of the
  two models' sample sets; `ybench compare` Brooks logic in `benchmarks/brooks.py`
  writes the shared-cohort summary + charts. Reconciles the sample set + LFC
  only — the shape metrics are still scored over each model's own window (not yet
  cross-model comparable; shape deferred to v2, see below)
- [x] Diagnostics recorded: inter-run JS94 reproducibility (noise floor) and
  asymmetric LFC over-prediction (details in spec / notebooks)
- [ ] **v2 — common shape readout window (coverage-shape metric deferred to v2).**
  The shape metrics (Pearson + JS) are **not part of the v1 headline.** They are
  scored over each model's full output region (Yorzoi 3,000 bp vs Shorkie
  14,336 bp), so the cross-model shape numbers are invalid — JS especially is
  support-size dependent. v2 scores them over a fixed common window (≤ 3 kb,
  CDS-centred) for every model; the full receptive field still goes in as input,
  only the scored region is shared. The window-agnostic artifact makes this cheap:
  the benchmark already slices each construct at run time (`window_slice`), so add
  a second fixed scored-region slice for the shape metric. (Re-baselines shape
  numbers — deliberately.)

### Foreign-DNA integration

Meneu et al. foreign-DNA chromosome integration — spec
`docs/benchmarks/meneu_foreign_dna.md`. Implemented; Shorkie + Yorzoi run zero-shot
(PR #21).

- [x] Spec, build script, `MeneuForeignDNABenchmark` +
  `TiledCoverageTrackPredictor` protocol, Shorkie (T0) + Yorzoi (`illumina_exo`)
  adapters, registry, tests
- [x] Single task `meneu_foreign_dna` for both models — contigs tiled to each
  model's receptive field at run time from one window-agnostic
  `meneu_cov_<contig>.npz` (seq + fwd/rev) per contig (see "Unify window-split
  tasks" under Infrastructure)

### Code-structure refactor

Goal: models + tasks + thin adapters, not a fat adapter per (model × task).
Not blocking any benchmark.

- [x] Part 1 (PR #3): adapters take a `Yorzoi` / `Shorkie` wrapper instance; the
  wrappers own `from_pretrained` / RC averaging / the 8-fold ensemble / the
  strand swap (bit-identical, net −127 LOC)
- [x] Cross-task adapter consolidation — shared `MarginalizedLogSED` engine
  (`adapters/_marginalized_logsed.py`); `ShalemMarginalizedBase` and
  `MPRAMarginalizedBase` inherit it; Chen machinery in `_chen_marginalized.py`

#### Correctness sweep

- [x] All adapter families converted to per-base untransformed raw-count readout
  (brooks #12, eqtl #14, marginalized-logSED MPRA + Shalem, Wu #16, Hong #17,
  Chen #18); the `Yorzoi` / `Shorkie` wrappers own the inverse Borzoi transform
  (`yorzoi/yorzoi/utils.py`) + the per-base unbin. Re-baselining the resulting
  headline numbers → Pre-release cleanup; input-shift sensitivity → v2.

### Documentation

- [x] Spec per benchmark (`docs/benchmarks/*.md`) + index (`docs/benchmarks/README.md`)
- [ ] Final review and corrections of the `docs/benchmarks/*.md` spec files
- [ ] Each benchmark gets a figure — a graphical abstract of the task — in its
  `docs/benchmarks/*.md` spec
- [x] Architecture doc (`archive/architecture.md`)
- [ ] Revamp the main `README.md` — hero image / small logo, headline result
  figures + table, and a pass over the sections for correctness/completeness
- [ ] Bring the README extension guide ("Adding a new benchmark / model") in line with the code
- [x] Results summary — `ybench compare` writes `compare/summary.md` +
  `summary.csv`, auto-triggered after every run
- [ ] Per-stratum result tables auto-generated from saved scores

### Reproducibility

- [x] Run metadata (config hash, git commit, timestamp) per output dir
- [x] Raw scores / labels persisted (re-plot without re-scoring)
- [x] Data manifest with SHA256 lock (`src/yeastbench/data/manifest.lock.json`,
  `data/lock.py`)
- [x] `ybench data` CLI (PR #22): `get | status | verify | list | lock | publish
  | build` over a requester-pays GCS backend
  (`data/{cli,fetch,manifest,lock,backends}.py`); resolves shared refs
  (`R64-1-1.fa` / GTF, `R64-5-1.fa`) once; verifies each file against the lock;
  covered by `test_data_{backends,manifest}.py` + `test_fresh_install.py`
- [ ] Host the precomputed results for download, so the suite is usable without
  re-running on GPU — extend `ybench data` to fetch a results bundle, or publish
  it as a release / figshare asset. (Produced + pushed by the Pre-release cleanup
  step below.)

### Pre-release cleanup

Tasks before the first public release. First recompute and publish the final
result set; then strip the investigation / debugging artefacts (not part of the
public surface — specs, registered adapters, build scripts). `notebooks/` is
gitignored; remove from local checkouts too.

- [ ] Recompute all headline results on the converted (per-base, untransformed)
  code and push the final set to the hosted results store (see Reproducibility) —
  the conversion re-baselines
  every task (rank metrics ~stable; Pearson / absolute magnitudes shift). eqtl
  (Caudal / Kita) re-baseline still pending; confirm the marginalized / Wu / Hong
  / Chen headlines on the converted code (brooks done in #12, unchanged).
- [ ] `git rm` the investigation scripts:
  `scripts/chen/build_investigation_notebooks.py`,
  `scripts/cuperus/build_investigation_notebooks.py`,
  `scripts/cuperus/build_predictions_notebook.py`
- [ ] Remove the investigation notebooks (local only):
  `notebooks/chen_{shorkie,yorzoi}_investigation.ipynb`,
  `notebooks/brooks_yorzoi_coverage.ipynb`,
  `notebooks/wu_yorzoi_predictions.ipynb`,
  `notebooks/cuperus_{predictions,translation_features,data_summary}.ipynb`,
  `notebooks/hong_predictions_deep_dive.ipynb` (+ `.py`),
  `notebooks/_*cache*.pkl`, `notebooks/investigation_plots/`
- [ ] Verify `git ls-files | xargs grep -l "investigation" -- scripts/ src/` is
  empty and `tests/` imports none of the deleted files
- [ ] Confirm the specs reference the notebooks only as historical context
- [ ] Standardize the benchmark tables so they all share the same row names (consistent metric / row labels across every task's results table)

### Timon Cleaup Notes
- [x] Why does benchmarks/README say DREAM-RNN supervised baseline spec'd in the table
- [ ] Everything is zero-shot except for the IntProp thing, right? I think it would be better to have everything consistent

## v2 release

### ExoShorkie integration

ExoShorkie is a transfer-learning extension of Shorkie on
exogenous-RNA-seq-in-yeast (Mandl & Orenstein 2026). Most tasks in the suite are
exogenous-/foreign-DNA-in-yeast measurements — its training distribution — so it
is expected to match or beat vanilla Shorkie / Yorzoi. Code + weights vendored in
`shorkie-paper/`. (Deferred 2026-06-09; all ExoShorkie benchmarking moved here
2026-06-13.)

Prior WIP on branch `exoshorkie/distill` (PR #20, unmerged): Dense(1) head port +
a 6-genome distillation pipeline + distilled students + thin per-task adapters —
none of it on the main line yet. Open before promoting:

- *Distillation:* settle the final recipe (the WIP distils one student per genome,
  gate r > 0.98, then ensembles the 6 + RC) and validate it.
- *Merge + validate:* land the head port + adapters on the main line and re-run the
  per-task adapters below against current code.

Per-task adapters (once integration lands) — each is a thin adapter on an
`ExoShorkie` wrapper reusing the task's existing scaffold; only the model changes.
Migrate back under their benchmark sections when ExoShorkie becomes active.

- [ ] Rafi / de Boer marginalized (`SequenceExpressionScorer`)
- [ ] Shalem terminator (`TerminatorMarginalizedExpressionPredictor`)
- [ ] Chen synonymous CDS (`LocalCodingVariantPredictor`)
- [ ] Wu RFP insertions (`CassetteExpressionPredictor`)
- [ ] Hong IGR insertions (`IGRInsertionExpressionPredictor`)
- [ ] Cuperus 5′ UTR (`FivePrimeUtrReporterExpressionPredictor`)
- [ ] Lee YTK promoters (`IntegratedPromoterPanelPredictor`)
- [ ] Brooks SCRaMBLE (`CoverageTrackPredictor`)
- [ ] Meneu foreign DNA (`TiledCoverageTrackPredictor`)
- [ ] Meneu: check ExoShorkie vs the paper (reproduce the paper's exact metric —
  median Spearman over 16 bp bins of 14.3 kb windows — then alignment / scale,
  track subsets, NatShorkie strand-adaptation; doubles as the check on why our
  zero-shot Shorkie / Yorzoi sit below the published refs). See
  `docs/benchmarks/meneu_foreign_dna.md` Open Questions #1

### Species LM

Sequence language-model evaluation on yeast (Keren et al.). No spec or adapter
yet; gets `docs/benchmarks/species_lm.md` + a registry entry when it lands.

### Condition coherence

Does the model predict near-zero coverage at promoters that *should* be tightly
repressed in its training conditions — i.e. has it learned promoter-driven
condition logic, or does it read expression as a CDS-intrinsic property?
Motivating result (2026-05-22): both models correctly predict near-zero coverage
at native GAL1 in glucose, but swapping GAL1's CDS for a codon-optimized GFP at
the same locus pushes the predicted CDS-sum up ~29× (Shorkie) / ~11× (Yorzoi) —
same promoter, only the CDS changed. The first benchmark probing **absolute**
expression calibration rather than variant-effect ranking. Test set: repressed
promoter-CDS-terminator units (GAL, PHO, MET, MAL, anaerobic, heat-shock, mating)
vs constitutive controls (TDH3, PGK1, ACT1, ENO2, ALG9), scored on the unmodified
locus. Primary metric: pairwise AUROC (on-gene ranked above off-gene); secondary:
off-gene CDS-sum normalized to the on-set median, per gene-class.

- [ ] Spec `docs/benchmarks/condition_coherence.md`
- [ ] `LocusExpressionPredictor` protocol + curated gene list + benchmark

### Caudal eQTL effect-size calibration

The Caudal benchmark scores discrimination (`|score|` → AUROC/AUPRC) but not
magnitude. Plot the predicted variant-effect score against the measured GWAS
effect (`SnpWeight` β) for the positive eQTLs, faceted by target-gene strand and
distance-to-TSS bin (both columns in `negset_*.tsv`): the distance facet checks
the effect-size decay, the strand facet doubles as an adapter strand-handling
diagnostic.

- [ ] Per-(gene-strand, distance-bin) effect-size calibration plot

### Caudal eQTL positive-set cleanup

Deferred from the v1 ploidy/VAF audit (`scripts/eqtl/audit/`). The positives are
called from the 1011-isolate panel of mixed ploidy and zygosity; the confound is
real but small (clean vs not-clean barely moves Yorzoi), so v1 ships the full set
and v2 cleans it up.

- [ ] Re-normalise / drop the multiallelic-complex positives mislabelled `SNP`
  (already excluded by negset matching — fix for an honest count)
- [ ] Ship ploidy/zygosity strata as an optional diagnostic stratifier (full set
  primary, clean-carrier subset secondary), not a hard filter
- [ ] Re-run the clean vs not-clean split on Shorkie to confirm the v1 decision

### Per-stratum reporting and bootstrap CIs

Stratum-level analysis for the MPRA benchmarks. With n ≈ 71k a full-size bootstrap
CI is degenerate (≈ ±0.002 on Pearson r), so resample a 10 % subsample per stratum
(10 000×, percentile CIs); decide whether to rescale by √(m/n) for a true full-n
CI or keep it as a conservative band.

- [ ] Shalem — parse the `Description` / `SetName` values into ~9 coarse strata
  (RBP random, scanning mut-pos/neg/quantile, native 3′ UTRs, motif moves, GC
  variants) and report per stratum
- [ ] Rafi / de Boer marginalized — bootstrap CIs per stratum (8 strata incl. the
  SNV / motif pair strata)
- [ ] Shalem — bootstrap CIs per stratum

### Hong et al. extensions

Follow-ups to the implemented Hong IGR benchmark (spec `docs/benchmarks/hong_igr.md`).

- [ ] YeIP supervised reference baseline (AutoGluon over Hong Fig. 2A features;
  supervised upper bound; weights + code from the YeIP repo)
- [ ] External validation on Kong et al. 2022's independent IGR panel
- [ ] Promoter × IGR sub-task (Hong Fig. 2G–I; 6 promoters × 6 IGRs)
- [ ] Carbon-source sub-task (Hong Supp Fig. S11; blocked on a training-track
  metadata audit — not-applicable for any condition without matched tracks)

### Brooks et al. extensions

Follow-ups to the implemented Brooks SCRaMBLE benchmark (spec
`docs/benchmarks/brooks_scramble.md`).

**Leakage-free Yorzoi evaluation.** Yorzoi's training targets include the Brooks
Nanopore tracks (manifest verified 2026-05-20), so its headline is partly a
leakage measurement, not zero-shot — both the LFCs and the shape
metrics read back tracks the model was trained on. Shorkie is clean (T0 RNA-seq
tracks only). The hard part: Brooks is genuinely informative training data, so a
clean test set that still permits training is unresolved. The native genome is
byte-identical across strains, so it can't be held out — the only novel signal is
the cis-effect of each strain's new junctions within the receptive field.

- [ ] Define a leakage-clean split — hold a set of SCRaMBLE strains' Nanopore
  tracks out of (re)training and report the Yorzoi headline only on those
  held-out strains
- [ ] Interim: post-hoc strain mask on the track indices for the current Yorzoi
  weights (quantify the leakage gap without retraining)
- [ ] Context-stratified headline — report metrics on (low-hamming, high-hamming)
  alt-vs-native subsets separately; wire in once `hamming(alt, native)` is a
  stored column
- [ ] JS707 / JS710 recovery (low priority) — both strains yield zero samples
  because their GFF gene IDs don't match JS94's; recover by matching genes on
  coordinate homology instead of `gene_id` equality

### Meneu et al. extensions

Follow-ups to the implemented Meneu foreign-DNA benchmark (spec
`docs/benchmarks/meneu_foreign_dna.md`).

- [ ] Yorzoi track-group sweep — add the Nanopore-63 and SRA-1 modes to
  `_plus_axis_indices`, report each, name the best

### Bin-boundary sensitivity

Input-shift / bin-boundary sensitivity check (Karollus et al. 2023). Shift the
input window so a TSS / motif / CDS edge crosses a model bin (Yorzoi 10 bp,
Shorkie 16 bp) and measure how much the prediction moves; if it moves materially,
add a jittered / shift-averaged readout (or at least report prediction variance
under small shifts). Relevant to every binned model in the suite.

- [ ] Quantify input-shift sensitivity per adapter; add a jittered readout if
  material

### Random short promoters in genome
https://www.nature.com/articles/s41467-023-37610-w

### Baselines as first-class models

- [ ] Flatten baselines to first-class models — drop `adapters/baselines/`
  (`cai.py` / `codon_transformer.py` move up) and the `_build_*_baseline` naming;
  no `category` tag. Cosmetic: all five models already share the `MODELS`
  registry / dispatch / protocols. Also resolves the `dream_rnn` folder/naming
  inconsistency and the one hardcoded `dream_rnn` case in `mpra.py:compare_plot`.

### CI smoke-test

- [ ] CI smoke-test on synthetic data (no GPU) — wire `test_fresh_install` into CI
  (today CI runs only ruff F401, `.github/workflows/lint.yml`)
