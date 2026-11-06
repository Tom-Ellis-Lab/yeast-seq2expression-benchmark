# Brooks et al. — SCRaMBLE structural-rearrangement expression effect

![image](../../img/scramble_vertical.drawio.svg)

## At a glance

| | |
| --- | --- |
| **Task** | Predict how a SCRaMBLE structural rearrangement changes a rearranged gene's expression, from sequence. The v1 metric is the scalar log-fold-change of CDS coverage (rearranged vs unscrambled control). A coverage-*profile* shape metric is **deferred to v2** — see [Open questions](#open-questions--future-work). |
| **Source** | Brooks *et al.* 2022, *Transcriptional neighborhoods regulate transcript isoform lengths and expression levels*, Science 375(6584). DOI: [10.1126/science.abg0162](https://doi.org/10.1126/science.abg0162). |
| **Reference assembly** | Per-strain SCRaMBLE assemblies: native `chrI–chrXVI` (byte-identical across strains) plus one synthetic synIXR contig `JS<strain>_1`. Parental control: JS94 (expression) / JS96 (sequence). |
| **Assay** | Long-read Oxford Nanopore **direct RNA-seq** of multiple strains, each carrying a rearranged synthetic chr IX right arm (synIXR, ~91 kb, 43 loxPsym segments; loxPsym sits 3 bp after the stop codon of every nonessential CDS). Cre induces deletions / duplications / inversions / translocations. A rearranged CDS **keeps its native promoter but is decoupled from its native 3′UTR/downstream**, so the cis-predictable effect is principally the *new downstream context*. |
| **Control** | **JS94** = parental −SCRaMBLE strain (synIXR, no induced recombination). Same genetic background; the only valid "before" for the rearrangement effect (not BY4741, which has native chr IX). |
| **Eval set** | One **(gene × strain × copy)** sample — see [Eval-set definition](#eval-set-definition). ~58 SCRaMBLE strains available (not the Yorzoi-paper 5); sample set defined by an objective locked rule, not hand-picked. |
| **Primary metric** | LFC: **direction balanced accuracy** of sign(LFC), then Spearman ρ, then Pearson r, all on (pred, true) LFC across samples, read against the JS94 reproducibility ceiling. |
| **Secondary metric (deferred to v2)** | Coverage-profile shape (Yorzoi only): per-window Pearson + Jensen–Shannon divergence. Deferred to v2 — the shape readout window differs across models, so cross-model shape numbers aren't comparable yet (see [Open questions](#open-questions--future-work)). Not part of the v1 headline. |
| **Adapter protocol** | `CoverageTrackPredictor.predict_coverage_batch` (`src/yeastbench/adapters/protocols.py`) — per-base coverage per construct; the benchmark derives the CDS-sum LFC from it (v1); the shape profile is a v2 metric. |

## Contents

- [Why this benchmark exists](#why-this-benchmark-exists)
- [Results](#results)
- [Data](#data)
- [What's scored](#whats-scored)
- [Model contract](#model-contract)
- [Evaluation & metrics](#evaluation--metrics)
- [Files & registry](#files--registry)
- [Open questions / future work](#open-questions--future-work)

## Why this benchmark exists

Every other benchmark perturbs *local* cis-sequence. SCRaMBLE perturbs
**genome architecture** while leaving each gene's CDS and promoter
intact, moving it next to new neighbours / new downstream termination.
It is the genome-scale companion to the Wu position-effect benchmark
(Wu: constant cassette, moving locus; Brooks: constant gene, rearranged
chromosome). The paper's own Fig. 4 GBRT is the key prior: a model with
**measured transcriptional-neighbourhood** features predicts Δexpression
far better than a **sequence-only** model. A cis sequence-to-expression
model is, in effect, the sequence-only regime — so this benchmark
measures the **cis-predictable fraction** of rearrangement-induced
expression change. That ceiling is intrinsic and is reported explicitly
([below](#evaluation--metrics)), not hidden.

## Results

*2026-07-31 Modal run (v1, unified `brooks_scramble` task, 3-file artifact), recomputed per replicate 2026-09-02 from the saved per-replicate arrays — no model re-run. The cohort is the shared-cohort intersection (n = 327 scored) — the apples-to-apples comparison.*

Metrics are reported **per JS94 parental run** and are never averaged across
the three. Each run scores a different set of genes: a gene needs
`MIN_READS_PER_RUN` reads *in that run*, and because the cutoff falls per gene,
the shallow run's cohort is the strongly expressed genes. The three cohorts
therefore differ in size and in composition, and each carries its own
leave-one-out ceiling — so the three columns are not comparable to one another
either. Read each model against the ceiling in its own row, never across rows.

| LFC, shared cohort | JS94_r0 (n = 205) | JS94_r1 (n = 80) | JS94_r2 (n = 291) |
| --- | ---: | ---: | ---: |
| **direction balanced accuracy** | | | |
| Shorkie | 0.514 | 0.613 | 0.534 |
| Yorzoi | **0.624** | **0.650** | **0.630** |
| *reproducibility ceiling* | *0.761* | *0.963* | *0.694* |
| **Pearson r** | | | |
| Shorkie | −0.087 | 0.004 | 0.023 |
| Yorzoi | **0.297** | **0.212** | **0.153** |
| *reproducibility ceiling* | *0.789* | *0.954* | *0.671* |
| **Spearman ρ** | | | |
| Shorkie | −0.065 | −0.157 | 0.109 |
| Yorzoi | **0.439** | **0.275** | **0.224** |
| *reproducibility ceiling* | *0.779* | *0.955* | *0.589* |

The ceiling is computed on its own cohort (n = 198 / 80 / 169), which is not the
scoring cohort: it needs the *other* replicates to be finite, where the model
metric needs a finite prediction.

![Shared-cohort LFC per JS94 replicate, Shorkie vs Yorzoi, each against its own leave-one-out reproducibility ceiling.](../../img/results/brooks_scramble/shared_lfc.svg)
![Per-SCRaMBLE-strain LFC agreement on the shared cohort.](../../img/results/brooks_scramble/shared_per_sample.svg)

- Yorzoi leads Shorkie on every metric in every one of the three parental runs.
- Yorzoi recovers a real but modest slice of the cis-predictable LFC, and the
  slice shrinks as the reference run gets deeper and its cohort widens (r 0.297
  on r0's 205 genes, 0.153 on r2's 291). Against ceilings of 0.789 / 0.954 /
  0.671 it is well short throughout.
- Shorkie sits at ~0 on LFC — a readout limit, not a wrong call. The rearrangement keeps each gene's promoter and CDS intact and only changes its downstream context; Shorkie's score is RNA-seq coverage summed over the (unchanged) CDS, so its predicted rearranged and parental coverage come out nearly equal and the predicted log-fold-change is ~0 (see [Model contract](#model-contract)). Its magnitude is better-calibrated than Yorzoi's (mean |z| 2.0 vs 11.2), but that's calibration, not ranking.

Why the numbers look the way they do: [`model_failures.md`](model_failures.md). Artifacts: `results/brooks/compare/per_task/brooks_scramble/summary.json` (shared cohort) and `results/brooks/{shorkie,yorzoi}__*/summary.json`.

## Data

SCRaMBLE rearranges the synthetic chr IX right arm (synIXR) and each strain
is assayed by long-read Nanopore direct RNA-seq. This section covers the raw
per-strain data and how it is normalized into per-copy log-fold-changes.

### Raw data and preprocessing

- **Per-strain assemblies, no junction-walking needed.** Every strain's
  genome = the *byte-identical* native genome (`chrI–chrXVI`, `chrIXL`;
  identical contig sizes in every strain) **plus one synthetic contig**
  `JS<strain>_1` = its SCRaMBLEd synIXR. `JS94_1` = 98,752 bp
  (unrearranged parental); e.g. `JS606_1` = 165,327 bp (larger →
  duplications). The rearranged construct is just the focal copy's
  window in `JS<S>_1`; the native construct is the same gene in
  `JS94_1`. Table S3 junctions are needed only to *classify*
  rearrangement type — junctions are otherwise derivable by diffing
  `JS<S>_1` against `JS94_1`.
- **Per-strain GFF** annotates each synIXR gene with SGD ID,
  `essential_status`, `orf_classification`, and synthetic features.
- **Alignments** are per-strain read BEDs:
  `chrom start end read_id MAPQ strand`. MAPQ on `JS<n>_1` is
  ≈ all 60, **no MAPQ-0 multimappers** — "distinguished" keeps
  uniquely-assignable reads, so per-copy coverage by coordinate
  interval is well-defined.
- **Control sequence from JS96.** There is **no `JS94` FASTA**
  (only `.fai`/`.genome`). `JS96_1` is the parental synIXR: identical
  98,752 bp and byte-identical synIXR GFF coordinates to `JS94_1` (a
  rearrangement would change these). → JS96 supplies the parental
  *sequence*; JS94 runs supply the parental *expression* (shared
  coordinate system).
- **Replicates / run-depth filter.** `JS94` has 8 BED runs, but 3 are
  **rrp6Δ/xrn1Δ RNA-decay-mutant** libraries — *not* −SCRaMBLE WT and
  excluded by the strict run pattern. Of the 5 WT runs, 2 are
  failed/ultra-shallow (651 and 3,874 reads); a per-run native-library
  floor (`MIN_RUN_READS`, 50 k) drops them, leaving **3 deep WT JS94
  runs** (2018-02-14 / 06-28 / 12-03) for the denominator and the
  reproducibility ceiling. Same depth filter applied to every SCRaMBLE
  strain's runs.
- **Size factor.** The native genome is **genetically identical in every
  strain**, so library-depth and batch differences are removed with a
  **total-native-reads size factor** (reads on `chrI–chrXVI` minus
  synIXR, plus `chrIXL`; `chrMT` and the synthetic contig excluded) per
  strain/run, relative to the JS94-deep-run mean. Simple and robust, no
  per-gene modelling. (DESeq median-of-ratios is a documented v2
  refinement.)
- **ERCC92** is in the genome FASTAs but **0 ERCC reads survive in the
  processed BEDs** → spike-in normalisation is unavailable; the
  native-genome size factor is the correct tool regardless.

### Confounds

Raw `cov(SCRaMBLE)/cov(JS94)` is dominated by artefacts; the design
neutralises each:

| Confound | Reality in this data | Handling |
| --- | --- | --- |
| **Library depth + sequencing batch** | JS94 ≈ 0.30 M reads (2018); JS606 ≈ 1.43 M (2019). ~5× depth + a year-apart batch. | Total-native-reads size factor per strain/run (see [Raw data and preprocessing](#raw-data-and-preprocessing)), relative to the JS94-deep-run mean. |
| **Failed / wrong-condition runs** | `JS94` mixes in rrp6Δ/xrn1Δ decay-mutant libraries and 2 failed WT runs (651 / 3,874 reads). | Strict run pattern excludes the tagged decay-mutant libraries; `MIN_RUN_READS` floor drops failed/ultra-shallow runs (control *and* strain) before they corrupt the denominator/size-factor/ceiling. |
| **Technical noise / sparsity** | Nanopore direct-RNA is sparse on the small synIXR contig; many genes get <10 reads. | Per-sample `low_support` flag (CDS reads `< MIN_READS` in strain or JS94-mean); benchmark can restrict to well-supported samples. Mean of the **3 deep JS94 runs** for a stable denominator; per-run JS94 normalised coverages are stored so the **control–control reproducibility ceiling** is derivable and every model number is read against it. |
| **Trans / neighbourhood transcription** | Paper Fig. 4: measured neighbourhood transcription, not sequence, dominates Δexpr. | Not removable — an intrinsic ceiling. Mitigate by restricting to genes whose **cis context changes within the model's receptive field** (the loxPsym-after-stop design puts the cis effect in the *new downstream junction*). Frame the metric as the cis-predictable fraction and report it against the reproducibility ceiling, never against 1.0. |
| **Copy-number / dosage** | Duplications/deletions change dosage; one window can't encode it. | **Dissolved by per-copy sampling** (next section): each copy is its own single-copy sample with its own downstream junction; no dosage term. |
| **Deletions** | Gene absent in strain → no CDS to centre a window on; "expression≈0 because deleted" is trivially predictable. | **Excluded** from the predictive set (optional separate sanity check, not the benchmark). |

### Per-copy sampling

Because loxPsym sits 3 bp past the stop, **every rearrangement junction
gives the upstream gene a new downstream context** — duplicate copies of
a gene are virtually always *context-distinct*, not identical-dosage
duplicates. Reads are uniquely placed (MAPQ ≈ 60) and each copy occupies
distinct coordinates on `JS<S>_1`, so per-copy coverage is computable by
interval:

- `true LFC` for a copy = `log2( norm_cov(this copy's CDS, strain) /
  norm_cov(the gene's single CDS, JS94) )`.
- The model predicts that one copy's window (its specific downstream
  junction). Two copies of a gene → two independent cis samples.

Guardrails: require unique mapping + minimum per-copy read support; drop
the rare case where two copies are byte-identical within the receptive
field (not separable, not a distinct cis question).

### Eval-set definition

A `(gene, strain, copy)` is an evaluated sample iff, in that strain's
assembly/GFF:

1. the copy has an **intact CDS** and retains its **native promoter**;
2. its **downstream context within ± the model receptive field differs
   from JS94** (a new junction inside the window — i.e. the cis input
   actually changed);
3. it is **uniquely mappable** and has **≥ (threshold) reads** in both
   the strain and JS94 (threshold pinned empirically — open question);
4. not byte-identical to another retained copy within the window;
5. **deletions excluded**.

**Window-agnostic, self-contained artifact.** The build script (run once)
resolves every `(gene, strain, copy)` candidate and bakes a **three-file**
artifact under `data/tasks/brooks_scramble/`: `brooks_index.tsv` (one row
per construct: ids/strand/class, the **absolute** alt + native coords and
provenance — source genome / contig / slice offset — `true_lfc`, the
per-run JS94 replicate normalised CDS coverages so the reproducibility
ceiling is derivable, strain/JS94 read counts, native-genome size
factors), `brooks_constructs.fasta` (a generous gene-centred slice of alt
+ native sequence, ±FLANK bp), and `brooks_cov.npz` (the matching per-base
coverage). At eval time the benchmark re-cuts each construct to the model's
receptive field (`window_slice`) and re-applies the window-dependent
membership (alt + native fit, `alt != native`, dedup byte-identical copies
per gene). It depends on **these files alone** — no per-strain genomes /
GFF / BED, no R64 reference. One artifact serves any model with window ≤
FLANK (16384).

## What's scored

Each sample is one rearranged gene copy, scored against the same gene in
the parental control. The model is given two gene-centred windows and
predicts coverage for each:

- **Alt (rearranged), primary:** the model-window from `JS<S>_1`, **gene
  centred** on the focal copy's CDS, carrying its specific downstream
  junction and as much native flank as the window holds.
- **Native (baseline):** the same gene **centred** in `JS94_1` (parental
  −SCRaMBLE synIXR).
- Window length per model (Yorzoi 4992 bp); gene-centred so up- and
  downstream context are balanced (the CDS itself is short; the
  informative variation is at TSS/TES and the new junction, *outside* the
  CDS — see the shape metric in [Evaluation &
  metrics](#evaluation--metrics)).

## Model contract

```python
class CoverageTrackPredictor(Protocol):
    seq_len: int
    crop_bp_each_side: int
    batch_size: int
    varies_by_strain: bool   # False for models with no strain-specific tracks (e.g. Shorkie)

    def predict_coverage_batch(
        self,
        seqs: Sequence[str],
        strands: Sequence[str],
        strains: Sequence[str | None] | None = None,
    ) -> np.ndarray:
        """Per-base predicted RNA-seq-like coverage over each input window,
        in raw count units (adapters invert any model-specific transform
        and unbin to per-base). For track-based models, `strains` routes
        each construct to the matching experimental tracks; `varies_by_strain
        = False` lets the benchmark predict the native once and reuse it."""
```

The benchmark builds the gene-centred alt and native window strings and
the in-window CDS interval, calls `predict_coverage_batch` for each, then
forms the CDS-sum LFC (the v1 metric; the full-window shape is a deferred v2
metric).

**Shorkie scores via a proxy track and sits at ~0 on LFC.** Shorkie has no
track matching the Nanopore direct-RNA assay, so it runs LFC on a **T0
RNA-seq proxy track** (mean of the 384 unstranded T0 RNA-seq coverage
tracks). This ran and scored — n = 528 on the full set, 327 on the shared
cohort — but its predicted log-fold-change is ~0: the readout is
CDS-summed coverage, driven by the gene's promoter and CDS, which the
rearrangement leaves intact, so Shorkie's predicted rearranged and
parental coverage come out nearly equal. (`varies_by_strain = False` in the
adapter is only a compute shortcut — Shorkie has no strain-specific tracks,
so the parental prediction is computed once and reused across the JS94
replicates — it is **not** the cause of the ~0 score.)

## Evaluation & metrics

### LFC — scalar effect size (Yorzoi primary; Shorkie via T0 RNA-seq proxy track)

1. Per sample **and per JS94 parental run k**:
   `pred_LFC_k = log2( Σ_pred(alt CDS bins) / Σ_pred(native CDS bins, run k) )`;
   `true_LFC_k = log2( norm_cov_strain / norm_cov_js94_k )` from
   native-normalised Nanopore CDS coverage (per-copy), gene strand. Run `k`
   contributes only if the gene cleared `MIN_READS_PER_RUN` reads in it, so a
   sample carries 0–3 paired LFCs.
2. Metrics, in order: **(1) direction balanced accuracy** —
   `sign(pred_LFC_k)` vs `sign(true_LFC_k)`; **(2) Spearman ρ**;
   **(3) Pearson r** — each computed **within one run k**, across the samples
   that run supports.
3. **The three runs are reported separately and never averaged.** Averaging
   correlations across them would weight unequal cohorts equally (205 / 80 /
   291 samples here) and mix gene sets of different composition, since the
   read cutoff is decided per gene: the shallow run scores only the strongly
   expressed genes, and its ceiling is correspondingly higher. `n_scored_per_rep`
   ships alongside every number so the cohort behind it is always visible.
4. Each run is read against **its own** leave-one-out ceiling — run `k`'s true
   LFCs predicted from the mean of the other runs'. Never compare a model's
   number in one run to a ceiling from another.
5. Plot: predicted-vs-true LFC scatter (points are per-sample means with
   replicate envelopes), sign quadrants shaded, the **JS94×3 reproducibility
   band** overlaid, per-run r/ρ/acc annotated; plus a per-rearrangement-class
   breakdown.

### Shape — coverage profile (Yorzoi-only) — deferred to v2

> **Deferred to v2.** The shape metric is not part of the v1 headline; it is
> specified here for v2. The blocker: Pearson and JS are scored over each model's
> full output region (Yorzoi 3,000 bp vs Shorkie 14,336 bp), and JS divergence
> depends on support size, so cross-model shape numbers aren't comparable until a
> common scored window (≤ 3 kb, CDS-centred) is fixed.

Over the **full gene-centred window** (not CDS-only — the CDS profile is
usually flat; the signal is at TSS/TES/junction), at Yorzoi bin
resolution, strand-matched, on the **alt** construct (native as a sanity
baseline):

- **Pearson correlation** of predicted vs true binned coverage vectors
  (sensitive to peak co-location).
- **Jensen–Shannon divergence** as the **headline shape metric** —
  symmetric, bounded `[0, log2]`, finite without smoothing (each side
  compared to the mixture `M=½(P+Q)`, so the hard zeros in sparse
  Nanopore coverage cannot blow it up), and therefore comparable and
  averageable across samples of different depth. Computed on
  sum-1-normalised coverage vectors.
- **KL divergence** `D_KL(true‖pred)` (ε-smoothed) reported only as a
  *secondary* directional view ("model misses real signal"); not the
  headline because it is asymmetric, unbounded, undefined on zeros, and
  not comparable across samples. (`JSD = ½D_KL(P‖M)+½D_KL(Q‖M)`,
  `M=½(P+Q)`.) Both metrics ignore magnitude (the LFC metrics carry
  that); Pearson and JS are complementary — peak co-location vs
  mass-placement.

Report shape metrics against the **JS94×3 control–control** Pearson/JS
ceiling. Example loci plotted (true vs predicted profile, alt and native
overlaid), as in the Yorzoi Wu dump notebook.

### Reference ceiling & baseline

The headline is always read against the **JS94×3 control–control
reproducibility ceiling** (LFC dir-acc / Pearson) — a
model cannot beat the assay's own test–retest. The Yorzoi paper's own
numbers (r = 0.33, ρ = 0.32, balanced accuracy = 0.62) are **orientation
only, not a direct comparison target**: our sample set is the objective
locked rule, not the paper's hand-picked 41.

## Files & registry

### Raw upstream (build-time only)
- `scripts/brooks/build_brooks_distribution.py` — one-time builder: JS96
  parental sequence + JS94 deep-WT runs, run-depth filter, total-native-reads
  size factor, locked per-copy sample rule, gene-centred alt/native
  constructs, `true_lfc` + per-run JS94 coverages → the three-file artifact.

### Processed distribution (the sole run-time dependency)
Window-agnostic three-file artifact under `data/tasks/brooks_scramble/`, all
**built**; the benchmark reads nothing else (no genomes, no R64):
- `brooks_index.tsv` — one row per `(gene, strain, copy)` candidate: ids/strand/
  `rearr_class`/`n_copies`, the absolute alt + native coords and provenance
  (`*_source_genome`, `*_contig`, `*_contig_len`, `*_cds_start/end`,
  `*_slice_start`), `true_lfc`, `norm_cov_js94_runs` (→ ceiling derivable),
  read counts, size factor, `low_support`.
- `brooks_constructs.fasta` — `alt~<sample_id>` / `native~<gene_id>` generous
  gene-centred slices (±FLANK = 16384 bp, clamped to the contig; native deduped
  per gene).
- `brooks_cov.npz` — per-base int32 coverage, same record keys.

The benchmark re-cuts each construct to the adapter's receptive field at run
time and re-applies the per-window membership/dedup (bit-identical to the old
4992 / 16384 TSVs: 698 / 1055 constructs). Full panel built with `--strains all`
(~56 strains → 1786 candidate constructs).

Run via the `ybench` CLI — see the [quickstart](../../README.md#quickstart)
and the [CLI reference](../cli.md). Adapters live at
`src/yeastbench/adapters/{shorkie,yorzoi}_brooks.py`; the benchmark +
plots at `src/yeastbench/benchmarks/brooks.py`; adding a model or task is
one registry entry — see [`extending.md`](../extending.md).

## Open questions / future work

1. **Thresholds** — `MIN_RUN_READS`=50 k and `MIN_READS`=10 set
   pragmatically; pin empirically from the JS94 deep-run noise floor
   (the count at which control–control LFC variance is acceptable).
1a. **JS707 → 0 samples** — its single run passed depth but no gene met
   "present in JS94 + alt window ≠ native within ±2496 bp". Check
   whether the rule is too strict (e.g. junction just outside the
   gene-centred window) before scaling.
1b. **Scale to `--strains all`** — v1 built on 4 strains (37 samples);
   the objective rule over the full ~58-strain panel is the real n.
   Heavier download/compute; run before locking the headline.
1c. **Median-of-ratios size factor** — total-native-reads is the v1
   normaliser (simple/portable); MoR over native genes is a documented
   v2 refinement if a global trans shift is observed.
2. **Shorkie LFC on the T0 proxy track** — implemented and ran (n = 528
   full / 327 shared). It scores ~0 because its CDS-coverage readout is
   set by the unchanged promoter+CDS, not the rearranged downstream
   context; the open question is whether a path that actually reads the
   changed context (or a strain-aware track) is worth building.
3. **Rearrangement-type classification** — derive from `JS<S>_1` vs
   `JS94_1` diff; cross-check against Table S3 if obtainable.
4. **Antisense / both-strand coverage** — the paper uses both-strand
   cosine similarity for neighbourhood. v1 is gene-strand only; revisit
   adding the antisense channel to the shape metric as a v2 extension.
5. **Receptive-field window length** — Yorzoi 4992 bp; confirm the
   gene-centred placement keeps the relevant new junction in-window for
   the bulk of samples (drop / flag those where it doesn't).
6. **JS94 run handling** — mean of the 3 runs vs per-run pairing for the
   denominator and the ceiling; pin in the build script.
7. **Coverage-shape metric deferred to v2.** The shape profile (per-window
   Pearson + JS divergence) is not in the v1 headline. It is scored over each
   model's full output region (Yorzoi 3,000 bp vs Shorkie 14,336 bp), and JS
   divergence depends on support size, so the cross-model numbers aren't
   comparable. v2 scores shape over a fixed common window (≤ 3 kb, CDS-centred —
   the full receptive field still goes in as input, only the scored region is
   shared), re-baselines the numbers, and promotes shape to a reported secondary
   metric.
