# Benchmark results

Cross-model comparisons, one section per benchmark. Each table shows the
benchmark's **primary metric(s)** for every model that has run; the leading
value is **bold** and named in the per-section *Winner* line.

Conventions: higher is better unless a metric is marked ↓ (lower is better) or
the benchmark is a *negative result* (both models at chance / ≈ 0). Full
per-metric tables for each comparison live next to the runs in
`results/<group>/compare/summary.md`.

## Summary

| Benchmark | Models | Primary metric | Winner |
| --- | --- | --- | --- |
| [Caudal eQTL](#caudal-eqtl) | Shorkie, Yorzoi | \|score\| AUROC / AUPRC | **Shorkie** |
| [Kita eQTL](#kita-eqtl) | Shorkie, Yorzoi | AUROC / AUPRC | **Shorkie** |
| [Rafi / deBoer MPRA (promoter)](#rafi--deboer-mpra-promoter) | Shorkie, Yorzoi, DREAM-RNN | Pearson r / Spearman ρ | **Shorkie** (zero-shot) |
| [Shalem MPRA (terminator)](#shalem-mpra-terminator) | Shorkie, Yorzoi | Pearson r / Spearman ρ | **Yorzoi** |
| [Chen synonymous MPRA](#chen-synonymous-mpra) | CAI, CodonTransformer, Shorkie, Yorzoi | Pearson r / Spearman ρ | **Mixed** (Shorkie on GFP, CAI on TDH3) |
| [Wu RFP insertions](#wu-rfp-insertions) | Shorkie, Yorzoi | Pearson r / Spearman ρ | **Neither** (both ≈ 0) |
| [Hong IGR insertions](#hong-igr-insertions) | Shorkie, Yorzoi | Spearman ρ (held-out test set) | **Neither** (both ≈ 0) |
| [Gui GFP RNA](#gui-gfp-rna) | Shorkie, Shorkie random init, Yorzoi, ExoShorkie ensemble | Spearman ρ | **None established** (all 95% CIs include zero) |
| [Brooks SCRaMBLE](#brooks-scramble) | Shorkie, Yorzoi, ExoShorkie, Shorkie random-init | LFC direction balanced acc., per JS94 run | **Yorzoi** (all 3 runs) |
| [Cuperus 5′-UTR](#cuperus-5-utr) | Shorkie, Yorzoi | Spearman ρ (+ partial correlation) | **Shorkie** |
| [Lee YTK promoters](#lee-ytk-promoters) | Shorkie, Yorzoi | consensus dynamic-range recovery + Spearman ρ | **Shorkie** |
| [Meneu foreign DNA](#meneu-foreign-dna) | Shorkie, Yorzoi | per-window shape Pearson / JS↓ | **Yorzoi** |

---

## Caudal eQTL

Rank true single-nucleotide *cis*-eQTLs above distance-matched controls. Mean ±
SEM over 4 negative sets. ([spec](benchmarks/caudal_eqtl.md))

| metric | Shorkie | Yorzoi |
| --- | --- | --- |
| \|score\| AUROC | **0.5673 ± 0.0015** | 0.5296 ± 0.0007 |
| \|score\| AUPRC | **0.5793 ± 0.0022** | 0.5364 ± 0.0015 |

**Winner: Shorkie** — leads on both AUROC and AUPRC.

## Kita eQTL

Same task as Caudal on an independent positive panel (Promoter/UTR5/UTR3/ORF,
≤ 8 kb of TSS). ([spec](benchmarks/kita_eqtl.md))

Zero-shot, 2026-07-31 Modal run. Single-nucleotide only (605 pairs per negative set).

| metric | Shorkie | Yorzoi |
| --- | --- | --- |
| \|score\| AUROC | **0.6668 ± 0.0087** | 0.6008 ± 0.0028 |
| \|score\| AUPRC | **0.6399 ± 0.0097** | 0.5744 ± 0.0048 |

**Winner: Shorkie** — leads on both, the same ordering as Caudal.

## Rafi / deBoer MPRA (promoter)

~71k random 80 bp promoters, scored by marginalizing the insert's logSED effect
across native host-gene contexts. ([spec](benchmarks/rafi_mpra_promoter.md))

| metric | Shorkie | Yorzoi | DREAM-RNN (supervised ref) |
| --- | --- | --- | --- |
| Pearson r | **0.7595** | 0.6062 | 0.9719 |
| Spearman ρ | **0.7747** | 0.6254 | 0.9752 |

**Winner (zero-shot): Shorkie** — leads Yorzoi on both. DREAM-RNN, the supervised
in-distribution baseline, has now run and scores ~0.97 — the in-distribution
ceiling the zero-shot models reach for, reported separately (not a zero-shot
comparison).

## Shalem MPRA (terminator)

Designed 3′-end / terminator variants, marginalized logSED scoring.
([spec](benchmarks/shalem_mpra_terminator.md))

| metric | Shorkie | Yorzoi |
| --- | --- | --- |
| Pearson r | 0.6430 | **0.7075** |
| Spearman ρ | 0.6511 | **0.7066** |

**Winner: Yorzoi** — leads on both.

## Chen synonymous MPRA

Synonymous-codon effect on mRNA abundance, three libraries. Includes the CAI and
CodonTransformer sequence-feature baselines alongside the two S2F models.
([spec](benchmarks/chen_synonymous.md))

**gfp_r1** (two replicates)

| metric | CAI | CodonTransformer | Shorkie | Yorzoi |
| --- | --- | --- | --- | --- |
| Pearson r (rep1) | 0.3141 | 0.2091 | **0.3863** | 0.3067 |
| Pearson r (rep2) | **0.2655** | 0.1498 | 0.2169 | 0.2481 |
| Spearman ρ (rep1) | 0.3275 | 0.2971 | **0.4749** | 0.3759 |
| Spearman ρ (rep2) | 0.3055 | 0.2731 | 0.2979 | **0.3424** |

**gfp_r2** (two replicates)

| metric | CAI | CodonTransformer | Shorkie | Yorzoi |
| --- | --- | --- | --- | --- |
| Pearson r (rep1) | 0.3219 | 0.5560 | **0.6008** | 0.5695 |
| Pearson r (rep2) | 0.2469 | **0.5932** | 0.5455 | 0.5458 |
| Spearman ρ (rep1) | 0.3158 | 0.5735 | **0.6220** | 0.5891 |
| Spearman ρ (rep2) | 0.2385 | **0.6059** | 0.5564 | 0.5566 |

**tdh3** (single column)

| metric | CAI | CodonTransformer | Shorkie | Yorzoi |
| --- | --- | --- | --- | --- |
| Pearson r | **0.6752** | 0.3609 | 0.5270 | 0.6061 |
| Spearman ρ | **0.3887** | 0.1876 | 0.3024 | 0.2305 |

**Winner: Mixed.** Shorkie leads the two GFP libraries (CodonTransformer ties it
on gfp_r2 rep2); the CAI baseline leads TDH3 outright.

## Wu RFP insertions

Fixed RFP cassette moved across ORF-deletion loci — a pure genomic *position*
effect. ([spec](benchmarks/wu_rfpins.md))

| metric | Shorkie | Yorzoi |
| --- | --- | --- |
| Pearson r | -0.0387 | 0.0124 |
| Spearman ρ | -0.0290 | 0.0164 |

**Winner: Neither** — both models sit at ≈ 0 (informative negative; position
effect not captured). Yorzoi is marginally positive, Shorkie marginally negative.

## Hong IGR insertions

Fixed reporter across intergenic loci — the same position-effect probe as Wu, at
preserved intergenic regions. ([spec](benchmarks/hong_igr.md))

| metric | Shorkie | Yorzoi |
| --- | --- | --- |
| Spearman ρ (held-out test set, n=52) | -0.1475 | 0.0631 |
| Spearman ρ (secondary split, n=98) | -0.2687 | -0.0728 |

**Winner: Neither** — both ≈ 0 / negative (informative negative).

## Gui GFP RNA

GFP RNA abundance across 205 source-resolved heterozygous gene replacements.
All four models score the same genotypes and complete 717 bp GFP CDS, using
identical labels, constructs and reference input files.
([spec and full results](benchmarks/gui_gfp_rna.md))

| Model | Spearman ρ | 95% bootstrap CI |
| --- | ---: | ---: |
| Shorkie, eight folds + RC | 0.060 | −0.077 to 0.182 |
| Shorkie random init, eight folds + RC | 0.115 | −0.031 to 0.267 |
| Yorzoi, all Illumina + RC | 0.096 | −0.045 to 0.233 |
| ExoShorkie, six-student ensemble | −0.110 | −0.247 to 0.032 |

**Winner: None established.** All intervals include zero. Shorkie trained from
random initialization uses the published trained weights and the same 1,014
RNA-seq tracks as pretrained Shorkie. ExoShorkie averages the six students'
decoded, nonnegative CPM coverage on the reporter strand.

Shorkie and Yorzoi retain their 2026-11-11 runs at commit `b356a5f` on RTX A6000
GPU 1. The other two ran on 2026-11-12 at commit `1e91d39` on RTX A6000 GPU 0,
using `configs/default.yaml`. Runtime outputs are under
`results/default/<model>__gui_gfp_rna/`, with the original run metadata preserved.
See the [four-model comparison](../img/results/gui_gfp_rna/comparison.png) and
the task page for secondary correlations, IMP4 sensitivity and individual plots.
The historical 11-task random-init appendix below predates this Gui run.

## Brooks SCRaMBLE

SCRaMBLE genome rearrangement; per-construct log-fold-change of CDS coverage
against a parental control ([spec](benchmarks/brooks_scramble.md)). All four
models were rerun on 2026-11-12 with task version `v2`, which uses each
sequence's own CDS coordinates and strand. These scores supersede `v1`.

The shared cohort remains 327 samples. Runs support different subsets, so
metrics are reported separately and never averaged across the three runs.

| Metric | Run (n) | Shorkie | Yorzoi | ExoShorkie | Shorkie random-init | LOO ceiling |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| LFC direction balanced accuracy | JS94_r0 (205) | 0.513 | **0.593** | 0.512 | 0.534 | 0.761 |
|  | JS94_r1 (80) | 0.625 | **0.738** | 0.450 | 0.512 | 0.963 |
|  | JS94_r2 (291) | 0.548 | **0.602** | 0.553 | 0.559 | 0.694 |
| LFC Spearman ρ | JS94_r0 (205) | -0.083 | **0.461** | 0.004 | 0.085 | 0.779 |
|  | JS94_r1 (80) | -0.171 | **0.236** | 0.037 | 0.063 | 0.955 |
|  | JS94_r2 (291) | 0.153 | **0.337** | 0.169 | 0.246 | 0.589 |
| LFC Pearson r | JS94_r0 (205) | -0.123 | **0.391** | -0.058 | 0.026 | 0.789 |
|  | JS94_r1 (80) | -0.104 | **0.198** | -0.113 | -0.087 | 0.954 |
|  | JS94_r2 (291) | 0.201 | 0.384 | 0.228 | **0.420** | 0.671 |

Yorzoi leads on direction accuracy and Spearman correlation in every run;
Random-init Shorkie leads on Run 3 Pearson correlation. Correlations remain
below the leave-one-out ceilings, computed on 198 / 80 / 169 samples.
Run 2 has only measured decreases, so its direction score is the recall for
that observed class. Shape diagnostics remain outside the headline comparison
because the models use different output windows.

Inference used `configs/brooks_native_v2.yaml` at commit `21cc528` on RTX A6000
GPU 0. See the [audited metrics and provenance](benchmarks/brooks_native_v2_results.json)
and [per-sample predictions](benchmarks/brooks_native_v2_predictions.csv).
Full outputs are in `results/brooks_native_v2/`.

## Cuperus 5′-UTR

Random 50 bp 5′-UTRs (read-depth stratified) plus native yeast 5′-UTR fragments,
in the HIS3 reporter. ([spec](benchmarks/cuperus_mpra_5utr.md))

| metric | Shorkie | Yorzoi |
| --- | --- | --- |
| random Spearman ρ | **0.2518** | 0.1166 |
| random clean ρ (top read-depth bucket) | **0.2803** | 0.1142 |
| random partial-correlation ρ (beyond Kozak) | **0.2701** | 0.0924 |
| native Spearman ρ | **0.2012** | 0.1639 |
| native partial-correlation ρ (beyond Kozak) | **0.2611** | 0.0005 |

**Winner: Shorkie** — leads every cut, and retains signal in the partial
correlation (after regressing out hand-crafted Kozak features) where Yorzoi
drops to ≈ 0.

## Lee YTK promoters

Nineteen promoters driving mRuby2 or Venus from the same integrated construct
context. The main question is whether the models recover the measured max/min
range, not only promoter rank. ([spec](benchmarks/ytk_promoter.md))

Zero-shot, 2026-08-25 Modal A10 run. Both models scored all 38 constructs.

| consensus metric | Shorkie | Yorzoi |
| --- | ---: | ---: |
| predicted max/min (observed 162.079×) | **2.315×** | 1.739× |
| dynamic-range recovery | **0.1650** | 0.1088 |
| range fidelity | **0.1650** | 0.1088 |
| Spearman ρ | **0.6526** | 0.5702 |
| log10 Pearson *r* | **0.6660** | 0.5358 |

**Winner: Shorkie**, though both predictions are severely compressed. Shorkie
recovers some ordering but only 16.5% of the observed log10 span; Yorzoi recovers
10.9%.

## Meneu foreign DNA

Whole bacterial chromosomes integrated into yeast; tile each contig and predict
RNA-seq coverage zero-shot (far-OOD). Scored per contig over 5 kb windows.
([spec](benchmarks/meneu_foreign_dna.md))

**_M. mycoides_** (Mmmyco)

| metric | Shorkie | Yorzoi |
| --- | --- | --- |
| shape Pearson | 0.2638 | **0.3843** |
| shape JS ↓ | 0.2844 | **0.2382** |
| magnitude fold-change error (mean) | 0.7027 | 0.8066 |

**_M. pneumoniae_** (Mpneumo)

| metric | Shorkie | Yorzoi |
| --- | --- | --- |
| shape Pearson | 0.0912 | **0.1103** |
| shape JS ↓ | 0.0982 | **0.0826** |
| magnitude fold-change error (mean) | -0.0702 | 0.0464 |

**Winner: Yorzoi** — higher within-window shape correlation and lower JS
divergence on both chromosomes.

<!-- shorkie-random-init:start -->
## Shorkie trained from random initialization

Completed 11/11 full tasks. Updated 2026-11-07T00:10:24 UTC.

Eight published supervised-from-scratch folds, with reverse-complement averaging.
Rafi retains its existing bfloat16/compiled adapter default; other tasks use
float32. Chen uses four variants per batch, expanded over all host contexts.

Pretrained columns use available saved runs; missing baselines are left blank.
The earlier tables retain their historical results. Full summaries, sample counts,
and run metadata remain with the local benchmark outputs.

| Task | Metric(s) | Pretrained Shorkie | Random-init Shorkie | Eval time |
| --- | --- | --- | --- | --- |
| Caudal | AUROC / AUPRC | 0.5673 / 0.5793 | 0.5377 / 0.5381 | 26.0 min |
| Kita | AUROC / AUPRC | 0.6668 / 0.6399 | 0.6614 / 0.6181 | 8.5 min |
| Rafi | Pearson r / Spearman ρ | — | 0.5684 / 0.5808 | 239.9 min |
| Shalem | Pearson r / Spearman ρ | — | 0.6249 / 0.6415 | 133.4 min |
| Chen | mean r / mean ρ | — | 0.4708 / 0.4369 | 63.3 min |
| Wu | Pearson r / Spearman ρ | -0.0387 / -0.0290 | 0.0582 / 0.1040 | 1.5 min |
| Hong | held-out ρ / fitted-track ρ | -0.1475 / 0.1826 | 0.3520 / 0.1621 | 0.5 min |
| Brooks (v2) | shared-cohort balanced accuracy, r0 / r1 / r2 | 0.5128 / 0.6250 / 0.5476 | 0.5338 / 0.5125 / 0.5588 | 2.6 min |
| Cuperus | random clean / partial ρ; native / partial ρ | — | 0.1565 / 0.1510 / 0.2548 / 0.2420 | 268.3 min |
| YTK | range recovery / Spearman ρ | 0.1650 / 0.6526 | 0.0626 / 0.8035 | 0.0 min |
| Meneu | shape r / JS↓: M. pneumoniae; M. mycoides | 0.0912 / 0.0982 / 0.2638 / 0.2844 | 0.1405 / 0.0851 / 0.0616 / 0.3867 | 0.2 min |

The Brooks row was updated after the 2026-11-12 v2 scoring correction; other
rows retain their historical runs. Brooks uses the same 698 shared sample IDs.
Its three replicate metrics are reported separately. The raw-result audit checks
all reconstructable metrics; Brooks's secondary shape diagnostics are available
only in its saved summary because coverage arrays are not persisted.

<!-- shorkie-random-init:end -->
