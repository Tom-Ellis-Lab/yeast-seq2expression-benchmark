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
| [Brooks SCRaMBLE](#brooks-scramble) | Shorkie, Yorzoi | LFC direction balanced acc., per JS94 run | **Yorzoi** (all 3 runs) |
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

## Brooks SCRaMBLE

SCRaMBLE genome rearrangement; per-construct log-fold-change of CDS coverage vs.
an unscrambled parental control, against a leave-one-out reproducibility ceiling.
([spec](benchmarks/brooks_scramble.md))

Shared cohort (n = 327 — the intersection both models' receptive fields support,
the apples-to-apples view). Metrics are reported **per JS94 parental run** and
never averaged across the three: each run scores a different set of genes (a
gene needs enough reads *in that run*, and the cutoff falls per gene), so the
cohorts differ in size and composition and each has its own ceiling. Compare
models within a row, never across rows.

| metric | run (n) | Shorkie | Yorzoi | LOO ceiling |
| --- | --- | --- | --- | --- |
| LFC direction balanced acc. | JS94_r0 (205) | 0.514 | **0.624** | 0.761 |
| | JS94_r1 (80) | 0.613 | **0.650** | 0.963 |
| | JS94_r2 (291) | 0.534 | **0.630** | 0.694 |
| LFC Spearman ρ | JS94_r0 (205) | -0.065 | **0.439** | 0.779 |
| | JS94_r1 (80) | -0.157 | **0.275** | 0.955 |
| | JS94_r2 (291) | 0.109 | **0.224** | 0.589 |
| LFC Pearson r | JS94_r0 (205) | -0.087 | **0.297** | 0.789 |
| | JS94_r1 (80) | 0.004 | **0.212** | 0.954 |
| | JS94_r2 (291) | 0.023 | **0.153** | 0.671 |

**Winner: Yorzoi** — leads Shorkie on all three metrics in all three parental
runs, nine out of nine. Shorkie sits at chance throughout. Both stay well short
of the reproducibility ceiling in every run. Full-set numbers (Shorkie n=528,
Yorzoi n=327) are kept as secondary in `results/brooks/compare/`. The
coverage-**shape** metric (Yorzoi 0.83 / Shorkie 0.42 Pearson) is **deferred to
v2** — cross-model shape numbers aren't comparable until a common readout
window is fixed (see the [spec](benchmarks/brooks_scramble.md)).

> 2026-07-31 Modal run, recomputed per replicate 2026-09-02 from the saved
> per-replicate arrays — the models were not re-run. The earlier single-number
> headline averaged these three correlations, which weighted the 80-sample run
> equally with the 291-sample one.

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
