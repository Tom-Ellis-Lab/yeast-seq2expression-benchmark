# Meneu et al. — foreign-DNA RNA-seq coverage-track prediction

![image](../../img/meneu_banner.svg)

Figure partially from [Meneu et al.](https://doi.org/10.1126/science.adm9466)

## At a glance

| | |
| --- | --- |
| **Task** | Tile each integrated foreign-chromosome reference end-to-end, predict a per-base RNA-seq coverage track zero-shot, and correlate against the measured RNA-seq per chromosome. A whole-distribution OOD test, not a local cis-perturbation probe. |
| **Source** | Meneu *et al.* 2025, *Sequence-dependent activity and compartmentalization of foreign DNA in a eukaryotic nucleus*, Science 387(6734):eadm9466. DOI: [10.1126/science.adm9466](https://doi.org/10.1126/science.adm9466). PDFs in `papers/science.adm9466.pdf` (+ `_sm.pdf`). |
| **Data** | Genome FASTAs + per-base normalized RNA-seq coverage (`.npz`, fwd/rev per contig) from the **ExoShorkie figshare** ([10.6084/m9.figshare.31075375](https://doi.org/10.6084/m9.figshare.31075375)) — no bigwig / pyBigWig; the coverage array is keyed by contig and matches the FASTA length. Provenance: GEO **GSE217022**, Zenodo **14024599** / **7198985**. |
| **Assay** | Stranded directional mRNA RNA-seq (rRNA-depleted, PE150, 3 biological replicates), CPM-normalized per-base coverage. On bacterial DNA the signal follows bacterial gene orientation. |
| **Expression label** | Per-base **CPM coverage**, unstranded (forward + reverse summed). Scale-free — Pearson/Spearman only need monotone correspondence. |
| **Reference assembly** | The scored sequences are the two foreign contigs: *M. pneumoniae* M129 (`Mpneumo`, 817,946 bp) and *M. mycoides* PG1 (`Mmmyco`, 1,222,199 bp), from the ExoShorkie figshare FASTAs. Host nuclear context is *S. cerevisiae* R64-1-1 (suite default), but only the foreign contig is tiled and scored. |
| **Eval set** | **Mpneumo** (*M. pneumoniae* M129, ~818 kb, 40% GC — yeast-like, transcribed) and **Mmmyco** (*M. mycoides* PG1, ~1,222 kb, 24% GC — AT-rich, near-silent). Scored **per chromosome, never pooled.** |
| **Primary metric** | Per chromosome, over non-overlapping 5 kb windows at 1 bp resolution: median raw Pearson + median JS divergence (within-region shape) + per-window fold-change error after depth-normalization (across-region magnitude). See *Metrics*. |
| **Adapter protocol** | `TiledCoverageTrackPredictor` — new protocol, identical surface to `CoverageTrackPredictor`, so the registry dispatches Meneu to its own adapters without colliding with Brooks. Reused via thin adapter subclasses. |

## Contents

- [At a glance](#at-a-glance)
- [Why this benchmark exists](#why-this-benchmark-exists)
- [The sequences scored](#the-sequences-scored)
- [Data](#data)
- [Metrics](#metrics)
- [Results](#results)
- [Sign convention (verify empirically)](#sign-convention-verify-empirically)
- [Model contract](#model-contract)
- [Files (target layout)](#files-target-layout)
- [Open questions / future work](#open-questions--future-work)

## Why this benchmark exists

Every other benchmark in this suite perturbs *local* cis-sequence (a 5′-UTR, a
terminator, an intergenic insertion, a structural rearrangement). Meneu provides
the opposite extreme: ~2 Mb of genuinely **foreign bacterial sequence** placed in
a yeast nucleus and profiled by RNA-seq. Running a yeast-trained model over it
zero-shot is a far-out-of-distribution stress test of sequence composition — the
natural home for the ROADMAP "native-genome track prediction" item, generalized
from held-out yeast regions to non-native DNA the model provably never saw.

It is mechanically the **Brooks SCRaMBLE coverage benchmark with the LFC
machinery stripped out**: same `CoverageTrackPredictor`-style protocol
(sequence-in / per-base coverage-out), same model adapters. The genuinely new
code is (a) tiling a megabase contig end-to-end (Brooks predicts isolated gene
windows), to the model's receptive field at run time from a single
window-agnostic per-contig artifact, and (b) a per-chromosome Pearson/Spearman
over the stitched track instead of paired LFC.

### The reference ladder (orientation, not competition)

The right references come from **ExoShorkie** (Mandl & Orenstein 2026,
`papers/2026.01.25.701486v1.full.pdf`), which predicts RNA-seq coverage of
exogenous genomes — the same task and assay, on **the same two chromosomes**. (The
Meneu paper itself has no RNA-seq model, so it offers no comparison here.) Read
every result against these as *rough* orientation — different processing, metric,
and training regime, not exact targets:

| Reference | Regime | ~Mpneumo | ~Mmmyco |
| --- | --- | --- | --- |
| **ExoShorkie** | Shorkie *transfer-learned* on exogenous RNA-seq | ~0.60–0.68 | ~0.62–0.76 |
| **NatShorkie** | native-trained Shorkie, **zero-shot** (≈ our Shorkie) | ~0.46 | ~0.58 |
| **ExoYorzoi** | Yorzoi, zero-shot on bacterial (≈ our Yorzoi) | ~0.38 | ~0.58 |

(Median per-window Spearman, read from ExoShorkie Fig. 3; orientation only.)
The takeaway: the task is *learnable* with transfer learning (ExoShorkie), and
native-trained models already capture a meaningful zero-shot fraction
(NatShorkie / ExoYorzoi). Our benchmark measures **that zero-shot fraction** as a
portable, model-agnostic eval.

### GC gradient — informative, but Mmmyco is not the worst case

Order results by GC distance from yeast (Mmmyco 24% → Mpneumo 40% ≈ yeast 38%).
But note ExoShorkie scores **Mmmyco higher than Mpneumo** despite its 24% GC and
near-silent transcription — sparse signal can still rank well. So "AT-rich =
hardest" is *not* assumed; the GC axis is reported, not editorialized.

### Caveat: the cis model can't see trans / chromatin context

The measured RNA-seq reflects the foreign chromosome's full nuclear context
(3D compartment, chromatin state, decay). A cis sequence-to-expression model sees
only local sequence, so there is an intrinsic ceiling below 1.0. We frame results
as the cis-predictable, zero-shot fraction and read them against the references
above, never against a perfect correlation.

## The sequences scored

The integrated reference per strain is a chimeric contig:
`telomere — [bacterial arm] — CEN6/ARS-HIS3 cassette — [bacterial arm] — telomere`.
The bacterial chromosome was cloned circular with a yeast CEN6/ARS-HIS3 selection
cassette, then CRISPR-linearized and capped with yeast telomere seeds — so the
deposited contig (~818 kb Mpneumo ≈ native M129 + ~1.5 kb cassette; ~1,222 kb
Mmmyco ≈ native PG1 + ~10.5 kb cassette) contains a small **internal** non-bacterial
insert plus telomere ends.

- **Tile the whole deposited contig.** Step a `seq_len`-wide window by its
  *predicted-region* length (Shorkie 14,336 bp / Yorzoi 3,000 bp) so the central
  regions tile `[0, L)` contiguously; N-pad contig ends (the one-hot encoder maps
  `N` → an all-zero column, so padding is safe).
- **No masking (v1).** ExoShorkie deposits the exogenous contig as one sequence and
  scored it unmasked; the internal CEN6/ARS-HIS3 cassette + telomere ends are
  <0.3% of the contig and unannotated in the figshare FASTA, so v1 scores the whole
  contig (ExoShorkie-consistent). Masking is a v2 refinement if coordinates appear.
- **Unstranded.** v1 scores forward + reverse summed coverage. This is the only
  target physically defined for both models on a tiled contig: Shorkie is
  strand-blind, and every window straddles forward- and reverse-oriented bacterial
  genes, so there is no single per-tile strand. Per-strand is a Yorzoi-only v2.

## Data

### Provenance and inventory
- **ExoShorkie figshare ([10.6084/m9.figshare.31075375](https://doi.org/10.6084/m9.figshare.31075375))** —
  the **primary build-time source**. Per genome: a FASTA (`Mpneumo.fa` = one contig
  `Mpneumo`, 817,946 bp; `Mmmyco.fa` = `Mmmyco`, 1,222,199 bp) plus per-base
  normalized RNA-seq coverage as **`.npz` dicts keyed by contig**
  (`<genome>_fwd_norm.npz` / `_rev_norm.npz`, float32; each file holds all 16 yeast
  chroms + Mito + the exogenous contig). The exogenous coverage is
  `npz['<contig>']`, whose length equals the FASTA length → **no pyBigWig, and no
  coordinate-matching hazard** (the array key *is* the contig). Unstranded = fwd + rev.
- **GEO `GSE217022`** / **Zenodo `14024599` / `7198985`** — provenance, and the
  source for a future *per-replicate* reproducibility ceiling (the figshare tracks
  are replicate-merged). Not needed for v1.

### Measured RNA-seq label
Per-base normalized coverage, unstranded (forward + reverse summed). Scale cancels
in the shape metrics; the magnitude metric depth-normalizes the prediction to the
true total.

### Per-chromosome eval units
Each chromosome (Mpneumo, Mmmyco) is scored independently. **Never pooled.**

### Masking / what's excluded
**None in v1** — the whole deposited contig is scored. The ~1.5 kb internal
CEN6/ARS-HIS3 cassette + telomere ends are <0.3% of the contig and unannotated in
the figshare FASTA, and ExoShorkie scored the contig unmasked; N-padded contig ends
fall outside the central predicted regions anyway. Masking is a v2 refinement.

## Metrics

### Per-chromosome correlation, 1 bp resolution
After each adapter unbins its prediction to per-base, stitch the tiled predicted
regions into one whole-contig per-base vector, drop masked / N-padded positions,
and align to the measured per-base coverage. Each model is tiled **independently
at its own window / stride** (Shorkie 16,384 / 14,336 bp, Yorzoi 4,992 / 3,000 bp,
where stride = the predicted-region length) — there is no shared window, so neither
model is starved of context nor fed mid-contig Ns (only the two contig ends N-pad,
and those are excluded). The comparison unit is the
**whole-chromosome** correlation against the same per-base truth; the
receptive-field difference (Shorkie integrates ~14 kb of context per central
region, Yorzoi ~3 kb) is an intrinsic model property, reported as such, not an
artifact to equalize. 1 bp resolution needs no common-bin choice and is uniform
across every model (each unbins to per-base).

Three metrics per chromosome (shape co-variation, shape mass-placement, magnitude):

- **Shape — median per-window Pearson.** Tile the stitched per-base tracks into
  fixed **non-overlapping 5 kb windows** (the same windows for every model,
  independent of each model's prediction tiling), compute **Pearson on the raw
  per-base coverage** within each window, and report the **median across windows**.
  Equal-weighting windows stops a few high-expression loci from dominating and
  forces local profile reconstruction. Raw Pearson is scale-invariant, so no
  depth-normalization is needed here (the magnitude metric below *does*
  depth-normalize). **Zero-variance floor:** skip windows whose *true* coverage is
  flat (variance ≤ `FLOOR_EPS` = 1e-9 — the fully-silent windows, where Pearson is
  undefined; windows with real but low signal are kept), and report the
  scored-window count (`n_windows_kept` / `n_windows_total`). A flat or all-zero
  *prediction* also makes Pearson NaN and is excluded; the contributing count is
  reported as **`n_windows_pearson`** (`n_windows_kept − n_windows_pearson` =
  NaN-Pearson drops, 0 for both models on both chromosomes in the first runs).
- **Shape, mass-placement — Jensen–Shannon divergence of the normalized profiles.**
  Within each 5 kb window, normalize true and predicted coverage to sum 1 and
  compute the JS divergence (bits); report the **median across surviving windows**
  (same windows + floor as the Pearson). JS is symmetric, bounded `[0, 1]` bits,
  and finite without smoothing, so it averages and compares cleanly across windows,
  chromosomes, and the future reproducibility ceiling — which is why **JS, not KL**,
  is used (KL is asymmetric, unbounded, and sparsity-dependent; this matches the
  Brooks Tier-2 shape convention and reuses `brooks.py:_js_divergence`). Pearson and
  JS are complementary: Pearson catches peak co-location, JS catches where the
  transcriptional mass sits.
- **Magnitude — relative log (fold-change) error**, adapting the Yorzoi paper's
  fold-change error (Schneider et al. 2025, `papers/2025.09.20.677345v1.full.pdf`,
  "Pearson Correlation and Fold-Change Error") to a per-window form. First
  **depth-normalize genome-wide**: scale the predicted track by
  `sum(true)/sum(pred)` over the whole contig so the totals match — this removes
  the model's arbitrary global scale, so the error measures *regional allocation*
  rather than global mis-calibration. Then per 5 kb window take the fold-change
  error of the **window totals**, `log2((Σ_window pred + 1) / (Σ_window true + 1))`
  (their `log2(Ŷ/Y)`, +1 for finiteness), and summarize across **all** windows as
  mean ± spread. Unlike the two shape metrics, magnitude does **not** apply the
  low-signal floor: a window that is silent in truth but where the model predicts
  coverage is a genuine mis-allocation that must register, and correctly predicting
  silence (`Σpred ≈ Σtrue ≈ 0 → FC ≈ 0`) is credited. Because every model is scored
  on the identical window set, the shared "easy zeros" don't bias the cross-model
  comparison — and it keeps the rule simple (one window set for magnitude, no floor
  to reason about). It asks: did the model allocate the right *amount* of signal to
  each region? — the level dimension the scale-invariant shape metrics are blind
  to. Together: *right shape within regions?* (Pearson + JS) vs *right level across
  regions?* (fold-change error). Read the **spread** as the primary magnitude
  signal: because per-window errors are averaged in log space while the depth-norm
  matches *linear* totals, a perfectly-calibrated model with dispersed regional
  errors gets a slightly *negative* `mag_fc_mean` (Jensen's inequality), so a nonzero
  mean is not on its own evidence of global over- or under-prediction.

v1 fixes **5 kb windows + raw Pearson** (decided). No per-gene metric — there are
no meaningful genes on the artificial chromosome. Residual caveat: a window holding
an on- and an off-region scores high partly for the on/off contrast rather than
quantitative level; a finer shape metric is a separate project.

### Reference baseline
The ExoShorkie / NatShorkie / ExoYorzoi numbers above, cited as rough orientation,
not as targets to beat.

## Results

*2026-07-31 Modal run (v1, unified `meneu_foreign_dna` task, `.npz`-sidecar artifact). Per-window (5 kb) over each foreign chromosome. (The prior 2026-06-02 build reproduces these same numbers to the printed precision.)*

| metric | contig | Shorkie | Yorzoi |
| --- | --- | ---: | ---: |
| shape Pearson (median) | *M. pneumoniae* | 0.091 | 0.110 |
| | *M. mycoides* | 0.264 | 0.384 |
| shape JS divergence | *M. pneumoniae* | 0.098 | 0.083 |
| | *M. mycoides* | 0.284 | 0.238 |
| fold-change error (mean ± sd) | *M. pneumoniae* | −0.07 ± 1.19 | 0.05 ± 1.05 |
| | *M. mycoides* | 0.70 ± 1.85 | 0.81 ± 1.72 |

![Yorzoi predicted vs measured coverage over the M. mycoides contig (its best case) with the per-window-r histogram.](../../img/results/meneu_foreign_dna/yorzoi_Mmmyco.png)
![Shorkie over M. pneumoniae (the hardest case).](../../img/results/meneu_foreign_dna/shorkie_Mpneumo.png)

- Both models sit well below the ExoShorkie transfer-learning ladder (≈ 0.46–0.76 shape Pearson) — expected for far-OOD bacterial sequence neither model trained on. Yorzoi edges Shorkie on shape for both contigs.
- Which contig looks easier depends on the metric. *M. mycoides* scores about three times the shape Pearson of *M. pneumoniae* for both models, even though it is the contig further from yeast in GC (24% against yeast's 38%; *M. pneumoniae* is 40%). JS divergence and the fold-change spread rank them the other way, with *M. pneumoniae* ahead on both. The likely cause is transcription, not composition: *M. pneumoniae* is covered almost everywhere while *M. mycoides* is near-silent and spiky, and per-window Pearson rewards the on/off contrast a spiky profile offers. ExoShorkie reports the same Pearson ordering, so this is a property of the data rather than of our pipeline.
- Read the fold-change spread (sd) as the magnitude signal — the means are confounded by over-prediction vs dispersion (see the *Metrics* caveat).

Why the numbers look the way they do: [`model_failures.md`](model_failures.md). Artifacts: `results/meneu/{shorkie,yorzoi}__meneu_foreign_dna/summary.json` and `results/meneu/compare/summary.md`.

## Sign convention (verify empirically)
Higher predicted coverage ↔ higher measured coverage → positive correlation.
Verify on the first run: a far-OOD model can emit near-flat coverage on low-GC
sequence, which simply yields a low / undefined correlation — reported honestly,
no special handling.

## Model contract

The shipped coverage-track protocol surface (`protocols.py`). The Meneu protocol
is a distinct type with the **identical** surface (so registry dispatch does not
collide with Brooks' `CoverageTrackPredictor`):

```python
@runtime_checkable
class TiledCoverageTrackPredictor(Protocol):
    """Predict an RNA-seq-like coverage profile for a batch of constructs,
    for whole-contig tiled-coverage benchmarks (Meneu foreign DNA). Same
    surface as CoverageTrackPredictor; a distinct type purely so the
    registry can dispatch tiled-coverage tasks to their own adapters."""

    seq_len: int
    crop_bp_each_side: int
    batch_size: int
    varies_by_strain: bool

    def predict_coverage_batch(
        self,
        seqs: Sequence[str],
        strands: Sequence[str],
        strains: Sequence[str | None] | None = None,
    ) -> np.ndarray: ...
```

Returns `(B, seq_len - 2 * crop_bp_each_side)` in **raw per-base predicted-count
units** (the adapter inverts any training transform and unbins to per-base). The
benchmark reads `seq_len`/`crop_bp_each_side` to map the tile's true coverage to
the predicted central region. (The Brooks spec's older `predict_coverage(seq,
strand)` block is stale — this batched API is the live one.)

## Files (target layout)

### Raw upstream (build-time only)
- ExoShorkie figshare genomes + processed RNA-seq, cached under
  `data/tasks/meneu_foreign_dna/_cache/` (idempotent download).
- GEO `GSE217022` per-replicate bigwigs — only if the reproducibility ceiling is
  built (deferred; see *Open questions*).

### Processed distribution (the run-time dependency)
- `data/tasks/meneu_foreign_dna/meneu_cov_<contig>.npz` — **built, gitignored.**
  One **window-agnostic** sidecar per contig holding the full contig `seq`
  (uint8/ASCII) plus per-base `fwd` / `rev` coverage (float32). The benchmark
  reads `seq`, tiles it to the adapter's receptive field **at run time**
  (`tile_contig`: stride = predicted-region length, N-padded ends), stitches each
  tile's central prediction into a per-base profile, and scores it against
  `fwd + rev`. One artifact serves every model — there is no per-window TSV any
  more, so this is a single registry task (`meneu_foreign_dna`) regardless of
  receptive field. At run time it depends on these built files alone — no
  figshare, no GEO. (The *Results* numbers come from this unified task; an earlier
  per-window-TSV build with a separate `meneu_foreign_dna_shorkie` task reproduced
  them exactly.)

### Track subsets / RC averaging
- **Shorkie:** the **384-track T0 subset** (`SHORKIE_T0_RNA_SEQ_TRACK_IDS`), for
  consistency with every other Shorkie benchmark (Brooks/Hong/Chen/Shalem/Wu/MPRA).
  These are a steady-state subset of the higher-quality `RNA-Seq` group (paper
  Pearson ~0.776) — not the worse `1000-RNA-Seq` 1000-strains collection (~0.629) —
  and steady-state log-phase matches Meneu's assay. No subset override needed (it
  is already the `ShorkieBrooksPredictor` default). `varies_by_strain=False`.
- **Yorzoi:** there is no generic-RNA-seq group in `track_annotation.json` (the
  81 plus-tracks are ~70 Brooks Nanopore direct-RNA, ~10 exogenous-human Illumina,
  1 SRA Illumina). The headline config is the **10 exogenous-human Illumina
  tracks** (the set ExoYorzoi used). Record metrics under each group
  (Illumina-10 / Nanopore-63 / SRA-1), report the table, and document the best in
  prose. Set `varies_by_strain=False` (bacterial strains have no matched tracks).
- Both adapters average forward + RC internally; the Yorzoi adapter additionally
  sums its + and − track axes so it returns one unstranded track per tile.

### Build script
- `scripts/meneu/build_meneu_distribution.py` — the only component that touches
  figshare: downloads each genome's FASTA + fwd/rev coverage `.npz` and writes one
  window-agnostic `meneu_cov_<contig>.npz` per genome (`seq` + `fwd` + `rev`). No
  tiling at build time — the benchmark tiles to the model's receptive field at run
  time. `uv run python scripts/meneu/build_meneu_distribution.py`.
  No pyBigWig (coverage is already per-base `.npz`); no masking (whole contig scored).

## Open questions / future work

1. **Investigate the gap vs ExoShorkie — triple-check every assumption.**
   Our zero-shot numbers (see *Results*) are well below ExoShorkie's reported
   NatShorkie / ExoYorzoi (~0.4–0.6). Likely mostly the metric, but verify in order:
   (a) **Metric** — reproduce ExoShorkie's *exact* metric (per-window **median
   Spearman** over **16 bp bins** of **14,336 bp** windows) on our stitched
   predictions and see how much of the gap that alone closes (Pearson→Spearman,
   1 bp→16 bp, 5 kb→14.3 kb).
   (b) **Alignment / scale** — confirm prediction & truth share the per-base
   coordinate frame and orientation at a known-expressed locus (positive Mmmyco r
   says broadly yes, but check directly; eyeball `results/meneu/*/<contig>.png`).
   (c) **Track subsets** — Shorkie T0 (384) vs the broader `RNA-Seq` group; the
   Yorzoi sweep (#2).
   (d) **NatShorkie's native strand-adaptation fine-tune** (ours is raw Shorkie) —
   quantify its effect.
   (e) **Unstranded fwd+rev summing** vs ExoShorkie's per-strand handling; the
   depth-norm and the +1 pseudocount. Reproduce one NatShorkie / ExoYorzoi figure
   with their pipeline before trusting our absolute values.
2. **Yorzoi track-group sweep — only `illumina_exo` run in v1.** Add Nanopore-63 and
   SRA-1 modes to `YorzoiMeneuPredictor._plus_axis_indices`, run all three (seconds
   each), and report the best (per #5).
3. **Reproducibility ceiling — v2 unless figshare makes it trivial.** The honest
   denominator for low OOD numbers (especially near-silent Mmmyco): rep↔rep
   test-retest Pearson/Spearman per chromosome. ExoShorkie appears to deposit
   *replicate-merged* tracks (Picard-merged per its methods), so this would need
   the 3 per-replicate bigwigs from `GSE217022` (`GSM6703670-675`). Confirm at
   build: if per-replicate tracks are already on figshare it is cheap enough for
   v1; otherwise defer to v2.
4. **No yeast in-distribution anchor.** chrXVI was considered as a "familiar
   yeast" reference and dropped — it is ~74% inside Yorzoi's training (not a clean
   control), and this benchmark is about foreign DNA. Out of scope; the held-out
   chrXVI carve-out is irrelevant.
5. **Yorzoi track-group selection rule.** Whether to pick the headline group purely
   descriptively (report all, name the best) or via a fixed selection set; chrXVI-
   anchored selection is off the table for v1 (no anchor). Default: descriptive.
6. **Per-base track metric — decided.** See *Metrics* for the full definition
   (median per-window raw Pearson + JS divergence for shape, per-window fold-change
   error for magnitude) and the rejected alternatives. Impl details to pin on real
   data: the floor ε and the fold-change-error ε.
7. **Per-strand correlation — Yorzoi-only v2.** Tests whether the model transcribes
   bacterial genes in the correct orientation; impossible for strand-blind Shorkie.
   The Meneu RNA-seq *is* stranded (directional library, fwd/rev bigwigs), so the
   ground truth supports it — v1 sums strands only because Shorkie cannot.
8. **Mosaic / translocation strains — v2.** The XVIf* strains *do* have RNA-seq
   coverage bigwigs (`GSM8640818-825`), so they extend the *same* coverage pipeline
   (not a separate DESeq2 task). Deferred for perturbation confounds (+thiolutin /
   upf1 / rrp6) and per-strain coordinate-matching cost. YACs (P. falciparum,
   Phytoplasma) have no RNA-seq and cannot be scored here.
9. **Registry name.** `TiledCoverageTrackPredictor` vs another task-named protocol;
   identical surface to `CoverageTrackPredictor`, distinct type for dispatch only.
