from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class Variant:
    chrom: str
    pos: int  # 1-based, R64-1-1
    ref: str
    alt: str
    gene_id: str  # Ensembl ID


@runtime_checkable
class VariantEffectScorer(Protocol):
    def score_variants(self, variants: Sequence[Variant]) -> np.ndarray: ...


@runtime_checkable
class CassetteExpressionPredictor(Protocol):
    """Predict expression of a reporter in one constant cassette that is
    integrated at varying genomic loci by **CDS replacement** (Wu et al.
    position-effect task: the cassette replaces a deleted ORF span).
    ``loci`` is a sequence of resolved ``WuLocus`` records; returns one
    scalar per locus, aligned to input order."""

    def predict_expressions(self, loci: Sequence) -> np.ndarray: ...


@runtime_checkable
class IntegratedPromoterPanelPredictor(Protocol):
    """Predict RNA abundance for varying reporter constructs at one locus.

    ``constructs`` contains complete promoter-reporter cassettes and the exact
    reporter CDS span in each cassette. Implementations return one scalar per
    construct, aligned to input order. Track models must sum raw, untransformed,
    unbinned per-base coverage over those exact CDS bases.
    """

    def predict_reporter_expressions(self, constructs: Sequence) -> np.ndarray: ...


@runtime_checkable
class IGRInsertionExpressionPredictor(Protocol):
    """Predict expression of a reporter in one constant cassette that is
    integrated at varying genomic loci by **intergenic-region insertion**
    (Hong et al. position-effect task: the cassette inserts between two
    intact native genes at the gRNA-defined cut site, no deletion).
    ``loci`` is a sequence of resolved ``HongLocus`` records; returns
    one scalar per locus, aligned to input order.

    Kept as a separate protocol from ``CassetteExpressionPredictor`` so
    the registry can dispatch the two assays to different adapters per
    model — the cassettes, locus shapes, and window-anchor conventions
    differ even though the surface method signature is identical.

    **Optional extension for IntTrain-fitted IntProp ρ.** Adapters may
    *additionally* implement
    ``predict_diagnostic_readouts(loci) → dict[str, np.ndarray]`` to
    expose multiple candidate readouts (different track groups ×
    readout regions, with biological signs applied). The Hong benchmark
    selects the (track group × region) combination that maximises
    signed Spearman ρ on IntTrain and reports its ρ on IntProp as the
    IntTrain-fitted IntProp ρ metric. The returned dict must include a
    ``'primary'`` key holding the same array that
    ``predict_expressions`` returns. All other keys are candidates for
    IntTrain selection. The benchmark uses
    ``hasattr(adapter, 'predict_diagnostic_readouts')`` to detect
    support; adapters without it get only Primary reported."""

    def predict_expressions(self, loci: Sequence) -> np.ndarray: ...


@runtime_checkable
class SequenceExpressionScorer(Protocol):
    """Return one scalar per input sequence that should track its measured
    expression — *however the model computes it*. The Rafi/deBoer benchmark
    correlates these scalars against measured expression per stratum, so only
    monotone correspondence matters (Pearson/Spearman are scale-free).

    Two kinds of model implement this:

    - **Zero-shot foundation models** (Shorkie, Yorzoi) return the mean logSED
      of the insert marginalized across a committed list of native host-gene
      contexts (inserted upstream of the TSS) — they predict *effects in native
      context*, never reporter readouts.
    - **Supervised in-distribution baselines** (DREAM-RNN) return the directly
      predicted reporter expression of the insert in its own reporter context.

    They are scored on the same axis but fed their native substrate; any figure
    placing them together must say so (see ``docs/benchmarks/rafi_mpra_promoter.md``)."""

    def predict_expression_scores(self, seqs: Sequence[str]) -> np.ndarray: ...


@runtime_checkable
class CoverageTrackPredictor(Protocol):
    """Predict an RNA-seq-like coverage profile for a batch of constructs.
    Used by the Brooks SCRaMBLE benchmark (sequence-in / coverage-out,
    not variant-effect).

    **Output contract:** ``predict_coverage_batch`` returns a 2D numpy
    array of shape ``(B, seq_len - 2 * crop_bp_each_side)``, in **raw
    per-base predicted-count units** — adapters are responsible for
    inverting any model-specific training transform (e.g. Borzoi/Yorzoi's
    `x^0.75 + sqrt`-squash + 4 bp binning) and unbinning back to per-base
    before returning, so the benchmark can compute LFC / Pearson / JSD
    directly against raw per-base pileups in matching units.

    Adapters expose ``seq_len`` and ``crop_bp_each_side`` so the
    benchmark can slice the true per-base coverage to the predicted
    central region and map CDS coordinates correctly. Adapters also
    expose ``batch_size`` — the maximum batch the benchmark may pass
    in one call (the benchmark chunks larger sets internally).

    **Optional per-sample ``strains``.** For track-based models (e.g.
    Yorzoi), the benchmark may pass per-sample strain identifiers so
    the adapter routes each prediction to the matching experimental
    tracks (Brooks SCRaMBLE strain S → use S's Nanopore tracks;
    native construct → use JS94's deep-WT tracks). Adapters without
    per-condition track selection may ignore them.

    **``varies_by_strain``** (class-level attribute, default ``True``).
    Set to ``False`` for adapters whose predictions are identical for
    different strain values (e.g. Shorkie, which has no Brooks-
    specific output tracks). The Brooks benchmark uses this to skip
    redundant native predictions when computing per-replicate LFCs —
    if ``False``, the native is predicted once and broadcast across
    the JS94 replicate axis."""

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


@runtime_checkable
class TiledCoverageTrackPredictor(Protocol):
    """Predict an RNA-seq-like coverage profile for a batch of windows
    that **tile a whole contig** (Meneu foreign-DNA benchmark). The
    benchmark slides fixed-stride windows across each contig, calls
    ``predict_coverage_batch`` per window, and stitches each window's
    central ``seq_len - 2 * crop_bp_each_side`` prediction into one
    per-base contig-length profile.

    **Output contract:** ``predict_coverage_batch`` returns a 2D numpy
    array of shape ``(B, seq_len - 2 * crop_bp_each_side)``, in **raw
    per-base predicted-count units** and **unstranded** (fwd + rev) —
    adapters invert any model-specific training transform and sum the
    forward and reverse strands before returning so the benchmark can
    compare directly against the unstranded ``fwd + rev`` truth.

    Distinct from ``CoverageTrackPredictor`` (same method surface) purely
    so the registry dispatches whole-contig tiled-coverage tasks (Meneu)
    to their own adapters, separate from the isolated-construct Brooks
    SCRaMBLE adapters — mirroring the precedent on
    ``TerminatorMarginalizedExpressionPredictor`` (same signature as
    ``SequenceExpressionScorer`` but a different assay).

    ``varies_by_strain`` is part of the surface for parity with
    ``CoverageTrackPredictor``; Meneu adapters set it ``False`` (their
    track subset does not depend on the strain)."""

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


@runtime_checkable
class TerminatorMarginalizedExpressionPredictor(Protocol):
    """Predict the marginalized effect of each input sequence across native
    host-gene contexts, inserted **downstream** of the host-gene stop codon
    with a non-terminating filler (terminator-MPRA flavour, e.g. Shalem).
    Returns mean logSED across a committed list of host genes.

    Distinct from ``SequenceExpressionScorer`` because the
    insertion *site* and the surrounding scaffold are semantically
    different (promoter-region insertion vs terminator-region insertion),
    so the two protocols cleanly disambiguate in the registry."""

    def predict_terminator_marginalized(self, seqs: Sequence[str]) -> np.ndarray: ...


@runtime_checkable
class LocalCodingVariantPredictor(Protocol):
    """Predict scalar expression for synonymous (or local) coding-region
    variants of a single construct gene per library. Given a parallel pair
    of (library_id, 36 nt variant block) sequences, return one scalar per
    variant in adapter-defined units; the benchmark only requires monotone
    correspondence to the measured label for Pearson scoring.

    Used by the Chen et al. 2017 synonymous-mutation MPRA benchmark.
    See ``docs/benchmarks/chen_synonymous.md``.

    Contract:
    - ``library_ids[i]`` is one of the strings the adapter advertises via
      its registry config (e.g. ``"gfp_r1"``, ``"gfp_r2"``, ``"tdh3"``).
    - ``variant_seqs[i]`` is the 36-nt variable block for variant ``i``.
    - The same call may mix libraries; adapters are free to chunk by
      library internally to amortize per-library context computation.
    """

    def predict_local_variants(
        self,
        library_ids: Sequence[str],
        variant_seqs: Sequence[str],
    ) -> np.ndarray: ...


@runtime_checkable
class FivePrimeUtrReporterExpressionPredictor(Protocol):
    """Predict expression of a 5'-UTR in the fixed Cuperus reporter construct
    (``CYC1`` promoter - [50 bp UTR] - ``HIS3`` CDS - ``CYC1`` terminator).
    Only the UTR varies; the construct is constant, so — unlike the
    marginalized/cassette protocols — this takes raw UTR **strings** (not
    genomic loci) and scores one fixed reporter rather than marginalizing
    over host genes.

    Given a list of UTR strings, return one scalar per UTR = the model's
    predicted ``HIS3`` expression (e.g. summed ``HIS3``-CDS coverage),
    aligned to input order. NaN is allowed for any UTR the model can't
    score. Spearman is the scale-free headline, so adapters may return the
    readout in any monotone-faithful units.

    Used by the Cuperus et al. 2017 5'-UTR MPRA benchmark.
    See ``docs/benchmarks/cuperus_mpra_5utr.md``."""

    def predict_utr_expressions(self, utrs: Sequence[str]) -> np.ndarray: ...
