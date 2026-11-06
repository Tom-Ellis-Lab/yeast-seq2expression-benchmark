"""Brooks et al. SCRaMBLE structural-rearrangement expression benchmark.

Two tiers (see ``docs/benchmarks/brooks_scramble.md``):

  LFC (``lfc_*``) — scalar effect size.  **Per-replicate** true LFCs: for
    each sample, compute ``log2((norm_cov_strain + 1) / (norm_cov_js94_k + 1))``
    for each JS94 deep run ``k`` whose raw CDS read count for the gene
    meets ``MIN_READS_PER_RUN`` (default 10). Yields 0–3 supporting
    LFCs per sample. Predicted LFC is per-replicate too (from per-base
    predicted-count units, alt CDS sum vs native CDS sum against run k).
    Headline metrics are reported **per JS94 replicate** — one Pearson r,
    Spearman ρ and direction balanced accuracy for each run k, over the
    samples that run supports (finite true *and* predicted LFC for k, and
    not ``low_support``). They are never averaged across replicates: the
    three runs cover different gene sets of very different size (a gene
    needs ``MIN_READS_PER_RUN`` reads *in that run*, and the cutoff is
    decided per gene, so the shallow run's cohort is the strongly
    expressed genes), and averaging correlations over unequal, unlike
    cohorts is not a meaningful summary. ``n_scored_per_rep`` records the
    cohort size behind each number; each run carries its own leave-one-out
    ceiling on the same axis, counted by ``n_ceiling_per_rep``.
    Calibration metrics over the ``n_reps ≥ 2`` subset (range defined):
      * Within-range hit rate — fraction of samples where ``pred_lfc``
        lies in ``[min(true_lfcs), max(true_lfcs)]``.
      * Mean standardised residual ``|z|`` where
        ``z = (pred - mean) / max(range, eps)``.

  Shape (``shape_*``) — coverage profile.  Per-base predicted vs per-base
    true Nanopore pileup over the central ``seq_len - 2 * crop`` region;
    metrics: Pearson + Jensen–Shannon divergence per sample, mean across
    the ``n_reps ≥ 1`` AND not ``low_support`` cohort.

**Units.** The benchmark expects adapter predictions in **raw per-base
predicted-count units** (i.e. with any model-specific training transform
inverted inside the adapter); the distribution's ``true_cov_*`` columns
are raw per-base Nanopore pileups. Library size cancels in the LFC
ratio and is normalised away by the sum-to-1 step before the shape
metrics.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Mapping

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import balanced_accuracy_score

from yeastbench.adapters.protocols import CoverageTrackPredictor
from yeastbench.benchmarks.base import (
    Benchmark,
    BenchmarkInfo,
    model_color,
)

PSEUDOCOUNT = 1.0
MIN_READS_PER_RUN = 10  # per-JS94-run raw read floor for that run to
# contribute a per-replicate true_lfc for the sample
RANGE_EPS = 1e-6  # avoid division by zero in |z| when min == max


JS94_REPLICATE_STRAIN_KEYS: tuple[str, ...] = ("JS94_r0", "JS94_r1", "JS94_r2")


def per_rep_lfc_metrics(
    true_lfc_runs: np.ndarray,
    pred_lfc_runs: np.ndarray,
    low_support: np.ndarray,
) -> dict[str, np.ndarray]:
    """Per-JS94-replicate LFC metrics + leave-one-out ceiling.

    The single source of truth for the reported LFC numbers: ``evaluate``,
    ``load_results`` and the cross-model compare path all call this, so the
    three can never drift apart.

    For each replicate ``k`` the metric cohort is the samples where the true
    *and* predicted LFC against run ``k`` are both finite and the sample is
    not ``low_support``. The ceiling cohort drops the predicted-finite
    requirement and instead needs the mean of the *other* replicates to be
    finite, so the two cohorts differ and both sizes are returned.

    Nothing here is averaged across ``k``. The replicates cover different,
    unequal gene sets (the read-depth cutoff is applied per gene per run),
    so a mean over ``k`` would weight unlike cohorts equally.
    """
    n_reps = true_lfc_runs.shape[1]
    finite_t = np.isfinite(true_lfc_runs)
    finite_p = np.isfinite(pred_lfc_runs)
    ok = ~low_support

    out = {
        name: np.full(n_reps, np.nan)
        for name in (
            "pearson_r_per_rep",
            "spearman_rho_per_rep",
            "dir_balanced_acc_per_rep",
            "ceiling_r_per_rep",
            "ceiling_rho_per_rep",
            "ceiling_dir_acc_per_rep",
        )
    }
    out["n_scored_per_rep"] = np.zeros(n_reps, dtype=np.int64)
    out["n_ceiling_per_rep"] = np.zeros(n_reps, dtype=np.int64)

    for k in range(n_reps):
        mk = ok & finite_t[:, k] & finite_p[:, k]
        out["n_scored_per_rep"][k] = int(mk.sum())
        if mk.sum() >= 2:
            t_k, p_k = true_lfc_runs[mk, k], pred_lfc_runs[mk, k]
            out["pearson_r_per_rep"][k] = float(pearsonr(p_k, t_k).statistic)
            out["spearman_rho_per_rep"][k] = float(spearmanr(p_k, t_k).statistic)
            out["dir_balanced_acc_per_rep"][k] = float(
                balanced_accuracy_score(
                    np.sign(t_k).astype(int), np.sign(p_k).astype(int)
                )
            )

        others = [j for j in range(n_reps) if j != k]
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            loo_true = np.nanmean(true_lfc_runs[:, others], axis=1)
        ck = ok & finite_t[:, k] & np.isfinite(loo_true)
        out["n_ceiling_per_rep"][k] = int(ck.sum())
        if ck.sum() >= 2:
            t_k, l_k = true_lfc_runs[ck, k], loo_true[ck]
            out["ceiling_r_per_rep"][k] = float(pearsonr(l_k, t_k).statistic)
            out["ceiling_rho_per_rep"][k] = float(spearmanr(l_k, t_k).statistic)
            out["ceiling_dir_acc_per_rep"][k] = float(
                balanced_accuracy_score(
                    np.sign(t_k).astype(int), np.sign(l_k).astype(int)
                )
            )
    return out


@dataclass(frozen=True)
class BrooksResults:
    sample_ids: list[str]
    # Per-replicate predicted LFCs. For each sample and each JS94 deep WT
    # run k, pred_lfc_runs[i, k] = log2((alt_cds + 1) / (nat_cds_k + 1)),
    # where nat_cds_k is the model's prediction using only JS94's k-th
    # track. NaN if the truth side's j_raws[i, k] is below
    # MIN_READS_PER_RUN (no useful comparison axis).
    pred_lfc_runs: np.ndarray  # (N, 3) float64
    # Per-replicate true LFCs (same shape, same NaN structure).
    true_lfc_runs: np.ndarray  # (N, 3) float64
    n_reps_supported: np.ndarray  # (N,) int — finite-count per row
    low_support: np.ndarray  # (N,) bool — strain-side only
    # Cohort counts
    n_total: int
    n_scored: int  # n_reps >= 1 AND not low_support
    n_calibration: int  # n_reps >= 2 AND not low_support
    n_weak_baseline: int  # n_reps == 0 (per-gene, all JS94 thin)
    n_low_support: int  # low_support == True
    # Per-replicate headline + ceiling — the reported numbers. Each k uses
    # only samples where both true_lfc_runs[:, k] and pred_lfc_runs[:, k]
    # are finite and the sample is not low_support; ceiling_k uses the mean
    # of the *other* JS94 replicates as a "test-retest predictor". Never
    # averaged across k — see the module docstring.
    pearson_r_per_rep: np.ndarray  # (3,) float64
    spearman_rho_per_rep: np.ndarray  # (3,) float64
    dir_balanced_acc_per_rep: np.ndarray  # (3,) float64
    ceiling_r_per_rep: np.ndarray  # (3,) float64
    ceiling_rho_per_rep: np.ndarray  # (3,) float64
    ceiling_dir_acc_per_rep: np.ndarray  # (3,) float64
    # Cohort size behind each per-replicate number. The metric and ceiling
    # masks differ (pred-finite vs leave-one-out-finite), so both are kept.
    n_scored_per_rep: np.ndarray  # (3,) int64
    n_ceiling_per_rep: np.ndarray  # (3,) int64
    # Calibration on the sample-level mean LFCs (n_reps >= 2 cohort)
    within_range_rate: float
    mean_abs_z: float
    # Shape (mean over n_scored; alt construct, full predicted region)
    shape_pearson_mean: float
    shape_js_mean: float


# ── gene-centred windowing (shared by the build + the benchmark) ─────
#
# The single source of truth for how a gene-centred window is cut, so the
# build-time slice extraction and the runtime re-slicing can never drift.
# ``cds_start``/``cds_end`` are 1-based GFF coords (start inclusive), matching
# the build's per-CDS ``start``/``end``.


def cov_key(kind: str, ident: str) -> str:
    """Record name for a construct's sequence (FASTA) / coverage (npz) track.
    ``kind`` is ``"alt"`` or ``"native"``; ``ident`` is the sample_id (alt) or
    gene_id (native). Colons/bars are mapped to ``~`` so the key is safe as a
    FASTA header token and an npz member name."""
    return f"{kind}:{ident}".replace(":", "~").replace("|", "~")


def gene_centre(cds_start: int, cds_end: int) -> int:
    """0-based contig coord of the gene centre, from 1-based inclusive CDS
    coords. The single source of truth for centring — the build's slice
    extraction and the benchmark's re-cut both call this, so every stored slice
    offset stays re-cuttable."""
    return (cds_start - 1 + cds_end) // 2


def window_slice(
    contig_len: int, cds_start: int, cds_end: int, window: int
) -> tuple[int, int, int] | None:
    """Reproduce the build's ``gene_window`` clamp. Returns
    ``(w0, cds_start_in_window, cds_end_in_window)`` — where ``w0`` is the
    window's 0-based contig start — or ``None`` if the contig can't fill a
    full window around the gene, or the CDS doesn't fit inside it. The window
    is gene-centred but clamped to stay inside the contig, so a gene near a
    contig end sits off-centre (and the window is all real sequence, no pad)."""
    if contig_len < window:
        return None
    w0 = max(0, min(gene_centre(cds_start, cds_end) - window // 2, contig_len - window))
    cs = (cds_start - 1) - w0
    ce = cds_end - w0
    if cs < 0 or ce > window:
        return None
    return w0, cs, ce


def _read_fasta(path: Path) -> dict[str, str]:
    """Map FASTA record name → sequence (header token up to first whitespace)."""
    seqs: dict[str, list[str]] = {}
    cur: str | None = None
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                cur = line[1:].split()[0]
                seqs[cur] = []
            elif cur is not None:
                seqs[cur].append(line.strip())
    return {k: "".join(v) for k, v in seqs.items()}


# ── shape metric helpers ─────────────────────────────────────


def _crop_to_output(per_base: np.ndarray, crop: int, out_len: int) -> np.ndarray:
    """Slice a length-`window_len` per-base vector to the
    `[crop, crop + out_len)` region the adapter actually predicts."""
    return per_base[crop : crop + out_len].astype(np.float64)


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """JS divergence in bits; symmetric, bounded [0, 1], no smoothing
    needed (mixture absorbs zeros). p and q must sum to 1."""
    m = 0.5 * (p + q)

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * (_kl(p, m) + _kl(q, m))


# ── benchmark class ──────────────────────────────────────────


class BrooksScrambleBenchmark(Benchmark[CoverageTrackPredictor, BrooksResults]):
    adapter_protocol: ClassVar[type] = CoverageTrackPredictor

    def __init__(self, data_path: Path, info: BenchmarkInfo) -> None:
        # ``data_path`` is the task data directory. The window-agnostic artifact
        # is three files: a per-construct index, a FASTA of generous gene-centred
        # slices (alt + native), and per-base coverage. The benchmark re-cuts
        # each model's window from the slices at eval time (``_materialize``).
        p = Path(data_path)
        self.data_path = p
        self.info = info
        self.data_dir = p if p.is_dir() else p.parent
        idx_path = self.data_dir / "brooks_index.tsv"
        assert idx_path.exists(), (
            f"no brooks_index.tsv in {self.data_dir} — fetch the distribution "
            "with `ybench data get`"
        )
        self.index = pd.read_csv(idx_path, sep="\t").reset_index(drop=True)
        for col in (
            "sample_id",
            "gene_id",
            "strain",
            "copy_idx",
            "strand",
            "alt_contig_len",
            "alt_cds_start",
            "alt_cds_end",
            "alt_slice_start",
            "native_contig_len",
            "native_cds_start",
            "native_cds_end",
            "native_slice_start",
            "norm_cov_strain",
            "norm_cov_js94_runs",
            "js94_reads_runs",
            "low_support",
        ):
            assert col in self.index.columns, f"{col} missing from {idx_path}"
        self._fasta = _read_fasta(self.data_dir / "brooks_constructs.fasta")
        # Lazy NpzFile — per-construct coverage is decompressed on access.
        self._cov = np.load(self.data_dir / "brooks_cov.npz")

    def _materialize(self, window: int) -> pd.DataFrame:
        """Re-cut every construct to ``window``, reproducing the build's
        membership: alt + native windows must fit, ``alt != native`` within the
        window, and byte-identical alt copies of a gene are deduped (kept in
        copy order). Returns one row per surviving construct with the windowed
        alt/native sequence, windowed alt coverage, in-window CDS interval, and
        the window-agnostic truth scalars the scorer needs. Bit-identical to the
        old per-window TSVs (pinned by the golden test)."""
        recs: list[dict] = []
        for (strain, gid), grp in self.index.groupby(["strain", "gene_id"], sort=False):
            nrow = grp.iloc[0]
            nws = window_slice(
                int(nrow.native_contig_len),
                int(nrow.native_cds_start),
                int(nrow.native_cds_end),
                window,
            )
            if nws is None:
                continue  # native window doesn't fit → whole gene drops
            nw0, _ncs, _nce = nws
            nk = cov_key("native", gid)
            noff = nw0 - int(nrow.native_slice_start)
            nat_full = self._fasta[nk]
            assert 0 <= noff and noff + window <= len(nat_full), (
                f"window {window} exceeds the stored slice for native {gid}; "
                "rebuild the distribution with a larger --flank"
            )
            native_seq = nat_full[noff : noff + window]
            seen: set[str] = set()
            for _, row in grp.sort_values("copy_idx").iterrows():
                aws = window_slice(
                    int(row.alt_contig_len),
                    int(row.alt_cds_start),
                    int(row.alt_cds_end),
                    window,
                )
                if aws is None:
                    continue
                aw0, acs, ace = aws
                ak = cov_key("alt", row.sample_id)
                aoff = aw0 - int(row.alt_slice_start)
                alt_full = self._fasta[ak]
                assert 0 <= aoff and aoff + window <= len(alt_full), (
                    f"window {window} exceeds the stored slice for "
                    f"{row.sample_id}; rebuild with a larger --flank"
                )
                alt_seq = alt_full[aoff : aoff + window]
                if alt_seq == native_seq:
                    continue  # no cis change in-window → not a sample
                if alt_seq in seen:
                    continue  # byte-identical duplicate copy
                seen.add(alt_seq)
                recs.append(
                    {
                        "sample_id": row.sample_id,
                        "strain": strain,
                        "strand": row.strand,
                        "alt_seq": alt_seq,
                        "native_seq": native_seq,
                        "true_cov_alt": self._cov[ak][aoff : aoff + window],
                        "cds_start_in_window": acs,
                        "cds_end_in_window": ace,
                        "norm_cov_strain": row.norm_cov_strain,
                        "norm_cov_js94_runs": row.norm_cov_js94_runs,
                        "js94_reads_runs": row.js94_reads_runs,
                        "low_support": bool(row.low_support),
                    }
                )
        return pd.DataFrame(recs).reset_index(drop=True)

    def _parse_norm_runs(self, s: str) -> np.ndarray:
        return np.fromstring(s, sep=",", dtype=np.float64)

    def _parse_raw_runs(self, s: str) -> np.ndarray:
        return np.fromstring(s, sep=",", dtype=np.int64)

    def _run_batched(
        self,
        adapter: CoverageTrackPredictor,
        seqs: list[str],
        strands: list[str],
        strains: list[str | None],
        *,
        desc: str,
    ) -> np.ndarray:
        """Chunk a list of inputs into adapter-sized batches and call
        ``predict_coverage_batch`` on each chunk. Returns shape
        ``(len(seqs), out_len)``."""
        from tqdm import tqdm

        if not seqs:
            return np.empty((0, 0), dtype=np.float64)
        bs = max(1, int(getattr(adapter, "batch_size", 1)))
        out_chunks: list[np.ndarray] = []
        for start in tqdm(range(0, len(seqs), bs), desc=desc, ncols=80):
            end = min(start + bs, len(seqs))
            arr = adapter.predict_coverage_batch(
                seqs=seqs[start:end],
                strands=strands[start:end],
                strains=strains[start:end],
            )
            out_chunks.append(np.asarray(arr, dtype=np.float64))
        return np.concatenate(out_chunks, axis=0)

    def evaluate(self, adapter: CoverageTrackPredictor) -> BrooksResults:
        crop = adapter.crop_bp_each_side
        out_len = adapter.seq_len - 2 * crop  # per-base prediction length
        # Re-cut every construct to the adapter's receptive field (membership +
        # dedup reproduced inside). One task serves any model with window <= the
        # build flank.
        df = self._materialize(adapter.seq_len)
        n = len(df)
        n_reps = len(JS94_REPLICATE_STRAIN_KEYS)
        if n == 0:
            # No construct survives the window/membership rules at this seq_len
            # (e.g. a subset build, or a window larger than every contig). Return
            # an empty cohort rather than indexing a column-less DataFrame.
            nan_reps = np.full(n_reps, np.nan)
            return BrooksResults(
                sample_ids=[],
                pred_lfc_runs=np.empty((0, n_reps), dtype=np.float64),
                true_lfc_runs=np.empty((0, n_reps), dtype=np.float64),
                n_reps_supported=np.empty(0, dtype=np.int64),
                low_support=np.empty(0, dtype=bool),
                n_total=0,
                n_scored=0,
                n_calibration=0,
                n_weak_baseline=0,
                n_low_support=0,
                pearson_r_per_rep=nan_reps.copy(),
                spearman_rho_per_rep=nan_reps.copy(),
                dir_balanced_acc_per_rep=nan_reps.copy(),
                ceiling_r_per_rep=nan_reps.copy(),
                ceiling_rho_per_rep=nan_reps.copy(),
                ceiling_dir_acc_per_rep=nan_reps.copy(),
                n_scored_per_rep=np.zeros(n_reps, dtype=np.int64),
                n_ceiling_per_rep=np.zeros(n_reps, dtype=np.int64),
                within_range_rate=float("nan"),
                mean_abs_z=float("nan"),
                shape_pearson_mean=float("nan"),
                shape_js_mean=float("nan"),
            )
        # Per-replicate prediction + truth LFCs, same (N, 3) shape.
        pred_lfc_runs = np.full((n, n_reps), np.nan, dtype=np.float64)
        true_lfc_runs = np.full((n, n_reps), np.nan, dtype=np.float64)
        shape_pearson = np.full(n, np.nan)
        shape_js = np.full(n, np.nan)

        varies_by_strain = bool(getattr(adapter, "varies_by_strain", True))

        # ── Phase 1: per-replicate true LFCs (no GPU work) ──
        for i, row in df.iterrows():
            s_norm = float(row.norm_cov_strain)
            j_norms = self._parse_norm_runs(row.norm_cov_js94_runs)
            j_raws = self._parse_raw_runs(row.js94_reads_runs)
            for k in range(min(len(j_norms), len(j_raws), n_reps)):
                if j_raws[k] < MIN_READS_PER_RUN:
                    continue
                true_lfc_runs[i, k] = float(
                    np.log2((s_norm + PSEUDOCOUNT) / (j_norms[k] + PSEUDOCOUNT))
                )

        # ── Phase 2: batched alt predictions across all samples ──
        all_alt_seqs = df.alt_seq.tolist()
        all_strands = df.strand.tolist()
        all_strains = df.strain.tolist()
        pred_alt_all = self._run_batched(
            adapter,
            all_alt_seqs,
            all_strands,
            all_strains,
            desc=f"alt   (n={n})",
        )
        assert pred_alt_all.shape == (n, out_len), (
            f"adapter returned {pred_alt_all.shape}, expected ({n}, {out_len})"
        )

        # ── Phase 3: batched native predictions ──
        # `pred_nat_runs[i, k]` is the model's prediction for sample i
        # against JS94 replicate k. NaN where unused (truth NaN).
        pred_nat_runs = np.full((n, n_reps, out_len), np.nan, dtype=np.float64)
        all_native_seqs = df.native_seq.tolist()
        if varies_by_strain:
            # One batched call per JS94 replicate; restrict to samples
            # that need this replicate (truth is finite for it).
            for k, alias in enumerate(JS94_REPLICATE_STRAIN_KEYS):
                mask = np.isfinite(true_lfc_runs[:, k])
                idx = np.where(mask)[0]
                if idx.size == 0:
                    continue
                sub_pred = self._run_batched(
                    adapter,
                    seqs=[all_native_seqs[i] for i in idx],
                    strands=[all_strands[i] for i in idx],
                    strains=[alias] * idx.size,
                    desc=f"nat {alias} (n={idx.size})",
                )
                pred_nat_runs[idx, k] = sub_pred
        else:
            # One forward across all samples; broadcast into supported reps.
            pred_nat_one = self._run_batched(
                adapter,
                seqs=all_native_seqs,
                strands=all_strands,
                strains=["JS94"] * n,
                desc=f"nat   (n={n})",
            )
            for k in range(n_reps):
                mask = np.isfinite(true_lfc_runs[:, k])
                pred_nat_runs[mask, k] = pred_nat_one[mask]

        # ── Phase 4: per-sample LFCs + shape (CPU only) ──
        for i, row in df.iterrows():
            pred_alt = pred_alt_all[i]
            cs = max(0, int(row.cds_start_in_window) - crop)
            ce = min(out_len, int(row.cds_end_in_window) - crop)
            if ce <= cs:
                continue
            alt_cds = pred_alt[cs:ce].sum()

            for k in range(n_reps):
                if not np.isfinite(true_lfc_runs[i, k]):
                    continue
                nat_cds_k = pred_nat_runs[i, k, cs:ce].sum()
                pred_lfc_runs[i, k] = float(
                    np.log2((alt_cds + PSEUDOCOUNT) / (nat_cds_k + PSEUDOCOUNT))
                )

            true_alt = _crop_to_output(
                np.asarray(row.true_cov_alt, dtype=np.int32), crop, out_len
            )
            if true_alt.sum() > 0 and pred_alt.sum() > 0:
                shape_pearson[i] = float(pearsonr(true_alt, pred_alt).statistic)
                p = true_alt / true_alt.sum()
                q = pred_alt / pred_alt.sum()
                shape_js[i] = _js_divergence(p, q)

        # Per-sample replicate counts (truth side; pred side mirrors it
        # by construction since we only ran pred when truth was finite).
        finite_true = np.isfinite(true_lfc_runs)
        n_reps_supported = finite_true.sum(axis=1).astype(np.int64)

        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean_true = np.where(
                n_reps_supported > 0,
                np.nanmean(true_lfc_runs, axis=1),
                np.nan,
            )
            mean_pred = np.where(
                n_reps_supported > 0,
                np.nanmean(pred_lfc_runs, axis=1),
                np.nan,
            )

        low = df.low_support.to_numpy(dtype=bool)
        scored_mask = (
            (~low)
            & (n_reps_supported >= 1)
            & np.isfinite(mean_pred)
            & np.isfinite(mean_true)
        )
        calib_mask = scored_mask & (n_reps_supported >= 2)
        n_scored = int(scored_mask.sum())
        n_calibration = int(calib_mask.sum())
        n_weak_baseline = int((n_reps_supported == 0).sum())
        n_low = int(low.sum())

        # ── per-replicate LFC metrics + LOO ceiling (the reported numbers) ──
        per_rep = per_rep_lfc_metrics(true_lfc_runs, pred_lfc_runs, low)

        # ── calibration on the sample-mean LFCs (n_reps >= 2 cohort) ──
        if n_calibration < 1:
            within_range_rate = mean_abs_z = float("nan")
        else:
            idx = np.where(calib_mask)[0]
            hits = 0
            zs = []
            for ii in idx:
                row_runs = true_lfc_runs[ii][finite_true[ii]]
                lo_v, hi_v = float(row_runs.min()), float(row_runs.max())
                if lo_v <= mean_pred[ii] <= hi_v:
                    hits += 1
                rng = max(hi_v - lo_v, RANGE_EPS)
                zs.append(abs(mean_pred[ii] - mean_true[ii]) / rng)
            within_range_rate = hits / n_calibration
            mean_abs_z = float(np.mean(zs))

        # nanmean over the scored mask can hit an all-NaN slice (every scored
        # sample had a flat/all-zero pred or true profile) — suppress the
        # RuntimeWarning the same way the LFC-mean block above does.
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            shape_pearson_mean = (
                float(np.nanmean(shape_pearson[scored_mask]))
                if n_scored
                else float("nan")
            )
            shape_js_mean = (
                float(np.nanmean(shape_js[scored_mask])) if n_scored else float("nan")
            )

        return BrooksResults(
            sample_ids=df.sample_id.tolist(),
            pred_lfc_runs=pred_lfc_runs,
            true_lfc_runs=true_lfc_runs,
            n_reps_supported=n_reps_supported,
            low_support=low,
            n_total=n,
            n_scored=n_scored,
            n_calibration=n_calibration,
            n_weak_baseline=n_weak_baseline,
            n_low_support=n_low,
            **per_rep,
            within_range_rate=within_range_rate,
            mean_abs_z=mean_abs_z,
            shape_pearson_mean=shape_pearson_mean,
            shape_js_mean=shape_js_mean,
        )

    def plot(self, results: BrooksResults, out_dir: Path) -> None:
        import matplotlib.pyplot as plt

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        title_model = out_dir.name.split("__")[0] if "__" in out_dir.name else ""

        finite_true = np.isfinite(results.true_lfc_runs)
        finite_pred = np.isfinite(results.pred_lfc_runs)
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean_true = np.where(
                results.n_reps_supported > 0,
                np.nanmean(results.true_lfc_runs, axis=1),
                np.nan,
            )
            mean_pred = np.where(
                results.n_reps_supported > 0,
                np.nanmean(results.pred_lfc_runs, axis=1),
                np.nan,
            )
        m = (
            (~results.low_support)
            & (results.n_reps_supported >= 1)
            & np.isfinite(mean_pred)
            & np.isfinite(mean_true)
        )

        # ── LFC scatter — mean pred vs mean true, with replicate
        # envelopes shown as crosshair error bars on both axes ──
        p_arr, t_arr = mean_pred[m], mean_true[m]
        true_lo = np.array(
            [results.true_lfc_runs[ii][finite_true[ii]].min() for ii in np.where(m)[0]]
        )
        true_hi = np.array(
            [results.true_lfc_runs[ii][finite_true[ii]].max() for ii in np.where(m)[0]]
        )
        pred_lo = np.array(
            [
                results.pred_lfc_runs[ii][finite_pred[ii]].min()
                if finite_pred[ii].any()
                else mean_pred[ii]
                for ii in np.where(m)[0]
            ]
        )
        pred_hi = np.array(
            [
                results.pred_lfc_runs[ii][finite_pred[ii]].max()
                if finite_pred[ii].any()
                else mean_pred[ii]
                for ii in np.where(m)[0]
            ]
        )

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.axhline(0, color="grey", lw=0.5)
        ax.axvline(0, color="grey", lw=0.5)
        # Clamp to >=0; tiny float-precision noise around mean ≈ min ≈ max
        # for broadcast (varies_by_strain=False) predictions has slipped
        # below zero in practice and matplotlib's errorbar rejects it.
        x_lo = np.maximum(t_arr - true_lo, 0.0)
        x_hi = np.maximum(true_hi - t_arr, 0.0)
        y_lo = np.maximum(p_arr - pred_lo, 0.0)
        y_hi = np.maximum(pred_hi - p_arr, 0.0)
        ax.errorbar(
            t_arr,
            p_arr,
            xerr=[x_lo, x_hi],
            yerr=[y_lo, y_hi],
            fmt="o",
            ms=4,
            ecolor="lightgrey",
            elinewidth=1,
            alpha=0.7,
        )
        lim = max(np.nanmax(np.abs(t_arr)), np.nanmax(np.abs(p_arr)), 1) + 0.5
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal", "box")
        ax.set_xlabel("true log2 LFC (mean over supporting JS94 runs)")
        ax.set_ylabel("predicted log2 LFC (mean over supporting JS94 runs)")
        # Points are the per-sample means (with replicate envelopes); the
        # reported metrics are per JS94 replicate, so the title lists all
        # three rather than a single aggregate.
        per_rep_lines = "\n".join(
            f"{alias} (n={int(results.n_scored_per_rep[k])}): "
            f"dir-acc={results.dir_balanced_acc_per_rep[k]:.3f} "
            f"(ceiling {results.ceiling_dir_acc_per_rep[k]:.3f})  "
            f"r={results.pearson_r_per_rep[k]:.3f} "
            f"(ceiling {results.ceiling_r_per_rep[k]:.3f})  "
            f"ρ={results.spearman_rho_per_rep[k]:.3f}"
            for k, alias in enumerate(JS94_REPLICATE_STRAIN_KEYS)
        )
        ax.set_title(
            "Brooks SCRaMBLE — LFC"
            + (f" — {title_model}" if title_model else "")
            + f"\nn_scored={results.n_scored} overall; metrics per JS94 replicate\n"
            + per_rep_lines
            + f"\ncalibration (n={results.n_calibration}): "
            f"within-range={results.within_range_rate:.3f}  "
            f"|z|={results.mean_abs_z:.3f}",
            fontsize=8,
        )
        fig.tight_layout()
        fig.savefig(out_dir / "lfc_scatter.png", dpi=150)
        plt.close(fig)

        # ── Per-sample interval plot — every scored sample side by
        # side, true (blue) and pred (orange) ranges with mean dot.
        # Wide canvas; sort by mean true LFC for visual order. ──
        idx_sorted = np.array(
            sorted(
                np.where(m)[0],
                key=lambda i: mean_true[i],
            )
        )
        K = len(idx_sorted)
        if K > 0:
            fig_w = max(8.0, 0.08 * K)  # ~0.08" per sample
            fig, ax = plt.subplots(figsize=(fig_w, 6))
            ax.axhline(0, color="grey", lw=0.3)
            x = np.arange(K, dtype=float)
            off = 0.18
            # True (blue)
            true_means = mean_true[idx_sorted]
            t_lo = np.array(
                [results.true_lfc_runs[ii][finite_true[ii]].min() for ii in idx_sorted]
            )
            t_hi = np.array(
                [results.true_lfc_runs[ii][finite_true[ii]].max() for ii in idx_sorted]
            )
            ax.vlines(x - off, t_lo, t_hi, colors="#1f77b4", lw=1.0, alpha=0.7)
            ax.scatter(x - off, true_means, s=8, c="#1f77b4", label="true")
            # Pred (orange) — handle samples with only 1 finite pred (no range)
            pred_means = mean_pred[idx_sorted]
            p_lo = np.array(
                [
                    results.pred_lfc_runs[ii][finite_pred[ii]].min()
                    if finite_pred[ii].any()
                    else mean_pred[ii]
                    for ii in idx_sorted
                ]
            )
            p_hi = np.array(
                [
                    results.pred_lfc_runs[ii][finite_pred[ii]].max()
                    if finite_pred[ii].any()
                    else mean_pred[ii]
                    for ii in idx_sorted
                ]
            )
            ax.vlines(x + off, p_lo, p_hi, colors="#ff7f0e", lw=1.0, alpha=0.7)
            ax.scatter(x + off, pred_means, s=8, c="#ff7f0e", label="pred")
            ax.set_xlim(-1, K)
            ax.set_xlabel(f"sample (sorted by mean true LFC, n={K})")
            ax.set_ylabel("log2 LFC (alt / native)")
            ax.set_title(
                "Brooks SCRaMBLE — per-sample LFC ranges"
                + (f" — {title_model}" if title_model else "")
                + "   |   r per JS94 replicate: "
                + " / ".join(
                    f"{results.pearson_r_per_rep[k]:.3f}"
                    for k in range(len(JS94_REPLICATE_STRAIN_KEYS))
                )
                + "  (ceiling "
                + " / ".join(
                    f"{results.ceiling_r_per_rep[k]:.3f}"
                    for k in range(len(JS94_REPLICATE_STRAIN_KEYS))
                )
                + ")",
                fontsize=9,
            )
            ax.legend(loc="upper left", fontsize=9)
            fig.tight_layout()
            fig.savefig(out_dir / "lfc_per_sample.png", dpi=100)
            plt.close(fig)

    def save_results(self, results: BrooksResults, out_dir: Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "pred_lfc_runs.npy", results.pred_lfc_runs)
        np.save(out_dir / "true_lfc_runs.npy", results.true_lfc_runs)
        np.save(out_dir / "n_reps_supported.npy", results.n_reps_supported)
        (out_dir / "samples.json").write_text(
            json.dumps(
                {
                    "sample_ids": results.sample_ids,
                    "low_support": results.low_support.tolist(),
                },
                indent=2,
            )
        )

    def load_results(self, out_dir: Path) -> BrooksResults:
        out_dir = Path(out_dir)
        meta = json.loads((out_dir / "samples.json").read_text())
        pred_lfc_runs = np.load(out_dir / "pred_lfc_runs.npy")
        true_lfc_runs = np.load(out_dir / "true_lfc_runs.npy")
        n_reps_supported = np.load(out_dir / "n_reps_supported.npy").astype(np.int64)
        low = np.asarray(meta["low_support"], dtype=bool)
        n = len(low)

        finite_true = np.isfinite(true_lfc_runs)
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean_true = np.where(
                n_reps_supported > 0, np.nanmean(true_lfc_runs, axis=1), np.nan
            )
            mean_pred = np.where(
                n_reps_supported > 0, np.nanmean(pred_lfc_runs, axis=1), np.nan
            )
        scored = (
            (~low)
            & (n_reps_supported >= 1)
            & np.isfinite(mean_pred)
            & np.isfinite(mean_true)
        )
        calib = scored & (n_reps_supported >= 2)

        per_rep = per_rep_lfc_metrics(true_lfc_runs, pred_lfc_runs, low)

        if calib.sum() >= 1:
            hits = 0
            zs = []
            for ii in np.where(calib)[0]:
                runs = true_lfc_runs[ii][finite_true[ii]]
                lo_v, hi_v = float(runs.min()), float(runs.max())
                if lo_v <= mean_pred[ii] <= hi_v:
                    hits += 1
                zs.append(
                    abs(mean_pred[ii] - mean_true[ii]) / max(hi_v - lo_v, RANGE_EPS)
                )
            within = hits / calib.sum()
            mz = float(np.mean(zs))
        else:
            within = mz = float("nan")

        return BrooksResults(
            sample_ids=meta["sample_ids"],
            pred_lfc_runs=pred_lfc_runs,
            true_lfc_runs=true_lfc_runs,
            n_reps_supported=n_reps_supported,
            low_support=low,
            n_total=n,
            n_scored=int(scored.sum()),
            n_calibration=int(calib.sum()),
            n_weak_baseline=int((n_reps_supported == 0).sum()),
            n_low_support=int(low.sum()),
            **per_rep,
            within_range_rate=within,
            mean_abs_z=mz,
            shape_pearson_mean=float("nan"),
            shape_js_mean=float("nan"),
        )

    def summary_dict(self, results: BrooksResults) -> dict[str, Any]:
        out: dict[str, Any] = {
            "n_total": results.n_total,
            "n_scored": results.n_scored,
            "n_calibration": results.n_calibration,
            "n_weak_baseline": results.n_weak_baseline,
            "n_low_support": results.n_low_support,
            # Reported per JS94 replicate; never averaged across replicates
            # (the three cover different, unequal gene sets).
            "lfc_n_scored_per_rep": results.n_scored_per_rep.tolist(),
            "lfc_pearson_r_per_rep": results.pearson_r_per_rep.tolist(),
            "lfc_spearman_rho_per_rep": results.spearman_rho_per_rep.tolist(),
            "lfc_dir_balanced_acc_per_rep": results.dir_balanced_acc_per_rep.tolist(),
            "lfc_n_ceiling_per_rep": results.n_ceiling_per_rep.tolist(),
            "lfc_ceiling_r_per_rep": results.ceiling_r_per_rep.tolist(),
            "lfc_ceiling_rho_per_rep": results.ceiling_rho_per_rep.tolist(),
            "lfc_ceiling_dir_acc_per_rep": results.ceiling_dir_acc_per_rep.tolist(),
            "lfc_within_range_rate": results.within_range_rate,
            "lfc_mean_abs_z": results.mean_abs_z,
            "shape_pearson_mean": results.shape_pearson_mean,
            "shape_js_mean": results.shape_js_mean,
        }
        # The cross-task summary.csv / summary.md keep only scalar values, so
        # also emit each replicate under its own key. Named, not indexed, so
        # the aggregate table stays readable.
        per_rep_scalars = (
            ("lfc_n_scored", results.n_scored_per_rep),
            ("lfc_pearson_r", results.pearson_r_per_rep),
            ("lfc_spearman_rho", results.spearman_rho_per_rep),
            ("lfc_dir_balanced_acc", results.dir_balanced_acc_per_rep),
            ("lfc_n_ceiling", results.n_ceiling_per_rep),
            ("lfc_ceiling_r", results.ceiling_r_per_rep),
            ("lfc_ceiling_rho", results.ceiling_rho_per_rep),
            ("lfc_ceiling_dir_acc", results.ceiling_dir_acc_per_rep),
        )
        for stem, arr in per_rep_scalars:
            for k, alias in enumerate(JS94_REPLICATE_STRAIN_KEYS):
                if k < len(arr):
                    out[f"{stem}_{alias}"] = (
                        int(arr[k]) if stem.startswith("lfc_n_") else float(arr[k])
                    )
        return out

    def headline(self, results: BrooksResults) -> str:
        # One line per JS94 replicate — the three are not comparable to each
        # other (different gene sets, different ceilings), so no aggregate.
        lines = [f"LFC per JS94 replicate (n_scored={results.n_scored} overall):"]
        for k, alias in enumerate(JS94_REPLICATE_STRAIN_KEYS):
            lines.append(
                f"  {alias} (n={int(results.n_scored_per_rep[k])}): "
                f"dir-acc {results.dir_balanced_acc_per_rep[k]:.3f} "
                f"(ceiling {results.ceiling_dir_acc_per_rep[k]:.3f})  "
                f"r {results.pearson_r_per_rep[k]:.3f} "
                f"(ceiling {results.ceiling_r_per_rep[k]:.3f})  "
                f"ρ {results.spearman_rho_per_rep[k]:.3f} "
                f"(ceiling {results.ceiling_rho_per_rep[k]:.3f})"
            )
        lines.append(
            f"calibration (n={results.n_calibration}): "
            f"within-range {results.within_range_rate:.3f}  "
            f"|z| {results.mean_abs_z:.3f}  | "
            f"shape: r̄ {results.shape_pearson_mean:.3f}  "
            f"JS̄ {results.shape_js_mean:.3f}"
        )
        return "\n".join(lines)

    # ── Cross-model comparison override ──────────────────────────────────
    #
    # One registry task now, but the two models still produce different sample
    # sets at run time (a model with a larger window keeps byte-identical copies
    # that a smaller window dedups), so the headline is computed on the
    # **intersection of sample_ids** for an apples-to-apples comparison;
    # per-model full-set metrics are recorded as secondary. Generalises to N.

    def compare_plot(
        self,
        model_dirs: Mapping[str, Path],
        out_dir: Path,
    ) -> Path | None:
        """Shared-cohort comparison across N models. Writes:

          - ``shared_lfc.svg``: bar chart of Pearson r / Spearman ρ /
            dir-acc per model on the shared cohort, with the LOO
            reproducibility ceiling marked as a grey dashed line.
          - ``shared_per_sample.png``: per-sample interval plot — every
            shared-cohort sample gets one blue range (truth) plus one
            range per model, sorted left-to-right by mean true LFC.
          - ``summary.json``: shared-cohort + secondary full-set numbers
            for every model.

        Returns the LFC plot path so the runner can include it in
        the cross-task mosaic."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        loaded = {name: _load_brooks_run_dir(Path(d)) for name, d in model_dirs.items()}
        # Drop any that don't have the per-replicate arrays (older / partial
        # runs). Need at least 2 to compare.
        loaded = {n: r for n, r in loaded.items() if r is not None}
        if len(loaded) < 2:
            return None

        shared = sorted(
            set.intersection(*(set(r["sample_ids"]) for r in loaded.values()))
        )
        if not shared:
            return None
        indexers = {
            name: np.array(
                [{sid: i for i, sid in enumerate(r["sample_ids"])}[s] for s in shared]
            )
            for name, r in loaded.items()
        }

        shared_cohort: dict[str, Any] = {
            "sample_ids": shared,
            "n": len(shared),
        }
        for name, r in loaded.items():
            shared_cohort[name] = _brooks_metrics(r, indexers[name])

        secondary = {
            name: _brooks_metrics(r, np.arange(len(r["sample_ids"])))
            for name, r in loaded.items()
        }

        out_summary = {
            "shared_cohort": shared_cohort,
            "secondary_full_set": secondary,
            "note": (
                "Headline metrics are computed on the intersection of all "
                "models' sample sets. Full-set metrics for each model are "
                "kept as secondary so the gap is documented. LFC metrics are "
                "reported per JS94 replicate and are never averaged across "
                "replicates: each replicate scores a different, unequally "
                "sized gene set (the read-depth cutoff is applied per gene "
                "per run) against its own leave-one-out ceiling, so the three "
                "are not comparable to one another either."
            ),
        }
        (out_dir / "summary.json").write_text(json.dumps(out_summary, indent=2))

        plot_path = out_dir / "shared_lfc.svg"
        _plot_brooks_shared_metrics(loaded, indexers, shared_cohort, plot_path)
        _plot_brooks_shared_per_sample(
            loaded, indexers, out_dir / "shared_per_sample.svg"
        )
        return plot_path


# ── Brooks-specific compare helpers (shared-cohort intersection
#    used by `ybench compare`) ─────────────────────────────────────────


def _load_brooks_run_dir(model_dir: Path) -> dict | None:
    """Load one model's per-replicate prediction arrays + sample IDs.
    Returns None if the expected files aren't present (older / partial
    runs from before the per-replicate framework)."""
    samples_path = model_dir / "samples.json"
    pred_path = model_dir / "pred_lfc_runs.npy"
    true_path = model_dir / "true_lfc_runs.npy"
    n_reps_path = model_dir / "n_reps_supported.npy"
    if not all(p.exists() for p in (samples_path, pred_path, true_path, n_reps_path)):
        return None
    meta = json.loads(samples_path.read_text())
    return {
        "sample_ids": meta["sample_ids"],
        "low_support": np.asarray(meta["low_support"], dtype=bool),
        "pred_lfc_runs": np.load(pred_path),
        "true_lfc_runs": np.load(true_path),
        "n_reps_supported": np.load(n_reps_path),
    }


def _brooks_metrics(d: dict, idx: np.ndarray) -> dict:
    """Per-replicate r / ρ / dir-acc + LOO ceiling + calibration metrics
    on the row subset ``idx``. Shares ``per_rep_lfc_metrics`` with the
    in-class compute path so the two can never drift."""
    pred = d["pred_lfc_runs"][idx]
    true = d["true_lfc_runs"][idx]
    n_reps = d["n_reps_supported"][idx]
    low = d["low_support"][idx]

    finite_t = np.isfinite(true)
    per_rep = per_rep_lfc_metrics(true, pred, low)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_true = np.where(n_reps > 0, np.nanmean(true, axis=1), np.nan)
        mean_pred = np.where(n_reps > 0, np.nanmean(pred, axis=1), np.nan)

    calib_mask = (
        (~low) & (n_reps >= 2) & np.isfinite(mean_true) & np.isfinite(mean_pred)
    )
    if calib_mask.sum() >= 1:
        hits = 0
        zs = []
        for ii in np.where(calib_mask)[0]:
            runs = true[ii][finite_t[ii]]
            lo, hi = float(runs.min()), float(runs.max())
            if lo <= mean_pred[ii] <= hi:
                hits += 1
            zs.append(abs(mean_pred[ii] - mean_true[ii]) / max(hi - lo, RANGE_EPS))
        within = hits / calib_mask.sum()
        mz = float(np.mean(zs))
    else:
        within = mz = float("nan")

    scored = (~low) & (n_reps >= 1)
    return {
        "n_total": int(len(idx)),
        "n_low_support": int(low.sum()),
        "n_scored": int(scored.sum()),
        "n_calibration": int(calib_mask.sum()),
        "n_weak_baseline": int((n_reps == 0).sum()),
        # Per JS94 replicate; never averaged across replicates.
        "n_scored_per_rep": per_rep["n_scored_per_rep"].tolist(),
        "pearson_r_per_rep": per_rep["pearson_r_per_rep"].tolist(),
        "spearman_rho_per_rep": per_rep["spearman_rho_per_rep"].tolist(),
        "dir_balanced_acc_per_rep": per_rep["dir_balanced_acc_per_rep"].tolist(),
        "n_ceiling_per_rep": per_rep["n_ceiling_per_rep"].tolist(),
        "ceiling_pearson_r_per_rep": per_rep["ceiling_r_per_rep"].tolist(),
        "ceiling_spearman_rho_per_rep": per_rep["ceiling_rho_per_rep"].tolist(),
        "ceiling_dir_balanced_acc_per_rep": per_rep["ceiling_dir_acc_per_rep"].tolist(),
        "within_range_rate": within,
        "mean_abs_z": mz,
    }


def _plot_brooks_shared_metrics(
    loaded: Mapping[str, dict],
    indexers: Mapping[str, np.ndarray],
    shared_cohort: dict,
    out_path: Path,
) -> None:
    """Bar chart of Pearson r / Spearman ρ / dir-acc per model on the
    shared cohort, one row per (metric, JS94 replicate). Each replicate
    carries its own leave-one-out ceiling and its own cohort size — the
    three are not averaged, and are not comparable to each other (they
    cover different gene sets)."""
    import matplotlib.pyplot as plt

    metrics = [
        ("Pearson r", "pearson_r_per_rep", "ceiling_pearson_r_per_rep"),
        ("Spearman ρ", "spearman_rho_per_rep", "ceiling_spearman_rho_per_rep"),
        ("dir-acc", "dir_balanced_acc_per_rep", "ceiling_dir_balanced_acc_per_rep"),
    ]
    model_names = sorted(loaded.keys())
    colors = [model_color(m, model_names) for m in model_names]
    n_reps = len(shared_cohort[model_names[0]]["pearson_r_per_rep"])
    rows = [(mn, key, ck, k) for mn, key, ck in metrics for k in range(n_reps)]

    fig, axes = plt.subplots(len(rows), 1, figsize=(8, 1.5 * len(rows)), squeeze=False)
    for i, (metric_name, key, ceil_key, k) in enumerate(rows):
        ax = axes[i, 0]
        ys = [shared_cohort[m][key][k] for m in model_names]
        bars = ax.barh(model_names, ys, color=colors, alpha=0.85)
        ax.axvline(0, color="grey", lw=0.5)
        # The ceiling depends only on the truth labels, which are identical
        # for all models on the shared cohort — read it from the first model.
        ceil = shared_cohort[model_names[0]][ceil_key][k]
        if np.isfinite(ceil):
            ax.axvline(
                ceil, color="grey", lw=1.0, ls="--", label=f"LOO ceiling {ceil:+.3f}"
            )
            ax.legend(loc="lower right", fontsize=8)
        for b, v in zip(bars, ys):
            ax.text(
                v + 0.005,
                b.get_y() + b.get_height() / 2,
                f"{v:+.3f}",
                va="center",
                fontsize=9,
            )
        all_vals = [v for v in ys + [ceil] if np.isfinite(v)] or [0.0]
        ax.set_xlim(min(-0.05, min(all_vals) - 0.05), max(1.0, *all_vals) * 1.05 + 0.05)
        n_k = shared_cohort[model_names[0]]["n_scored_per_rep"][k]
        ax.set_title(
            f"{metric_name}  ({JS94_REPLICATE_STRAIN_KEYS[k]}, n={n_k})",
            fontsize=10,
        )
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# Per-sample plot: truth gets a fixed colour (grey) so it visually
# anchors every sample, models pick from the shared MODEL_COLORS so
# they match the metrics-bar plot above and the cross-task mosaic.
_TRUTH_COLOR = "#444444"


def _plot_brooks_shared_per_sample(
    loaded: Mapping[str, dict],
    indexers: Mapping[str, np.ndarray],
    out_path: Path,
) -> None:
    """Per-sample interval plot on the shared cohort. Each scored sample
    becomes one column: truth (blue) + one range per model (red, green,
    purple, …). Models with `varies_by_strain=False` collapse to a dot.
    Sorted left-to-right by mean true LFC."""
    import matplotlib.pyplot as plt

    model_names = sorted(loaded.keys())
    # Use the first model's truth as the canonical truth (identical
    # across models on the shared cohort by construction).
    ref = loaded[model_names[0]]
    ref_idx = indexers[model_names[0]]
    true = ref["true_lfc_runs"][ref_idx]
    n_reps_ref = ref["n_reps_supported"][ref_idx]
    finite_t = np.isfinite(true)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_true = np.where(n_reps_ref > 0, np.nanmean(true, axis=1), np.nan)

    # Which samples are scored on every model?
    scored_mask = (
        (~ref["low_support"][ref_idx]) & (n_reps_ref >= 1) & np.isfinite(mean_true)
    )
    for m in model_names:
        idx = indexers[m]
        scored_mask = scored_mask & (~loaded[m]["low_support"][idx])
        pred = loaded[m]["pred_lfc_runs"][idx]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            finite_p = np.isfinite(pred)
            mean_pred = np.where(finite_p.any(axis=1), np.nanmean(pred, axis=1), np.nan)
        scored_mask = scored_mask & np.isfinite(mean_pred)

    K = int(scored_mask.sum())
    if K == 0:
        return
    idx_sorted = np.array(sorted(np.where(scored_mask)[0], key=lambda i: mean_true[i]))
    x = np.arange(K, dtype=float)

    # Layout: 1 (truth) + N model entries; one column per sample with
    # equal-spaced x-offsets.
    n_series = 1 + len(model_names)
    total_width = 0.85
    spacing = total_width / n_series
    offsets = np.linspace(
        -total_width / 2 + spacing / 2,
        total_width / 2 - spacing / 2,
        n_series,
    )

    fig_w = max(8.0, 0.08 * K)
    fig, ax = plt.subplots(figsize=(fig_w, 6))
    ax.axhline(0, color="grey", lw=0.3)

    def _ranges(
        arr: np.ndarray, mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        lo = np.array(
            [arr[i][mask[i]].min() if mask[i].any() else np.nan for i in idx_sorted]
        )
        hi = np.array(
            [arr[i][mask[i]].max() if mask[i].any() else np.nan for i in idx_sorted]
        )
        mn = np.array(
            [arr[i][mask[i]].mean() if mask[i].any() else np.nan for i in idx_sorted]
        )
        return lo, hi, mn

    t_lo, t_hi, t_mn = _ranges(true, finite_t)
    ax.vlines(x + offsets[0], t_lo, t_hi, colors=_TRUTH_COLOR, lw=1.0, alpha=0.7)
    ax.scatter(x + offsets[0], t_mn, s=8, c=_TRUTH_COLOR, label="true")

    for j, m in enumerate(model_names):
        idx = indexers[m]
        pred = loaded[m]["pred_lfc_runs"][idx]
        finite_p = np.isfinite(pred)
        lo, hi, mn = _ranges(pred, finite_p)
        color = model_color(m, model_names)
        ax.vlines(x + offsets[1 + j], lo, hi, colors=color, lw=1.0, alpha=0.7)
        ax.scatter(x + offsets[1 + j], mn, s=8, c=color, label=f"{m} pred")

    ax.set_xlim(-1, K)
    ax.set_xlabel(f"sample (sorted by mean true LFC, n={K})")
    ax.set_ylabel("log2 LFC (alt / native)")
    ax.set_title("Brooks SCRaMBLE — per-sample LFC ranges (shared cohort)")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
