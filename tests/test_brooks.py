"""Tests for the Brooks SCRaMBLE benchmark.

The benchmark consumes a window-agnostic artifact (index TSV + FASTA of generous
gene-centred slices + per-base coverage npz) and re-cuts each construct to the
adapter's receptive field at run time. The tests build a tiny synthetic artifact
and verify the LFC + shape metrics, the window-dependent membership/dedup, and
(against the shipped data, when present) bit-exact reproduction of the original
per-window distributions.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from yeastbench.adapters.protocols import CoverageTrackPredictor
from yeastbench.benchmarks.base import BenchmarkInfo
from yeastbench.benchmarks.brooks import (
    BrooksScrambleBenchmark,
    _js_divergence,
    cov_key,
)
from yeastbench.registry import TASKS

INFO = BenchmarkInfo(name="test_brooks", version="test", description="t",
                     distribution_uri="")

# Mock model geometry. The stored slice (SLICE bp) is the synthetic contig, so a
# centred gene's window is clamped within it exactly like the real data.
SEQ_LEN = 300
CROP = 60
OUT_LEN = SEQ_LEN - 2 * CROP        # 180
SLICE = 400                         # stored gene-centred slice == whole contig
CDS_START, CDS_END = 150, 250       # 1-based, centred in the slice


def _rand_seq(rng: np.random.Generator, n: int = SLICE) -> str:
    return "".join(rng.choice(list("ACGT"), size=n))


def _write_artifact(data_dir: Path, constructs: list[dict]) -> None:
    """Write the 3-file artifact from a list of construct specs. Each spec needs
    the index columns plus ``alt_seq``/``native_seq`` (SLICE bp) and
    ``alt_cov``/``native_cov`` (SLICE int arrays)."""
    fasta: dict[str, str] = {}
    cov: dict[str, np.ndarray] = {}
    rows: list[dict] = []
    for c in constructs:
        ak = cov_key("alt", c["sample_id"])
        nk = cov_key("native", c["gene_id"])
        fasta[ak] = c["alt_seq"]
        cov[ak] = c["alt_cov"].astype(np.int32)
        fasta.setdefault(nk, c["native_seq"])
        cov.setdefault(nk, c["native_cov"].astype(np.int32))
        rows.append({k: v for k, v in c.items()
                     if k not in ("alt_seq", "native_seq", "alt_cov", "native_cov")})
    pd.DataFrame(rows).to_csv(data_dir / "brooks_index.tsv", sep="\t", index=False)
    with open(data_dir / "brooks_constructs.fasta", "w") as fh:
        for k in sorted(fasta):
            fh.write(f">{k}\n{fasta[k]}\n")
    np.savez_compressed(data_dir / "brooks_cov.npz", **cov)


def _construct(rng, sample_id, gene_id, strain, *, true_lfc, copy_idx=0,
               n_copies=1, low_support=False, raw_runs="30,40,50",
               alt_seq=None, native_seq=None):
    """One construct spec with scalars set so the benchmark recomputes
    ``true_lfc`` ≈ the requested value (norm_cov_js94 mean ≈ 10)."""
    s_norm = float(2.0 ** true_lfc * 11.0 - 1.0)   # log2((s+1)/(10+1)) == true_lfc
    norm_runs = ",".join(f"{v:.3f}" for v in rng.uniform(9.8, 10.2, 3))
    return {
        "sample_id": sample_id, "gene_id": gene_id, "strain": strain,
        "copy_idx": copy_idx, "n_copies": n_copies, "strand": "+",
        "rearr_class": "context_change",
        "alt_source_genome": f"genomes/{strain}_ERCC92.fasta",
        "alt_contig": f"{strain}_1", "alt_contig_len": SLICE,
        "alt_cds_start": CDS_START, "alt_cds_end": CDS_END, "alt_slice_start": 0,
        "native_source_genome": "genomes/JS96_ERCC92.fasta",
        "native_contig": "JS96_1", "native_contig_len": SLICE,
        "native_cds_start": CDS_START, "native_cds_end": CDS_END,
        "native_slice_start": 0,
        "strain_reads": 250, "js94_reads_runs": raw_runs,
        "size_factor_strain": 4.0,
        "norm_cov_strain": round(s_norm, 3),
        "norm_cov_js94_mean": 10.0, "norm_cov_js94_runs": norm_runs,
        "true_lfc": round(true_lfc, 4), "low_support": low_support,
        "alt_seq": alt_seq if alt_seq is not None else _rand_seq(rng),
        "native_seq": native_seq if native_seq is not None else _rand_seq(rng),
        "alt_cov": rng.poisson(0.5, SLICE), "native_cov": rng.poisson(0.5, SLICE),
    }


@pytest.fixture
def brooks_dir(tmp_path: Path) -> tuple[Path, dict[str, float]]:
    """6 single-copy genes, alternating ±1.5 true LFC; all 3 JS94 reps supported.
    Returns (data_dir, {sample_id: true_lfc})."""
    rng = np.random.default_rng(0)
    constructs, lfcs = [], {}
    for i in range(6):
        sid = f"JS60{i}:YIR0{i:02d}W:0"
        tl = 1.5 if i % 2 == 0 else -1.5
        lfcs[sid] = tl
        constructs.append(_construct(rng, sid, f"YIR0{i:02d}W", f"JS60{i}", true_lfc=tl))
    _write_artifact(tmp_path, constructs)
    return tmp_path, lfcs


class _MockAdapter:
    """Per-base batched predictor: a windowed alt sequence in ``scale`` returns a
    near-constant vector times ``2**true_lfc``; everything else (natives) ≈ 1.0.
    Already untransformed/unbinned, per the protocol contract."""
    seq_len = SEQ_LEN
    crop_bp_each_side = CROP
    batch_size = 4
    varies_by_strain = True

    def __init__(self, scale: dict[str, float], seed: int = 0):
        self._scale = scale
        self._rng = np.random.default_rng(seed)

    def predict_coverage_batch(self, seqs, strands, strains=None):
        out = np.empty((len(seqs), OUT_LEN), dtype=np.float64)
        for i, s in enumerate(seqs):
            base = self._rng.normal(1.0, 0.05, OUT_LEN).clip(0.1)
            out[i] = base * self._scale.get(s, 1.0)
        return out


class _MockBroadcastAdapter(_MockAdapter):
    varies_by_strain = False


def _mock(b, lfcs, *, broadcast=False):
    """Build a mock whose alt predictions track each construct's true LFC, keyed
    by the *windowed* alt sequence the benchmark will feed it."""
    mat = b._materialize(SEQ_LEN)
    scale = {row.alt_seq: 2.0 ** lfcs[row.sample_id] for _, row in mat.iterrows()}
    cls = _MockBroadcastAdapter if broadcast else _MockAdapter
    return cls(scale)


assert isinstance(_MockAdapter({}), CoverageTrackPredictor)


# ── helpers ───────────────────────────────────────────────────


class TestHelpers:
    def test_js_divergence_symmetric_and_zero(self):
        p = np.array([0.2, 0.3, 0.5])
        q = np.array([0.5, 0.3, 0.2])
        assert _js_divergence(p, p) < 1e-9
        assert abs(_js_divergence(p, q) - _js_divergence(q, p)) < 1e-12
        assert 0 <= _js_divergence(p, q) <= 1.0


# ── benchmark ─────────────────────────────────────────────────


class TestBrooksBenchmark:
    def test_init_loads(self, brooks_dir):
        d, _ = brooks_dir
        b = BrooksScrambleBenchmark(d, INFO)
        assert len(b.index) == 6
        mat = b._materialize(SEQ_LEN)
        assert len(mat) == 6
        assert (mat.alt_seq.str.len() == SEQ_LEN).all()
        assert (mat.native_seq.str.len() == SEQ_LEN).all()

    def test_init_requires_artifact(self, tmp_path):
        with pytest.raises(AssertionError):
            BrooksScrambleBenchmark(tmp_path, INFO)   # empty dir, no index

    def test_evaluate_perfect_predictor(self, brooks_dir):
        d, lfcs = brooks_dir
        b = BrooksScrambleBenchmark(d, INFO)
        res = b.evaluate(_mock(b, lfcs))
        assert res.n_total == 6 and res.n_scored == 6
        assert res.n_calibration == 6 and res.n_weak_baseline == 0
        assert np.all(res.dir_balanced_acc_per_rep > 0.99)
        assert np.all(res.pearson_r_per_rep > 0.95)
        assert (res.n_reps_supported == 3).all()
        # every sample backs every replicate, so each per-rep cohort is full
        assert np.all(res.n_scored_per_rep == 6)
        assert np.all(res.n_ceiling_per_rep == 6)

    def test_save_load_roundtrip(self, brooks_dir, tmp_path):
        d, lfcs = brooks_dir
        b = BrooksScrambleBenchmark(d, INFO)
        res = b.evaluate(_mock(b, lfcs))
        out = tmp_path / "out"
        b.save_results(res, out)
        loaded = b.load_results(out)
        np.testing.assert_array_almost_equal(loaded.pred_lfc_runs, res.pred_lfc_runs)
        np.testing.assert_array_almost_equal(loaded.true_lfc_runs, res.true_lfc_runs)
        np.testing.assert_array_equal(loaded.n_reps_supported, res.n_reps_supported)
        assert loaded.sample_ids == res.sample_ids
        assert loaded.n_scored == res.n_scored
        np.testing.assert_array_almost_equal(loaded.pearson_r_per_rep,
                                             res.pearson_r_per_rep)
        np.testing.assert_array_almost_equal(loaded.ceiling_r_per_rep,
                                             res.ceiling_r_per_rep)
        np.testing.assert_array_almost_equal(loaded.ceiling_rho_per_rep,
                                             res.ceiling_rho_per_rep)
        np.testing.assert_array_equal(loaded.n_scored_per_rep,
                                      res.n_scored_per_rep)
        np.testing.assert_array_equal(loaded.n_ceiling_per_rep,
                                      res.n_ceiling_per_rep)

    def test_plot_and_summary_and_headline(self, brooks_dir, tmp_path):
        d, lfcs = brooks_dir
        b = BrooksScrambleBenchmark(d, INFO)
        res = b.evaluate(_mock(b, lfcs))
        b.plot(res, tmp_path / "p")
        assert (tmp_path / "p" / "lfc_scatter.png").exists()
        assert (tmp_path / "p" / "lfc_per_sample.png").exists()
        s = b.summary_dict(res)
        for k in ("n_total", "n_scored", "n_calibration", "n_weak_baseline",
                  "lfc_n_scored_per_rep", "lfc_pearson_r_per_rep",
                  "lfc_spearman_rho_per_rep", "lfc_dir_balanced_acc_per_rep",
                  "lfc_n_ceiling_per_rep", "lfc_ceiling_r_per_rep",
                  "lfc_ceiling_rho_per_rep", "lfc_ceiling_dir_acc_per_rep",
                  "lfc_within_range_rate", "lfc_mean_abs_z",
                  "shape_pearson_mean", "shape_js_mean"):
            assert k in s
        for k in ("lfc_n_scored_per_rep", "lfc_pearson_r_per_rep",
                  "lfc_spearman_rho_per_rep", "lfc_dir_balanced_acc_per_rep",
                  "lfc_n_ceiling_per_rep", "lfc_ceiling_r_per_rep",
                  "lfc_ceiling_rho_per_rep", "lfc_ceiling_dir_acc_per_rep"):
            assert len(s[k]) == 3
        # No averaged-across-replicate scalar may reappear: those were the
        # numbers that silently weighted unequal cohorts equally.
        for k in ("lfc_pearson_r", "lfc_spearman_rho", "lfc_dir_balanced_acc",
                  "lfc_ceiling_pearson_r", "lfc_ceiling_dir_balanced_acc"):
            assert k not in s
        # ...but each replicate is also emitted as a named scalar, so the
        # cross-task summary.csv/summary.md (scalars only) still see Brooks.
        for alias in ("JS94_r0", "JS94_r1", "JS94_r2"):
            assert isinstance(s[f"lfc_pearson_r_{alias}"], float)
            assert isinstance(s[f"lfc_n_scored_{alias}"], int)
        h = b.headline(res)
        assert "LFC" in h and "ceiling" in h and "shape" in h
        # one line per replicate, each naming its cohort size
        for alias in ("JS94_r0", "JS94_r1", "JS94_r2"):
            assert alias in h

    def test_low_support_dropped(self, brooks_dir):
        d, lfcs = brooks_dir
        idx = pd.read_csv(d / "brooks_index.tsv", sep="\t")
        idx.loc[0, "low_support"] = True
        idx.to_csv(d / "brooks_index.tsv", sep="\t", index=False)
        b = BrooksScrambleBenchmark(d, INFO)
        res = b.evaluate(_mock(b, lfcs))
        assert res.n_total == 6 and res.n_scored == 5 and res.n_calibration == 5

    def test_weak_baseline_excluded_from_scored(self, brooks_dir):
        d, lfcs = brooks_dir
        idx = pd.read_csv(d / "brooks_index.tsv", sep="\t")
        idx.loc[0, "js94_reads_runs"] = "0,0,0"
        idx.to_csv(d / "brooks_index.tsv", sep="\t", index=False)
        b = BrooksScrambleBenchmark(d, INFO)
        res = b.evaluate(_mock(b, lfcs))
        assert res.n_scored == 5 and res.n_calibration == 5 and res.n_weak_baseline == 1

    def test_ceiling_is_high_when_replicates_agree(self, brooks_dir):
        d, lfcs = brooks_dir
        b = BrooksScrambleBenchmark(d, INFO)
        res = b.evaluate(_mock(b, lfcs))
        assert np.all(res.ceiling_r_per_rep > 0.99)
        assert np.all(res.ceiling_dir_acc_per_rep > 0.99)
        assert np.all(np.isfinite(res.ceiling_r_per_rep))
        assert np.all(np.isfinite(res.ceiling_rho_per_rep))

    def test_broadcast_adapter_yields_identical_pred_columns(self, brooks_dir):
        d, lfcs = brooks_dir
        b = BrooksScrambleBenchmark(d, INFO)
        res = b.evaluate(_mock(b, lfcs, broadcast=True))
        finite = np.isfinite(res.true_lfc_runs)
        for i in np.where(finite.all(axis=1))[0]:
            np.testing.assert_array_almost_equal(
                res.pred_lfc_runs[i, 0] * np.ones(3), res.pred_lfc_runs[i])
        assert np.all(np.isfinite(res.pearson_r_per_rep))

    def test_partial_replicate_support(self, brooks_dir):
        d, lfcs = brooks_dir
        idx = pd.read_csv(d / "brooks_index.tsv", sep="\t")
        idx.loc[0, "js94_reads_runs"] = "0,30,40"      # n_reps == 2
        idx.loc[1, "js94_reads_runs"] = "0,0,40"       # n_reps == 1
        idx.to_csv(d / "brooks_index.tsv", sep="\t", index=False)
        b = BrooksScrambleBenchmark(d, INFO)
        res = b.evaluate(_mock(b, lfcs))
        assert res.n_scored == 6 and res.n_calibration == 5
        assert res.n_reps_supported[0] == 2 and res.n_reps_supported[1] == 1
        # The replicates now cover different cohorts, which is exactly why the
        # per-rep numbers are reported separately rather than averaged: r0
        # loses both edited samples, r1 loses one, r2 keeps all six.
        np.testing.assert_array_equal(res.n_scored_per_rep, [4, 5, 6])


class TestWindowDependentMembership:
    def test_dedup_depends_on_window(self, tmp_path):
        """Two copies of a gene that are byte-identical within a small window but
        differ in the flank: the small window dedups them to 1, a window that
        reaches the flank keeps both — reproducing the real 698 vs 1055 effect."""
        rng = np.random.default_rng(1)
        native = _rand_seq(rng)
        core = _rand_seq(rng)                       # shared central region
        # central 300 bp [49:349] identical; flanks differ.
        alt0 = _rand_seq(rng, 49) + core[49:349] + _rand_seq(rng, SLICE - 349)
        alt1 = _rand_seq(rng, 49) + core[49:349] + _rand_seq(rng, SLICE - 349)
        assert alt0[49:349] == alt1[49:349] and alt0 != alt1
        cons = [
            _construct(rng, "JSX:YG:0", "YG", "JSX", true_lfc=1.0, copy_idx=0,
                       n_copies=2, alt_seq=alt0, native_seq=native),
            _construct(rng, "JSX:YG:1", "YG", "JSX", true_lfc=1.0, copy_idx=1,
                       n_copies=2, alt_seq=alt1, native_seq=native),
        ]
        _write_artifact(tmp_path, cons)
        b = BrooksScrambleBenchmark(tmp_path, INFO)
        assert len(b._materialize(300)) == 1     # identical in-window → deduped
        assert len(b._materialize(400)) == 2     # flank differs → both kept

    def test_alt_equals_native_dropped(self, tmp_path):
        rng = np.random.default_rng(2)
        same = _rand_seq(rng)
        cons = [_construct(rng, "JSY:YG:0", "YG", "JSY", true_lfc=0.0,
                           alt_seq=same, native_seq=same)]
        _write_artifact(tmp_path, cons)
        b = BrooksScrambleBenchmark(tmp_path, INFO)
        assert len(b._materialize(300)) == 0     # no cis change in-window

    def test_evaluate_empty_cohort(self, tmp_path):
        # _materialize → 0 rows must give an empty cohort, not an AttributeError
        # on the column-less DataFrame.
        rng = np.random.default_rng(3)
        same = _rand_seq(rng)
        cons = [_construct(rng, "JSZ:YG:0", "YG", "JSZ", true_lfc=0.0,
                           alt_seq=same, native_seq=same)]
        _write_artifact(tmp_path, cons)
        b = BrooksScrambleBenchmark(tmp_path, INFO)
        res = b.evaluate(_MockAdapter({}))
        assert res.n_total == 0 and res.n_scored == 0
        assert res.sample_ids == []
        assert np.isnan(res.shape_pearson_mean)


# ── registry ──────────────────────────────────────────────────


class TestBrooksRegistry:
    def test_single_task_registered(self):
        assert "brooks_scramble" in TASKS
        assert "brooks_scramble_shorkie" not in TASKS

    def test_factory_builds_benchmark(self, brooks_dir):
        d, _ = brooks_dir
        task = TASKS["brooks_scramble"](data_path=d)
        assert isinstance(task, BrooksScrambleBenchmark)
        assert task.adapter_protocol is CoverageTrackPredictor


# ── bit-exactness regression guard (real shipped data) ────────
#
# The benchmark's run-time windowing + membership/dedup must reproduce the
# original pre-cut per-window distributions. Digests are over the materialized
# constructs at each model window; captured from the legacy TSVs before removal.
# Skipped where the shipped data isn't present (e.g. CI without a data fetch).

_REAL = Path("data/tasks/brooks_scramble")
_GOLDEN = {
    4992: (698, "cbe2c75132bb8112c506356a", "577c673d74b359743c218511"),
    16384: (1055, "88498124d51a60b6ae5b5eda", "260e8f3d3e843192b8c475dc"),
}


@pytest.mark.parametrize("window", sorted(_GOLDEN))
def test_materialize_matches_shipped_distribution(window):
    if not (_REAL / "brooks_index.tsv").exists():
        pytest.skip("shipped Brooks data not present (not fetched)")
    n_expected, mem_expected, content_expected = _GOLDEN[window]
    b = BrooksScrambleBenchmark(_REAL, INFO)
    mat = b._materialize(window)
    assert len(mat) == n_expected
    sids = sorted(mat.sample_id)
    assert hashlib.sha256("".join(sids).encode()).hexdigest()[:24] == mem_expected
    by = mat.set_index("sample_id")
    h = hashlib.sha256()
    for sid in sids:
        r = by.loc[sid]
        h.update(sid.encode())
        h.update(r.alt_seq.encode())
        h.update(r.native_seq.encode())
        h.update(f"{int(r.cds_start_in_window)},{int(r.cds_end_in_window)}".encode())
        h.update(np.asarray(r.true_cov_alt, dtype=np.int32).tobytes())
    assert h.hexdigest()[:24] == content_expected
