"""Zero-shot GFP RNA abundance across Gui et al. replacement genotypes.

The label is mean deposited GFP TPM per genotype. Only verified edited
sequences reach an adapter; unresolved rows and failed predictions remain in
the exported table. Scores are RNA coverage readouts, not calibrated TPM.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Iterable, Mapping, Protocol, cast

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata, spearmanr

from yeastbench.adapters._gui_scaffold import GuiLocus
from yeastbench.adapters.protocols import GuiReporterExpressionPredictor
from yeastbench.benchmarks.base import Benchmark, BenchmarkInfo, model_color


class _GuiLocusRow(Protocol):
    """Validated, sequence-eligible pandas row, used only in this module."""

    @property
    def locus_id(self) -> str: ...  # e.g. "YNL075W"
    @property
    def chrom(self) -> str: ...  # e.g. "XIV"
    @property
    def deletion_start(self) -> int: ...  # 1-based inclusive
    @property
    def deletion_end(self) -> int: ...  # 1-based inclusive
    @property
    def cassette_strand(self) -> str: ...  # "+" or "-"
    @property
    def construct_id(self) -> str: ...  # FASTA record identifier
    @property
    def gfp_cds_start(self) -> int: ...  # 0-based payload offset
    @property
    def gfp_cds_end(self) -> int: ...  # exclusive payload offset


@dataclass(frozen=True)
class GuiGFPResults:
    predictions: pd.DataFrame  # All input genotypes, including excluded rows.
    summary: dict[str, Any]  # JSON-safe metrics, exclusions and input hashes.


_GEOMETRY_COLUMNS = (
    "locus_id", "chrom", "deletion_start", "deletion_end", "cassette_strand",
    "construct_id", "gfp_cds_start", "gfp_cds_end",
)
_REQUIRED_COLUMNS = (
    *_GEOMETRY_COLUMNS, "gfp_tpm", "n_libraries", "library_accessions",
    "sequence_status", "exclusion_reason",
)
_METRIC_NAMES = ("spearman_rho", "pearson_r", "pearson_log2_r")
_CONTEXT_METADATA_COLUMNS = (
    "window_sha256", "window_length", "gfp_start_in_window", "gfp_end_in_window",
    "gfp_readout_bases", "native_upstream_bp", "native_downstream_bp",
)


def _compute_correlation(predicted: np.ndarray, measured: np.ndarray, *, rank: bool = False) -> float | None:
    if len(predicted) < 2 or np.ptp(predicted) == 0 or np.ptp(measured) == 0:
        return None
    coefficient = spearmanr(predicted, measured).statistic if rank else pearsonr(predicted, measured).statistic
    return float(coefficient) if np.isfinite(coefficient) else None


def _compute_bootstrap_correlations(predicted: np.ndarray, measured: np.ndarray, *, rank: bool) -> np.ndarray:
    """One correlation per paired locus-bootstrap draw, with ties preserved."""
    if rank:
        predicted = rankdata(predicted, axis=1)
        measured = rankdata(measured, axis=1)
    centered_predictions = predicted - predicted.mean(axis=1, keepdims=True)
    centered_measurements = measured - measured.mean(axis=1, keepdims=True)
    denominator = np.sqrt(
        np.sum(centered_predictions**2, axis=1)
        * np.sum(centered_measurements**2, axis=1)
    )
    correlations = np.full(len(predicted), np.nan)
    np.divide(
        np.sum(centered_predictions * centered_measurements, axis=1),
        denominator, out=correlations, where=denominator > 0,
    )
    return np.clip(correlations, -1, 1)


def compute_gui_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    bootstrap_samples: int = 2000,
    bootstrap_seed: int = 57,
) -> dict[str, Any]:
    """Correlations with paired-locus percentile intervals, conditional on labels.

    Log Pearson uses strictly positive scores and labels without an arbitrary
    pseudocount. Its sample count is reported independently. A constant score
    has undefined correlation (JSON null), never a fabricated zero.
    """
    if bootstrap_samples < 0:
        raise ValueError("bootstrap_samples must be nonnegative")
    scores, labels = np.asarray(scores, dtype=float), np.asarray(labels, dtype=float)
    if scores.ndim != 1 or scores.shape != labels.shape:
        raise ValueError("Scores and labels must be aligned one-dimensional arrays")
    available = np.isfinite(scores) & np.isfinite(labels)
    predicted, measured = scores[available], labels[available]
    positive = (predicted > 0) & (measured > 0)
    log_predictions, log_measurements = np.log2(predicted[positive]), np.log2(measured[positive])
    metrics: dict[str, Any] = {
        "n_scored": int(available.sum()),
        "n_log2": int(positive.sum()),
        "spearman_rho": _compute_correlation(predicted, measured, rank=True),
        "pearson_r": _compute_correlation(predicted, measured),
        "pearson_log2_r": _compute_correlation(log_predictions, log_measurements),
        "confidence_intervals_95": {},
        "bootstrap_valid_draws": {},
    }
    generator = np.random.default_rng(bootstrap_seed)
    for metric, predictions, measurements, ranked in (
        ("spearman_rho", predicted, measured, True),
        ("pearson_r", predicted, measured, False),
        ("pearson_log2_r", log_predictions, log_measurements, False),
    ):
        coefficients = np.array([], dtype=float)
        if metrics[metric] is not None and bootstrap_samples:
            indices = generator.integers(len(predictions), size=(bootstrap_samples, len(predictions)))
            coefficients = _compute_bootstrap_correlations(predictions[indices], measurements[indices], rank=ranked)
            coefficients = coefficients[np.isfinite(coefficients)]
        metrics["bootstrap_valid_draws"][metric] = len(coefficients)
        metrics["confidence_intervals_95"][metric] = (
            np.quantile(coefficients, [.025, .975]).tolist() if len(coefficients) else None
        )
    return metrics


def _compute_file_hash(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


class GuiGFPRNABenchmark(Benchmark[GuiReporterExpressionPredictor, GuiGFPResults]):
    adapter_protocol: ClassVar[type] = GuiReporterExpressionPredictor

    def __init__(
        self,
        distribution_dir: Path,
        fasta_path: Path,
        info: BenchmarkInfo,
        bootstrap_samples: int = 2000,
        bootstrap_seed: int = 57,
    ) -> None:
        if bootstrap_samples < 0:
            raise ValueError("bootstrap_samples must be nonnegative")
        self.distribution_dir = Path(distribution_dir)
        self.labels_path = self.distribution_dir / "gui_gfp_rna_v1.tsv"
        self.constructs_path = self.distribution_dir / "constructs.fasta"
        self._fasta_path = Path(fasta_path)
        self.info = info
        self.bootstrap_samples = bootstrap_samples
        self.bootstrap_seed = bootstrap_seed
        self._table = pd.read_csv(self.labels_path, sep="\t", keep_default_na=False)
        missing = set(_REQUIRED_COLUMNS) - set(self._table)
        if missing:
            raise ValueError(f"Missing Gui input columns: {sorted(missing)}")
        if self._table.empty or self._table.locus_id.eq("").any() or self._table.locus_id.duplicated().any():
            raise ValueError("Gui labels must contain unique, nonempty locus IDs")
        self._table["gfp_tpm"] = pd.to_numeric(self._table.gfp_tpm, errors="raise")
        if not np.isfinite(self._table.gfp_tpm).all() or (self._table.gfp_tpm <= 0).any():
            raise ValueError("GFP TPM labels must be finite and positive")
        library_counts = pd.to_numeric(self._table.n_libraries, errors="raise")
        if not np.isfinite(library_counts).all() or (library_counts < 1).any() or (library_counts % 1 != 0).any():
            raise ValueError("n_libraries must be positive integers")
        self._table["n_libraries"] = library_counts.astype("int64")
        unknown_statuses = set(self._table.sequence_status) - {"verified", "unresolved", "excluded"}
        if unknown_statuses:
            raise ValueError(f"Unknown sequence_status values: {sorted(unknown_statuses)}")
        self._eligible = self._table.sequence_status.eq("verified").to_numpy()
        if not self._eligible.any():
            raise ValueError("No sequence-verified Gui genotypes are available; resolve constructs before inference")
        if self._table.loc[self._eligible, "exclusion_reason"].ne("").any():
            raise ValueError("Verified rows cannot carry a sequence exclusion reason")
        verified = self._table.loc[self._eligible, list(_GEOMETRY_COLUMNS)].copy()
        for column in ("deletion_start", "deletion_end", "gfp_cds_start", "gfp_cds_end"):
            coordinates = pd.to_numeric(verified[column], errors="raise")
            if not np.isfinite(coordinates).all() or (coordinates % 1 != 0).any():
                raise ValueError(f"{column} must contain integer coordinates for verified rows")
            verified[column] = coordinates.astype("int64")
        self.loci = tuple(
            GuiLocus(
                locus_id=row.locus_id, chrom=row.chrom,
                deletion_start=row.deletion_start, deletion_end=row.deletion_end,
                cassette_strand=row.cassette_strand, construct_id=row.construct_id,
                gfp_cds_start=row.gfp_cds_start, gfp_cds_end=row.gfp_cds_end,
            )
            for row in cast(Iterable[_GuiLocusRow], verified.itertuples(index=False))
        )

    @property
    def fasta_path(self) -> Path:
        return self._fasta_path

    def evaluate(self, adapter: GuiReporterExpressionPredictor) -> GuiGFPResults:
        scores = np.asarray(adapter.predict_expressions(self.loci), dtype=float)
        if scores.shape != (len(self.loci),):
            raise ValueError(f"Adapter returned shape {scores.shape}; expected {(len(self.loci),)}")
        if np.any(scores < 0):
            raise ValueError("GFP expression scores must be nonnegative linear RNA readouts")
        predictions = self._table.copy()
        predictions["prediction"] = np.nan
        predictions["evaluation_status"] = "unresolved_sequence"
        missing_reasons = predictions.exclusion_reason.eq("") & ~self._eligible
        predictions.loc[missing_reasons, "exclusion_reason"] = "sequence_not_verified"
        predictions.loc[self._eligible, "prediction"] = scores
        predictions.loc[self._eligible, "evaluation_status"] = "scored"
        prediction_exclusions = getattr(adapter, "prediction_exclusions", {})
        unavailable = self._eligible & ~np.isfinite(predictions.prediction.to_numpy())
        predictions.loc[unavailable, "prediction"] = np.nan
        predictions.loc[unavailable, "evaluation_status"] = "prediction_unavailable"
        predictions.loc[unavailable, "exclusion_reason"] = [
            prediction_exclusions.get(locus_id, "nonfinite_prediction")
            for locus_id in predictions.loc[unavailable, "locus_id"]
        ]
        context_metadata = getattr(adapter, "prediction_metadata", {})
        for column in _CONTEXT_METADATA_COLUMNS:
            predictions[column] = [
                context_metadata.get(locus_id, {}).get(column)
                for locus_id in predictions.locus_id
            ]
        metrics = compute_gui_metrics(
            predictions.prediction.to_numpy(), predictions.gfp_tpm.to_numpy(),
            bootstrap_samples=self.bootstrap_samples, bootstrap_seed=self.bootstrap_seed,
        )
        without_imp4 = predictions.loc[predictions.locus_id != "YNL075W"]
        summary = {
            **metrics,
            "n_rows_total": len(predictions),
            "n_sequence_verified": int(self._eligible.sum()),
            "n_excluded": int(predictions.evaluation_status.ne("scored").sum()),
            "exclusion_counts": predictions.loc[predictions.evaluation_status.ne("scored"), "exclusion_reason"].value_counts().to_dict(),
            "primary_metric": "spearman_rho",
            "label": "arithmetic mean GFP TPM per genotype",
            "readout": "linear RNA coverage summed over complete GFP CDS",
            "without_imp4": compute_gui_metrics(
                without_imp4.prediction.to_numpy(), without_imp4.gfp_tpm.to_numpy(),
                bootstrap_samples=self.bootstrap_samples, bootstrap_seed=self.bootstrap_seed,
            ),
            "bootstrap": {
                "unit": "integration genotype", "samples": self.bootstrap_samples,
                "seed": self.bootstrap_seed, "method": "paired percentile",
                "interpretation": "Conditional on measured labels; does not estimate RNA-seq measurement error",
            },
            "input_sha256": {
                "labels": _compute_file_hash(self.labels_path),
                "constructs": _compute_file_hash(self.constructs_path),
                "reference_fasta": _compute_file_hash(self.fasta_path),
            },
        }
        return GuiGFPResults(predictions=predictions, summary=summary)

    def save_results(self, results: GuiGFPResults, out_dir: Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        results.predictions.to_csv(out_dir / "predictions.tsv", sep="\t", index=False)
        (out_dir / "summary.json").write_text(json.dumps(results.summary, indent=2, allow_nan=False) + "\n")

    def load_results(self, out_dir: Path) -> GuiGFPResults:
        out_dir = Path(out_dir)
        predictions = pd.read_csv(out_dir / "predictions.tsv", sep="\t", keep_default_na=False)
        predictions["prediction"] = pd.to_numeric(predictions.prediction, errors="coerce")
        summary = json.loads((out_dir / "summary.json").read_text())
        return GuiGFPResults(predictions=predictions, summary=summary)

    def summary_dict(self, results: GuiGFPResults) -> dict[str, Any]:
        return dict(results.summary)

    def headline(self, results: GuiGFPResults) -> str:
        rho = results.summary["spearman_rho"]
        coefficient = f"{rho:.4f}" if rho is not None else "undefined"
        return f"GFP RNA Spearman ρ = {coefficient} (n = {results.summary['n_scored']}/{results.summary['n_rows_total']})"

    def headline_metric_labels(self) -> dict[str, str]:
        return {"spearman_rho": "Spearman ρ", "pearson_r": "Pearson r", "pearson_log2_r": "Log₂ Pearson r"}

    def plot(self, results: GuiGFPResults, out_dir: Path) -> None:
        import matplotlib.pyplot as plt

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        scored = results.predictions.loc[results.predictions.evaluation_status == "scored"]
        for scale in ("linear", "log"):
            panel = scored if scale == "linear" else scored.loc[scored.prediction > 0]
            figure, axis = plt.subplots(figsize=(6.5, 5.5), layout="constrained")
            axis.scatter(panel.gfp_tpm, panel.prediction, color="#397b99", alpha=.7, s=24)
            imp4 = panel.loc[panel.locus_id == "YNL075W"]
            if not imp4.empty:
                axis.scatter(imp4.gfp_tpm, imp4.prediction, color="#cf4054", s=34, label="IMP4")
                axis.legend(frameon=False)
            axis.set(xlabel="Measured GFP TPM", ylabel="Predicted GFP RNA score",
                     title=f"Gui GFP RNA · {out_dir.name.split('__')[0]} · n = {len(panel)}")
            if scale == "log" and not panel.empty:
                axis.set(xscale="log", yscale="log")
            for extension in ("png", "svg"):
                figure.savefig(out_dir / f"scatter_{scale}.{extension}", dpi=160)
            plt.close(figure)

    def compare_plot(self, model_dirs: Mapping[str, Path], out_dir: Path) -> Path | None:
        """Compare models on the exact intersection of successfully scored loci."""
        import matplotlib.pyplot as plt

        saved_results = {
            name: self.load_results(directory)
            for name, directory in sorted(model_dirs.items())
        }
        if len(saved_results) < 2:
            return None
        required_hashes = {"labels", "constructs", "reference_fasta"}
        reference_hashes = None
        for name, saved in saved_results.items():
            hashes = saved.summary.get("input_sha256")
            if not isinstance(hashes, dict) or set(hashes) != required_hashes or not all(hashes.values()):
                raise ValueError(f"{name}: missing Gui input provenance; rerun before comparing models")
            if reference_hashes is None:
                reference_hashes = hashes
            elif hashes != reference_hashes:
                raise ValueError(
                    f"{name}: incompatible Gui input hashes; rerun models on the same distribution and reference"
                )
        scored_by_model = {
            name: saved.predictions.set_index("locus_id").query("evaluation_status == 'scored'")
            for name, saved in saved_results.items()
        }
        shared_ids = sorted(set.intersection(*(set(frame.index) for frame in scored_by_model.values())))
        if not shared_ids:
            return None
        reference_labels = next(iter(scored_by_model.values())).loc[shared_ids, "gfp_tpm"].to_numpy()
        common_metrics = {}
        for name, frame in scored_by_model.items():
            labels = frame.loc[shared_ids, "gfp_tpm"].to_numpy()
            if not np.array_equal(labels, reference_labels):
                raise ValueError("Models have inconsistent GFP labels for the shared cohort")
            common_metrics[name] = compute_gui_metrics(frame.loc[shared_ids, "prediction"].to_numpy(), labels, bootstrap_samples=0)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "common_cohort_metrics.json").write_text(json.dumps({
            "locus_ids": shared_ids, "metrics": common_metrics,
            "original_n_scored": {name: len(frame) for name, frame in scored_by_model.items()},
            "input_sha256": reference_hashes,
        }, indent=2, allow_nan=False) + "\n")
        figure, axis = plt.subplots(figsize=(7, 5), layout="constrained")
        positions = np.arange(len(_METRIC_NAMES))
        width = .8 / len(common_metrics)
        for model_index, (name, metrics) in enumerate(common_metrics.items()):
            heights = [metrics[metric] if metrics[metric] is not None else np.nan for metric in _METRIC_NAMES]
            axis.bar(positions - .4 + width / 2 + model_index * width, heights,
                     width=width, color=model_color(name, list(common_metrics)), label=name)
        axis.axhline(0, color="#777", linewidth=.7)
        axis.set(xticks=positions, xticklabels=list(self.headline_metric_labels().values()),
                 ylabel="Correlation", ylim=(-1, 1), title=f"Gui GFP RNA · shared cohort n = {len(shared_ids)}")
        axis.legend(frameon=False)
        destination = out_dir / "gui_gfp_rna_comparison.png"
        figure.savefig(destination, dpi=160)
        plt.close(figure)
        return destination
