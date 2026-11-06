"""Lee et al. YTK constitutive-promoter range benchmark."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, ClassVar, Mapping

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from yeastbench.adapters._ytk_scaffold import YTKConstruct, load_constructs
from yeastbench.adapters.protocols import IntegratedPromoterPanelPredictor
from yeastbench.benchmarks.base import Benchmark, BenchmarkInfo, model_color


REPORTERS = ("mRuby2", "Venus")


@dataclass(frozen=True)
class RangeMetrics:
    n: int
    observed_fold_range: float
    predicted_fold_range: float
    observed_log10_span: float
    predicted_log10_span: float
    dynamic_range_recovery: float
    dynamic_range_fidelity: float
    span_error_decades: float
    pearson_log10: float
    spearman_rho: float


@dataclass(frozen=True)
class YTKResults:
    scores: np.ndarray
    labels: np.ndarray
    label_min: np.ndarray
    label_max: np.ndarray
    construct_ids: list[str]
    promoters: list[str]
    reporters: list[str]
    metrics: dict[str, RangeMetrics]


def _range_metrics(scores: np.ndarray, labels: np.ndarray) -> RangeMetrics:
    finite = np.isfinite(scores) & np.isfinite(labels)
    predicted = np.asarray(scores[finite], dtype=float)
    observed = np.asarray(labels[finite], dtype=float)
    n = len(predicted)
    empty = RangeMetrics(
        n=n,
        observed_fold_range=float("nan"),
        predicted_fold_range=float("nan"),
        observed_log10_span=float("nan"),
        predicted_log10_span=float("nan"),
        dynamic_range_recovery=float("nan"),
        dynamic_range_fidelity=float("nan"),
        span_error_decades=float("nan"),
        pearson_log10=float("nan"),
        spearman_rho=float("nan"),
    )
    if n < 2 or np.any(observed <= 0):
        return empty

    observed_fold = float(observed.max() / observed.min())
    observed_span = float(np.log10(observed_fold))
    if np.any(predicted <= 0):
        return RangeMetrics(
            **{
                **asdict(empty),
                "observed_fold_range": observed_fold,
                "observed_log10_span": observed_span,
                "dynamic_range_fidelity": 0.0,
            }
        )

    predicted_fold = float(predicted.max() / predicted.min())
    predicted_span = float(np.log10(predicted_fold))
    recovery = predicted_span / observed_span if observed_span > 0 else float("nan")
    if recovery == 0:
        fidelity = 0.0
    elif np.isfinite(recovery) and recovery > 0:
        fidelity = float(min(recovery, 1.0 / recovery))
    else:
        fidelity = float("nan")

    log_predicted = np.log10(predicted)
    log_observed = np.log10(observed)
    if np.unique(log_predicted).size < 2 or np.unique(log_observed).size < 2:
        pearson = spearman = float("nan")
    else:
        pearson = float(pearsonr(log_predicted, log_observed).statistic)
        spearman = float(spearmanr(log_predicted, log_observed).statistic)
    return RangeMetrics(
        n=n,
        observed_fold_range=observed_fold,
        predicted_fold_range=predicted_fold,
        observed_log10_span=observed_span,
        predicted_log10_span=predicted_span,
        dynamic_range_recovery=float(recovery),
        dynamic_range_fidelity=fidelity,
        span_error_decades=float(abs(predicted_span - observed_span)),
        pearson_log10=pearson,
        spearman_rho=spearman,
    )


def _geometric_mean(values: np.ndarray, axis: int) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.exp(np.mean(np.log(values), axis=axis))


def _metrics_by_panel(
    scores: np.ndarray,
    labels: np.ndarray,
    promoters: list[str],
    reporters: list[str],
) -> dict[str, RangeMetrics]:
    metrics: dict[str, RangeMetrics] = {}
    promoter_order = list(dict.fromkeys(promoters))
    score_matrix = np.full((len(promoter_order), len(REPORTERS)), np.nan)
    label_matrix = np.full_like(score_matrix, np.nan)
    row_by_pair = {
        (promoter, reporter): index
        for index, (promoter, reporter) in enumerate(zip(promoters, reporters))
    }
    for reporter_index, reporter in enumerate(REPORTERS):
        reporter_rows = [
            row_by_pair[(promoter, reporter)] for promoter in promoter_order
        ]
        metrics[reporter] = _range_metrics(scores[reporter_rows], labels[reporter_rows])
        score_matrix[:, reporter_index] = scores[reporter_rows]
        label_matrix[:, reporter_index] = labels[reporter_rows]

    metrics["consensus"] = _range_metrics(
        _geometric_mean(score_matrix, axis=1),
        _geometric_mean(label_matrix, axis=1),
    )
    return metrics


class YTKPromoterBenchmark(Benchmark[IntegratedPromoterPanelPredictor, YTKResults]):
    adapter_protocol: ClassVar[type] = IntegratedPromoterPanelPredictor

    def __init__(
        self,
        labels_path: Path,
        constructs_path: Path,
        constructs_fasta: Path,
        fasta_path: Path,
        info: BenchmarkInfo,
    ) -> None:
        self.labels_path = Path(labels_path)
        self.constructs_path = Path(constructs_path)
        self.constructs_fasta = Path(constructs_fasta)
        self._fasta_path = Path(fasta_path)
        self.info = info

        truth = pd.read_csv(self.labels_path, sep="\t")
        self.constructs: list[YTKConstruct] = load_constructs(
            self.constructs_path, self.constructs_fasta
        )
        truth_by_promoter = truth.set_index("promoter")
        self.construct_ids = [construct.construct_id for construct in self.constructs]
        self.promoters = [construct.promoter for construct in self.constructs]
        self.reporters = [construct.reporter for construct in self.constructs]
        self.labels = np.asarray(
            [
                truth_by_promoter.loc[
                    construct.promoter, f"{construct.reporter}_median_fold"
                ]
                for construct in self.constructs
            ],
            dtype=float,
        )
        self.label_min = np.asarray(
            [
                truth_by_promoter.loc[
                    construct.promoter, f"{construct.reporter}_min_fold"
                ]
                for construct in self.constructs
            ],
            dtype=float,
        )
        self.label_max = np.asarray(
            [
                truth_by_promoter.loc[
                    construct.promoter, f"{construct.reporter}_max_fold"
                ]
                for construct in self.constructs
            ],
            dtype=float,
        )

        expected_pairs = {
            (promoter, reporter)
            for promoter in truth["promoter"].astype(str)
            for reporter in REPORTERS
        }
        actual_pairs = set(zip(self.promoters, self.reporters))
        if actual_pairs != expected_pairs or len(actual_pairs) != len(self.constructs):
            raise ValueError(
                "constructs must contain one row per promoter and reporter"
            )

    @property
    def fasta_path(self) -> Path:
        return self._fasta_path

    def evaluate(self, adapter: IntegratedPromoterPanelPredictor) -> YTKResults:
        scores = np.asarray(
            adapter.predict_reporter_expressions(self.constructs), dtype=float
        )
        if len(scores) != len(self.constructs):
            raise ValueError(
                f"adapter returned {len(scores)} scores for "
                f"{len(self.constructs)} constructs"
            )
        return YTKResults(
            scores=scores,
            labels=self.labels.copy(),
            label_min=self.label_min.copy(),
            label_max=self.label_max.copy(),
            construct_ids=list(self.construct_ids),
            promoters=list(self.promoters),
            reporters=list(self.reporters),
            metrics=_metrics_by_panel(
                scores, self.labels, self.promoters, self.reporters
            ),
        )

    def plot(self, results: YTKResults, out_dir: Path) -> None:
        import matplotlib.pyplot as plt

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        promoter_order = list(dict.fromkeys(results.promoters))
        row_by_pair = {
            (promoter, reporter): index
            for index, (promoter, reporter) in enumerate(
                zip(results.promoters, results.reporters)
            )
        }

        def panel(values: np.ndarray, reporter: str) -> np.ndarray:
            return np.asarray(
                [
                    values[row_by_pair[(promoter, reporter)]]
                    for promoter in promoter_order
                ]
            )

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        ruby_truth, venus_truth = (
            panel(results.labels, "mRuby2"),
            panel(results.labels, "Venus"),
        )
        ruby_min, ruby_max = (
            panel(results.label_min, "mRuby2"),
            panel(results.label_max, "mRuby2"),
        )
        venus_min, venus_max = (
            panel(results.label_min, "Venus"),
            panel(results.label_max, "Venus"),
        )
        axes[0].errorbar(
            ruby_truth,
            venus_truth,
            xerr=np.vstack([ruby_truth - ruby_min, ruby_max - ruby_truth]),
            yerr=np.vstack([venus_truth - venus_min, venus_max - venus_truth]),
            fmt="o",
            markersize=4,
            linewidth=0.8,
            alpha=0.75,
        )
        axes[0].set(xscale="log", yscale="log")
        axes[0].set_xlabel("mRuby2 fluorescence (fold over background)")
        axes[0].set_ylabel("Venus fluorescence (fold over background)")
        axes[0].set_title("Digitised Figure 3A")

        ruby_pred = panel(results.scores, "mRuby2")
        venus_pred = panel(results.scores, "Venus")
        positive = (
            np.isfinite(ruby_pred)
            & np.isfinite(venus_pred)
            & (ruby_pred > 0)
            & (venus_pred > 0)
        )
        axes[1].scatter(ruby_pred[positive], venus_pred[positive], s=24, alpha=0.75)
        axes[1].set(xscale="log", yscale="log")
        axes[1].set_xlabel("predicted mRuby2 CDS coverage sum")
        axes[1].set_ylabel("predicted Venus CDS coverage sum")
        axes[1].set_title("Prediction")

        for promoter in ("pTDH3", "pRPL18B", "pREV1"):
            index = promoter_order.index(promoter)
            axes[0].annotate(
                promoter, (ruby_truth[index], venus_truth[index]), fontsize=8
            )
            if positive[index]:
                axes[1].annotate(
                    promoter, (ruby_pred[index], venus_pred[index]), fontsize=8
                )

        names = ("consensus", "mRuby2", "Venus")
        observed = [results.metrics[name].observed_fold_range for name in names]
        predicted = [results.metrics[name].predicted_fold_range for name in names]
        x = np.arange(len(names))
        width = 0.36
        axes[2].bar(x - width / 2, observed, width, label="observed")
        axes[2].bar(x + width / 2, predicted, width, label="predicted")
        axes[2].set_yscale("log")
        axes[2].set_xticks(x, names)
        axes[2].set_ylabel("max / min fold range")
        axes[2].set_title("Dynamic range")
        axes[2].legend()
        for index, name in enumerate(names):
            recovery = results.metrics[name].dynamic_range_recovery
            axes[2].text(
                index,
                max(observed[index], predicted[index]) * 1.12,
                f"recovery {recovery:.2f}",
                ha="center",
                fontsize=8,
            )

        title_model = out_dir.name.split("__")[0] if "__" in out_dir.name else ""
        title = "Lee et al. YTK promoter panel"
        if title_model:
            title += f" — {title_model}"
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(out_dir / "promoter_panel.png", dpi=150)
        plt.close(fig)

    def save_results(self, results: YTKResults, out_dir: Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            out_dir / "panel.npz",
            scores=results.scores,
            labels=results.labels,
            label_min=results.label_min,
            label_max=results.label_max,
        )
        (out_dir / "constructs.json").write_text(
            json.dumps(
                {
                    "construct_ids": results.construct_ids,
                    "promoters": results.promoters,
                    "reporters": results.reporters,
                },
                indent=2,
            )
            + "\n"
        )

    def load_results(self, out_dir: Path) -> YTKResults:
        arrays = np.load(Path(out_dir) / "panel.npz")
        metadata = json.loads((Path(out_dir) / "constructs.json").read_text())
        scores = arrays["scores"]
        labels = arrays["labels"]
        promoters = metadata["promoters"]
        reporters = metadata["reporters"]
        return YTKResults(
            scores=scores,
            labels=labels,
            label_min=arrays["label_min"],
            label_max=arrays["label_max"],
            construct_ids=metadata["construct_ids"],
            promoters=promoters,
            reporters=reporters,
            metrics=_metrics_by_panel(scores, labels, promoters, reporters),
        )

    def summary_dict(self, results: YTKResults) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "n_constructs_total": len(results.scores),
            "n_constructs_scored": int(np.isfinite(results.scores).sum()),
        }
        for panel_name, metrics in results.metrics.items():
            prefix = panel_name.lower()
            summary.update(
                {f"{prefix}_{key}": value for key, value in asdict(metrics).items()}
            )
        return summary

    def headline(self, results: YTKResults) -> str:
        metrics = results.metrics["consensus"]
        return (
            f"Consensus dynamic-range recovery = "
            f"{metrics.dynamic_range_recovery:.3f} "
            f"({metrics.predicted_fold_range:.1f}× predicted vs "
            f"{metrics.observed_fold_range:.1f}× observed); "
            f"fidelity = {metrics.dynamic_range_fidelity:.3f}; "
            f"Spearman ρ = {metrics.spearman_rho:.3f}; "
            f"log10 Pearson r = {metrics.pearson_log10:.3f}"
        )

    def headline_metric_labels(self) -> dict[str, str]:
        return {
            "consensus_dynamic_range_recovery": "range recovery (1 = matched)",
            "consensus_dynamic_range_fidelity": "range fidelity",
            "consensus_spearman_rho": "consensus Spearman ρ",
            "consensus_pearson_log10": "consensus log10 Pearson r",
        }

    def compare_plot(
        self,
        model_dirs: Mapping[str, Path],
        out_dir: Path,
    ) -> Path | None:
        """Plot raw reporter-CDS sums against fluorescence for every model."""
        import matplotlib.pyplot as plt

        model_names = sorted(model_dirs)
        loaded = [
            (model, self.load_results(model_dirs[model]))
            for model in model_names
            if (model_dirs[model] / "panel.npz").exists()
            and (model_dirs[model] / "constructs.json").exists()
        ]
        if not loaded:
            return None

        fig, axes = plt.subplots(
            len(REPORTERS),
            len(loaded),
            figsize=(5.2 * len(loaded), 4.5 * len(REPORTERS)),
            squeeze=False,
        )
        for reporter_index, reporter in enumerate(REPORTERS):
            for model_index, (model, results) in enumerate(loaded):
                ax = axes[reporter_index, model_index]
                rows = np.asarray(
                    [value == reporter for value in results.reporters], dtype=bool
                )
                measured = results.labels[rows]
                measured_min = results.label_min[rows]
                measured_max = results.label_max[rows]
                predicted = results.scores[rows]
                promoters = np.asarray(results.promoters)[rows]
                valid = (
                    np.isfinite(measured)
                    & np.isfinite(predicted)
                    & (measured > 0)
                    & (predicted > 0)
                )
                error_valid = (
                    valid
                    & np.isfinite(measured_min)
                    & np.isfinite(measured_max)
                    & (measured_min > 0)
                    & (measured_max > 0)
                )
                color = model_color(model, model_names)
                ax.errorbar(
                    measured[error_valid],
                    predicted[error_valid],
                    xerr=np.vstack(
                        [
                            measured[error_valid] - measured_min[error_valid],
                            measured_max[error_valid] - measured[error_valid],
                        ]
                    ),
                    fmt="none",
                    ecolor=color,
                    elinewidth=0.8,
                    alpha=0.35,
                    capsize=1.5,
                )
                ax.scatter(
                    measured[valid],
                    predicted[valid],
                    s=30,
                    color=color,
                    alpha=0.8,
                    edgecolors="white",
                    linewidths=0.35,
                )

                valid_indices = np.flatnonzero(valid)
                if valid_indices.size:
                    extrema = {
                        valid_indices[np.argmin(measured[valid])],
                        valid_indices[np.argmax(measured[valid])],
                    }
                    for index in extrema:
                        ax.annotate(
                            promoters[index],
                            (measured[index], predicted[index]),
                            xytext=(4, 4),
                            textcoords="offset points",
                            fontsize=8,
                        )

                metrics = results.metrics[reporter]
                ax.text(
                    0.04,
                    0.96,
                    f"Spearman ρ = {metrics.spearman_rho:.3f}\n"
                    f"range: {metrics.predicted_fold_range:.2f}× predicted / "
                    f"{metrics.observed_fold_range:.1f}× measured",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=8,
                )
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_xlabel(
                    f"measured {reporter} fluorescence (fold over background)"
                )
                ax.set_ylabel("predicted raw CDS coverage sum")
                ax.set_title(f"{model} — {reporter}")
                ax.grid(alpha=0.15, which="both")

        fig.suptitle(
            "Lee et al. YTK promoters: predicted reporter coverage vs fluorescence"
        )
        fig.tight_layout()
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        output = out_dir / "plot.svg"
        fig.savefig(output, bbox_inches="tight")
        plt.close(fig)
        return output

    def compare_plot_title(self) -> str:
        return "Lee et al. YTK promoter dynamic range"


__all__ = [
    "RangeMetrics",
    "YTKPromoterBenchmark",
    "YTKResults",
    "_range_metrics",
]
