"""Audit saved Brooks v2 runs and export the shared-cohort paper results.

Run from the repository root with PYTHONPATH=src. No inference or downloads.
The optional historical comparison uses only the local results/brooks tree.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

from yeastbench.benchmarks.brooks import _brooks_metrics, _load_brooks_run_dir
from yeastbench.config import load_config
from yeastbench.data.lock import locked_files, read_lock
from yeastbench.data.manifest import artifacts_for
from yeastbench.storage.files import sha256_file

MODELS = ("shorkie", "yorzoi", "exoshorkie_ensemble", "shorkie_random_init")
RUNS = ("JS94_r0", "JS94_r1", "JS94_r2")
COUNTS = [205, 80, 291]
CEILING_COUNTS = [198, 80, 169]
RAW_FILES = (
    "samples.json", "true_lfc_runs.npy", "pred_lfc_runs.npy",
    "n_reps_supported.npy", "summary.json", "run_metadata.json",
)


def file_record(path: Path) -> dict:
    return {"size": path.stat().st_size, "sha256": sha256_file(path)}


def independent_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict:
    """Use scipy correlations and a direct mean of per-class sign recall."""
    signs, predicted_signs = np.sign(truth), np.sign(prediction)
    return {
        "pearson_r": float(pearsonr(prediction, truth).statistic),
        "spearman_rho": float(spearmanr(prediction, truth).statistic),
        "dir_balanced_acc": float(np.mean([
            np.mean(predicted_signs[signs == value] == value)
            for value in np.unique(signs)
        ])),
    }


def assert_metrics(truth: np.ndarray, prediction: np.ndarray, expected: dict,
                   replicate: int, prefix: str = "") -> None:
    for key, value in independent_metrics(truth, prediction).items():
        np.testing.assert_allclose(
            value, expected[f"{prefix}{key}_per_rep"][replicate], rtol=1e-12, atol=1e-12,
        )


def subset(data: dict, sample_ids: list[str]) -> dict:
    positions = {sample: index for index, sample in enumerate(data["sample_ids"])}
    assert len(positions) == len(data["sample_ids"]), "Duplicate sample IDs"
    indices = np.array([positions[sample] for sample in sample_ids])
    return {key: sample_ids if key == "sample_ids" else value[indices]
            for key, value in data.items()}


def input_records(config) -> dict:
    lock = read_lock()
    records = {}
    for artifact in artifacts_for(["brooks_scramble"], list(MODELS)):
        for expected in locked_files(lock, artifact.id):
            path = Path(artifact.dest) / expected.path
            actual = file_record(path)
            assert actual == {"size": expected.size, "sha256": expected.sha256}, path
            records[path.as_posix()] = actual
    from huggingface_hub import snapshot_download

    yorzoi = next(run.model_config for run in config.runs if run.model == "yorzoi")
    snapshot = Path(snapshot_download(
        yorzoi["hf_repo"], revision=yorzoi["hf_revision"], local_files_only=True,
    ))
    for name in ("config.json", "model.safetensors"):
        records[f"hf://{yorzoi['hf_repo']}@{yorzoi['hf_revision']}/{name}"] = file_record(snapshot / name)
    return records


def unchanged_geometry_check(legacy_root: Path, current: dict) -> dict:
    from yeastbench.registry import TASKS

    for name in ("shorkie__brooks_scramble", "shorkie__brooks_scramble_shorkie"):
        directory = legacy_root / name
        previous = _load_brooks_run_dir(directory)
        if previous is not None:
            break
    else:
        return {"checked": False, "reason": "Historical Shorkie arrays absent"}
    frame = TASKS["brooks_scramble"]()._materialize(16384)
    unchanged = (
        (frame.cds_start_in_window - 1024).clip(lower=0)
        == (frame.native_cds_start_in_window - 1024).clip(lower=0)
    ) & (
        (frame.cds_end_in_window - 1024).clip(upper=14336)
        == (frame.native_cds_end_in_window - 1024).clip(upper=14336)
    )
    samples = sorted(set(frame.loc[unchanged, "sample_id"])
                     & set(previous["sample_ids"]) & set(current["sample_ids"]))
    before = subset(previous, samples)["pred_lfc_runs"]
    after = subset(current, samples)["pred_lfc_runs"]
    np.testing.assert_allclose(before, after, rtol=0, atol=1e-5, equal_nan=True)
    finite = np.isfinite(before) & np.isfinite(after)
    return {"checked": True, "samples": len(samples), "finite_pairs": int(finite.sum()),
            "absolute_tolerance": 1e-5, "maximum_absolute_difference": float(np.max(np.abs(before[finite] - after[finite]))),
            "raw_files": {str(directory / name): file_record(directory / name) for name in RAW_FILES}}


def report(config: Path, output_json: Path, output_csv: Path,
           legacy_summary: Path) -> None:
    config_path = config
    config = load_config(config_path)
    assert {run.model for run in config.runs} == set(MODELS)
    source_digest = hashlib.sha256()
    sources = {}
    for path in sorted(Path("src/yeastbench").rglob("*.py")):
        source_digest.update(str(path).encode())
        source_digest.update(path.read_bytes())
    for path in (
        config_path, Path(__file__).resolve().relative_to(Path.cwd()),
        Path("src/yeastbench/benchmarks/brooks.py"),
        Path("src/yeastbench/benchmarks/_brooks_reference.py"),
        Path("src/yeastbench/registry.py"),
        Path("src/yeastbench/benchmarks/brooks_native_genes.json"),
        Path("src/yeastbench/models/exoshorkie/params.json"),
        Path("src/yeastbench/data/manifest.lock.json"),
    ):
        sources[path.as_posix()] = file_record(path)

    loaded, records = {}, {}
    for model in MODELS:
        directory = config.out_dir / f"{model}__brooks_scramble"
        data = _load_brooks_run_dir(directory)
        assert data is not None, f"Missing complete results: {directory}"
        metadata = json.loads((directory / "run_metadata.json").read_text())
        assert metadata["task_version"] == "v2", metadata
        assert metadata["config_hash"] == config.source_hash, metadata
        assert metadata["code_sha256"] == source_digest.hexdigest(), "Source changed after inference"
        expected = next(run.model_config for run in config.runs if run.model == model)
        assert metadata["model_config"] == expected, metadata
        loaded[model] = data
        records[model] = {
            "run_metadata": metadata,
            "saved_summary": json.loads((directory / "summary.json").read_text()),
            "raw_files": {str(directory / name): file_record(directory / name) for name in RAW_FILES},
        }

    shared_ids = sorted(set.intersection(*(set(data["sample_ids"]) for data in loaded.values())))
    assert len(shared_ids) == 698, len(shared_ids)
    aligned = {model: subset(data, shared_ids) for model, data in loaded.items()}
    reference = aligned["yorzoi"]
    unchanged = ("true_lfc_runs", "n_reps_supported", "low_support")
    for model, data in aligned.items():
        for key in unchanged:
            np.testing.assert_array_equal(data[key], reference[key], err_msg=f"{model}: {key}")
        metrics = _brooks_metrics(data, np.arange(len(shared_ids)))
        assert metrics["n_scored"] == 327, metrics
        assert metrics["n_scored_per_rep"] == COUNTS, metrics
        assert metrics["n_ceiling_per_rep"] == CEILING_COUNTS, metrics
        for replicate in range(3):
            truth = data["true_lfc_runs"]
            prediction = data["pred_lfc_runs"]
            selected = ~data["low_support"] & np.isfinite(truth[:, replicate])
            assert np.isfinite(prediction[selected, replicate]).all()
            assert_metrics(truth[selected, replicate], prediction[selected, replicate], metrics, replicate)
            other = truth[:, [index for index in range(3) if index != replicate]]
            count = np.isfinite(other).sum(axis=1)
            mean = np.divide(np.nansum(other, axis=1), count,
                             out=np.full(len(other), np.nan), where=count > 0)
            ceiling = selected & np.isfinite(mean)
            assert int(ceiling.sum()) == CEILING_COUNTS[replicate]
            assert_metrics(truth[ceiling, replicate], mean[ceiling], metrics, replicate, "ceiling_")
        records[model]["shared_cohort_metrics"] = metrics
        records[model]["full_cohort_metrics"] = _brooks_metrics(loaded[model], np.arange(len(loaded[model]["sample_ids"])))

    legacy = {"checked": False, "reason": "Optional historical local results absent"}
    if legacy_summary.exists():
        previous = json.loads(legacy_summary.read_text())
        assert shared_ids == sorted(previous["shared_cohort"]["sample_ids"])
        legacy = {"checked": True, "summary": {str(legacy_summary): file_record(legacy_summary)},
                  "labels_and_support_checked": False}
        directory = legacy_summary.parent.parent / "yorzoi__brooks_scramble"
        old = _load_brooks_run_dir(directory)
        if old is not None:
            old = subset(old, shared_ids)
            for key in unchanged:
                np.testing.assert_array_equal(old[key], reference[key], err_msg=f"Legacy {key}")
            legacy["labels_and_support_checked"] = True
            legacy["raw_files"] = {str(path): file_record(path) for path in sorted(directory.glob("*.npy"))}
            legacy["raw_files"][str(directory / "samples.json")] = file_record(directory / "samples.json")
        legacy["unchanged_shorkie_geometry"] = unchanged_geometry_check(legacy_summary.parent.parent, loaded["shorkie"])

    columns = ["sample_id", "parental_run", "true_lfc", *[f"pred_lfc_{model}" for model in MODELS]]
    rows = []
    for index, sample in enumerate(shared_ids):
        if reference["low_support"][index]:
            continue
        for replicate, name in enumerate(RUNS):
            truth = float(reference["true_lfc_runs"][index, replicate])
            if np.isfinite(truth):
                rows.append([sample, name, truth, *[
                    float(aligned[model]["pred_lfc_runs"][index, replicate]) for model in MODELS
                ]])
    assert len(rows) == 576, len(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(columns)
        writer.writerows(rows)  # Python repr preserves round-trip float64 precision.
    with output_csv.open(newline="") as handle:
        restored = list(csv.DictReader(handle))
    np.testing.assert_array_equal(
        [[float(row[column]) for column in columns[2:]] for row in restored],
        [row[2:] for row in rows],
    )
    assert [(row["sample_id"], row["parental_run"]) for row in restored] == [tuple(row[:2]) for row in rows]

    result = {
        "task": "brooks_scramble", "task_version": "v2", "config": str(config_path),
        "shared_cohort": {"n_total": 698, "n_scored": 327, "n_scored_per_rep": COUNTS,
                          "n_ceiling_per_rep": CEILING_COUNTS, "sample_ids": shared_ids},
        "replicates": list(RUNS), "models": records, "legacy_validation": legacy,
        "validation": {"independent_scipy_metrics": True, "csv_exact_float_roundtrip": True,
                       "identical_labels_and_support_across_models": True},
        "predictions_csv": {"path": str(output_csv), "rows": len(rows), **file_record(output_csv)},
        "inputs": input_records(config), "source_files": sources,
        "python_source_sha256": source_digest.hexdigest(),
        "packages": {name: importlib.metadata.version(name) for name in (
            "numpy", "pandas", "scipy", "scikit-learn", "torch", "yorzoi", "h5py", "huggingface_hub",
        )},
        "notes": ["Replicate metrics are never averaged.",
                  "Historical Yorzoi runs did not record a Hub revision; both cached weight-bearing snapshots share the pinned model/config blobs.",
                  "Shape diagnostics in saved summaries are model-window dependent and are not headline comparisons."],
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for model in MODELS:
        metrics = records[model]["shared_cohort_metrics"]
        print(model, {key: [round(value, 4) for value in metrics[key]] for key in (
            "dir_balanced_acc_per_rep", "spearman_rho_per_rep", "pearson_r_per_rep",
        )})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/brooks_native_v2.yaml"))
    parser.add_argument("--output-json", type=Path, default=Path("docs/benchmarks/brooks_native_v2_results.json"))
    parser.add_argument("--output-csv", type=Path, default=Path("docs/benchmarks/brooks_native_v2_predictions.csv"))
    parser.add_argument("--legacy-summary", type=Path, default=Path("results/brooks/compare__shared/summary.json"))
    report(**vars(parser.parse_args()))
