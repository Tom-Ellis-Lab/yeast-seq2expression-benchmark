"""Refresh the random-init results appendix from completed benchmark outputs.

Run from the repository root with PYTHONPATH=src. Pending tasks remain explicit;
no estimates or historical prose values are substituted for missing results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from yeastbench.benchmarks.brooks import _brooks_metrics, _load_brooks_run_dir
from yeastbench.config import load_config
from yeastbench.registry import TASKS

MODEL = "shorkie_random_init"
START = "<!-- shorkie-random-init:start -->"
END = "<!-- shorkie-random-init:end -->"
# Ordered metric keys; every number comes from a persisted result.
METRICS = {
    "caudal_eqtl": ("Caudal", "AUROC / AUPRC", ["auroc_abs_mean", "auprc_abs_mean"]),
    "kita_eqtl": ("Kita", "AUROC / AUPRC", ["auroc_abs_mean", "auprc_abs_mean"]),
    "rafi_mpra_marginalized": (
        "Rafi",
        "Pearson r / Spearman ρ",
        ["overall_pearson_r", "overall_spearman_rho"],
    ),
    "shalem_mpra_marginalized": (
        "Shalem",
        "Pearson r / Spearman ρ",
        ["overall_pearson_r", "overall_spearman_rho"],
    ),
    "chen_synonymous": ("Chen", "mean r / mean ρ", ["pearson_mean", "spearman_mean"]),
    "wu_rfpins": ("Wu", "Pearson r / Spearman ρ", ["pearson_r", "spearman_rho"]),
    "hong_igr": (
        "Hong",
        "held-out ρ / fitted-track ρ",
        ["IntProp_spearman_rho", "inttrain_fitted_intprop_spearman_rho"],
    ),
    "brooks_scramble": ("Brooks", "shared-cohort balanced accuracy, r0 / r1 / r2", []),
    "cuperus_utr": (
        "Cuperus",
        "random clean / partial ρ; native / partial ρ",
        [
            "random_bucket5_spearman",
            "random_partial_spearman",
            "native_spearman",
            "native_partial_spearman",
        ],
    ),
    "ytk_promoter": (
        "YTK",
        "range recovery / Spearman ρ",
        ["consensus_dynamic_range_recovery", "consensus_spearman_rho"],
    ),
    "meneu_foreign_dna": (
        "Meneu",
        "shape r / JS↓: M. pneumoniae; M. mycoides",
        [
            "Mpneumo_shape_pearson",
            "Mpneumo_shape_js",
            "Mmmyco_shape_pearson",
            "Mmmyco_shape_js",
        ],
    ),
}


def report(config: Path, document: Path, artifact: Path) -> int:
    cfg = load_config(config)
    records = {}
    for task in METRICS:
        records[task] = {}
        for model in ("shorkie", MODEL):
            directory = cfg.out_dir / f"{model}__{task}"
            summary, metadata = (
                directory / "summary.json",
                directory / "run_metadata.json",
            )
            if summary.exists() and metadata.exists():
                records[task][model] = {
                    "summary": json.loads(summary.read_text()),
                    "metadata": json.loads(metadata.read_text()),
                    "summary_sha256": hashlib.sha256(summary.read_bytes()).hexdigest(),
                }
    brooks = {}
    if MODEL in records["brooks_scramble"]:
        models = ("shorkie", "yorzoi", MODEL)
        loaded = {
            m: _load_brooks_run_dir(cfg.out_dir / f"{m}__brooks_scramble")
            for m in models
        }
        if all(loaded.values()):
            before = set(loaded["shorkie"]["sample_ids"]) & set(
                loaded["yorzoi"]["sample_ids"]
            )
            shared = sorted(before & set(loaded[MODEL]["sample_ids"]))
            if set(shared) != before:
                raise ValueError(
                    "Brooks cohort shrank; do not overwrite the existing comparison"
                )
            brooks = {"n_before": len(before), "n_after": len(shared), "metrics": {}}
            for m, data in loaded.items():
                positions = {v: i for i, v in enumerate(data["sample_ids"])}
                brooks["metrics"][m] = _brooks_metrics(
                    data, np.array([positions[v] for v in shared])
                )

    def values(task, model, keys):
        record = records[task].get(model)
        if not record:
            return "—"
        if task == "brooks_scramble":
            if not brooks:
                return "comparison pending"
            numbers = brooks["metrics"][model]["dir_balanced_acc_per_rep"]
        else:
            numbers = [record["summary"][key] for key in keys]
        return " / ".join(f"{v:.4f}" for v in numbers)

    complete = sum(MODEL in record for record in records.values())
    # Before declaring the suite complete, re-read every raw result and verify
    # that its metrics reproduce the saved summary (including NaN positions).
    audited = False
    if complete == len(METRICS):

        def equivalent(actual, expected):
            if isinstance(actual, dict):
                return all(
                    k in expected and equivalent(v, expected[k])
                    for k, v in actual.items()
                )
            if isinstance(actual, (list, tuple)):
                return len(actual) == len(expected) and all(
                    equivalent(a, b) for a, b in zip(actual, expected)
                )
            if isinstance(actual, (int, float)):
                return bool(
                    np.isclose(actual, expected, rtol=1e-8, atol=1e-10, equal_nan=True)
                )
            return actual == expected

        for task in METRICS:
            benchmark = TASKS[task](**cfg.tasks_config.get(task, {}))
            loaded = benchmark.load_results(cfg.out_dir / f"{MODEL}__{task}")
            actual = benchmark.summary_dict(loaded)
            # Brooks persists LFC arrays, but not the coverage arrays needed
            # to recompute its two secondary shape diagnostics.
            if task == "brooks_scramble":
                actual.pop("shape_pearson_mean")
                actual.pop("shape_js_mean")
            if not equivalent(actual, records[task][MODEL]["summary"]):
                raise ValueError(
                    f"{task}: saved raw predictions do not reproduce the summary"
                )
        audited = True
    stamp = datetime.now(timezone.utc).isoformat()
    lines = [
        START,
        "## Shorkie trained from random initialization",
        "",
        f"Completed {complete}/11 full tasks. Updated {stamp[:19]} UTC.",
        "",
        "Eight published supervised-from-scratch folds, with reverse-complement averaging.",
        "Rafi retains its existing bfloat16/compiled adapter default; other tasks use",
        "float32. Chen uses four variants per batch, expanded over all host contexts.",
        "",
        "Pretrained columns use available saved runs; missing baselines are left blank.",
        "The earlier tables retain their historical results. Full summaries, sample counts,",
        "run metadata, and source checksums are in [the machine-readable report](results/shorkie_random_init.json).",
        "",
        "| Task | Metric(s) | Pretrained Shorkie | Random-init Shorkie | Eval time |",
        "| --- | --- | --- | --- | --- |",
    ]
    for task, (name, label, keys) in METRICS.items():
        current = records[task].get(MODEL)
        runtime = (
            f"{current['metadata']['elapsed_eval_s'] / 60:.1f} min"
            if current
            else "pending"
        )
        lines.append(
            f"| {name} | {label} | {values(task, 'shorkie', keys)} | {values(task, MODEL, keys)} | {runtime} |"
        )
    if brooks:
        lines += [
            "",
            f"Brooks uses the same {brooks['n_after']} shared sample IDs before and after adding random-init.",
            "Its three replicate metrics are reported separately. The raw-result audit checks",
            "all reconstructable metrics; Brooks's secondary shape diagnostics are available",
            "only in its saved summary because coverage arrays are not persisted.",
        ]
    lines += ["", END]
    content = document.read_text()
    section = "\n".join(lines)
    if START in content:
        prefix, rest = content.split(START, 1)
        _, suffix = rest.split(END, 1)
        content = prefix + section + suffix
    else:
        content = content.rstrip() + "\n\n" + section + "\n"
    document.write_text(content)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(
        json.dumps(
            {
                "updated_utc": stamp,
                "completed": complete,
                "total": 11,
                "raw_results_audited": audited,
                "raw_result_audit_exclusions": {
                    "brooks_scramble": ["shape_pearson_mean", "shape_js_mean"]
                },
                "input_lock": json.loads(
                    Path("src/yeastbench/data/manifest.lock.json").read_text()
                ),
                "source_sha256": {
                    name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                    for name in (
                        "src/yeastbench/models/shorkie/nn.py",
                        "src/yeastbench/models/shorkie/wrapper.py",
                        "src/yeastbench/adapters/shorkie_hong.py",
                        "src/yeastbench/registry.py",
                        "configs/default.yaml",
                    )
                },
                "runs": records,
                "brooks_shared_cohort": brooks,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Recorded {complete}/11 completed tasks in {document}")
    return complete


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--document", type=Path, default=Path("docs/results.md"))
    parser.add_argument(
        "--artifact", type=Path, default=Path("docs/results/shorkie_random_init.json")
    )
    args = parser.parse_args()
    report(args.config, args.document, args.artifact)
