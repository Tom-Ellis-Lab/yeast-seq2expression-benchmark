#!/usr/bin/env python3
"""Copy a curated set of result figures into ``img/results/`` for the docs.

The ``results/`` tree is gitignored, so the figures embedded in
``docs/benchmarks/*.md`` are copied here from a finished run and committed
under ``img/results/<task>/``. Re-run this after a fresh run to refresh the
committed figures.

Usage:
    python scripts/docs/collect_result_figures.py \
        --results-root results [--dest img/results] [--dry-run]

``--results-root`` defaults to ``results`` (where ``ybench run`` writes). Point
it elsewhere when the run lives in another worktree.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# task -> [(path under results-root, dest filename under img/results/<task>/)]
# One or two figures per task; keep the list tight and matched to what each
# spec's Results section embeds. Kita has no run yet, so it has no figures.
FIGURES: dict[str, list[tuple[str, str]]] = {
    "caudal_eqtl": [
        ("default/shorkie__caudal_eqtl/distance_stratified.png", "distance_stratified.png"),
        ("default/shorkie__caudal_eqtl/primary_roc_pr.png", "primary_roc_pr.png"),
    ],
    "rafi_mpra_promoter": [
        ("default/shorkie__rafi_mpra_marginalized/scatter_per_stratum.png", "scatter_per_stratum.png"),
    ],
    "shalem_mpra_terminator": [
        ("default/shorkie__shalem_mpra_marginalized/scatter.png", "shorkie_scatter.png"),
        ("default/yorzoi__shalem_mpra_marginalized/scatter.png", "yorzoi_scatter.png"),
    ],
    "chen_synonymous": [
        ("default/compare/per_task/chen_tdh3/plot.svg", "chen_tdh3_compare.svg"),
        ("default/compare/per_task/chen_gfp_r2/plot.svg", "chen_gfp_r2_compare.svg"),
    ],
    "wu_rfpins": [
        ("default/shorkie__wu_rfpins/scatter.png", "scatter.png"),
        ("default/shorkie__wu_rfpins/roc_pr_extreme_high.png", "roc_pr_extreme_high.png"),
    ],
    "hong_igr": [
        ("default/shorkie__hong_igr/scatter_primary.png", "scatter_primary.png"),
        ("default/shorkie__hong_igr/scatter_inttrain_fitted.png", "scatter_inttrain_fitted.png"),
    ],
    "brooks_scramble": [
        ("brooks/compare/per_task/brooks_scramble/shared_lfc.svg", "shared_lfc.svg"),
        ("brooks/compare/per_task/brooks_scramble/shared_per_sample.svg", "shared_per_sample.svg"),
    ],
    "cuperus_mpra_5utr": [
        ("default/shorkie__cuperus_utr/cuperus.png", "cuperus.png"),
        ("default/compare/per_task/cuperus_utr/plot.svg", "cuperus_compare.svg"),
    ],
    "meneu_foreign_dna": [
        ("meneu/yorzoi__meneu_foreign_dna/Mmmyco.png", "yorzoi_Mmmyco.png"),
        ("meneu/shorkie__meneu_foreign_dna_shorkie/Mpneumo.png", "shorkie_Mpneumo.png"),
    ],
}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-root", default="results", type=Path)
    p.add_argument("--dest", default=Path("img/results"), type=Path)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    copied, missing = 0, []
    for task, figs in FIGURES.items():
        for src_rel, dst_name in figs:
            src = args.results_root / src_rel
            dst = args.dest / task / dst_name
            if not src.exists():
                missing.append(str(src))
                continue
            print(f"{src} -> {dst}")
            if not args.dry_run:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
            copied += 1

    print(f"\n{copied} figure(s) {'would be ' if args.dry_run else ''}copied.")
    if missing:
        print(f"{len(missing)} missing source(s):", file=sys.stderr)
        for m in missing:
            print(f"  MISSING {m}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
