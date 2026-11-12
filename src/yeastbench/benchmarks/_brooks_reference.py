"""Recover parental strands for the original Brooks input distribution.

New distributions carry ``native_strand`` explicitly. The published legacy
index omitted it, so its exact parental gene coordinates can be matched to the
bundled JS94 annotation without downloading a GFF or guessing from gene names.
"""

from __future__ import annotations

import json
from importlib.resources import files

import pandas as pd


def resolve_native_strands(index: pd.DataFrame) -> pd.Series:
    """Use declared strands, or require an exact legacy reference match."""
    if "native_strand" in index:
        strands = index["native_strand"]
        if not strands.isin(("+", "-")).all():
            raise ValueError("Brooks native_strand values must be '+' or '-'")
        return strands.copy()

    required = {
        "gene_id", "native_source_genome", "native_contig", "native_contig_len",
        "native_cds_start", "native_cds_end",
    }
    missing = required - set(index.columns)
    if missing:
        raise ValueError(
            "Brooks legacy native-strand lookup requires reference metadata; "
            f"missing {sorted(missing)}. Rebuild with explicit native_strand."
        )

    reference = json.loads(
        files("yeastbench.benchmarks").joinpath("brooks_native_genes.json").read_text()
    )
    unknown = index.loc[~index["gene_id"].isin(reference["genes"]), "gene_id"]
    if not unknown.empty:
        raise ValueError(
            "Brooks legacy native-strand lookup has unknown genes "
            f"{sorted(set(unknown))}; rebuild with explicit native_strand."
        )
    annotations = (
        pd.DataFrame.from_dict(reference["genes"], orient="index")
        .reindex(index["gene_id"])
        .set_axis(index.index)
    )
    mismatched = (
        index["native_source_genome"].ne(reference["native_source_genome"])
        | index["native_contig"].ne(reference["native_contig"])
        | index["native_contig_len"].ne(reference["native_contig_len"])
        | index["native_cds_start"].ne(annotations["start"])
        | index["native_cds_end"].ne(annotations["end"])
    )
    if mismatched.any():
        genes = sorted(set(index.loc[mismatched, "gene_id"]))
        raise ValueError(
            "Brooks legacy native reference differs from the bundled JS94 "
            f"annotation for {genes}; rebuild with explicit native_strand."
        )
    return annotations["strand"].rename("native_strand").copy()
