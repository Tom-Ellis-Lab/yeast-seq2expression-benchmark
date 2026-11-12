"""Legacy Brooks inputs recover native strands from exact source coordinates."""

import pandas as pd
import pytest

from yeastbench.benchmarks._brooks_reference import resolve_native_strands


@pytest.fixture
def legacy_index():
    # Both rows contain an inverted alternate gene; the parental strand must
    # come from JS94, independently of the alternate strand and gene-name suffix.
    return pd.DataFrame([
        {
            "gene_id": "YIR015W", "strand": "-",
            "native_source_genome": "genomes/JS96_ERCC92.fasta",
            "native_contig": "JS96_1", "native_contig_len": 98752,
            "native_cds_start": 33125, "native_cds_end": 33559,
        },
        {
            "gene_id": "YIR001C", "strand": "+",
            "native_source_genome": "genomes/JS96_ERCC92.fasta",
            "native_contig": "JS96_1", "native_contig_len": 98752,
            "native_cds_start": 7086, "native_cds_end": 7838,
        },
    ], index=[7, 3])


def test_legacy_strands_use_parental_annotation_and_preserve_row_order(legacy_index):
    strands = resolve_native_strands(legacy_index)
    assert strands.tolist() == ["+", "-"]
    assert strands.index.tolist() == [7, 3]
    assert "native_strand" not in legacy_index
    repeated = legacy_index.iloc[[1, 0, 1]].reset_index(drop=True)
    assert resolve_native_strands(repeated).tolist() == ["-", "+", "-"]


@pytest.mark.parametrize("column,value", [
    ("native_source_genome", "genomes/another.fasta"),
    ("native_contig", "chrIX"),
    ("native_contig_len", 98753),
    ("native_cds_start", 33126),
    ("native_cds_end", 33558),
])
def test_legacy_reference_mismatch_fails_closed(legacy_index, column, value):
    legacy_index.loc[7, column] = value
    with pytest.raises(ValueError, match="native reference differs"):
        resolve_native_strands(legacy_index)


def test_legacy_lookup_requires_known_gene_and_reference_metadata(legacy_index):
    with pytest.raises(ValueError, match="missing.*native_contig"):
        resolve_native_strands(legacy_index.drop(columns="native_contig"))
    legacy_index.loc[7, "gene_id"] = "UNKNOWN_W"
    with pytest.raises(ValueError, match="unknown genes"):
        resolve_native_strands(legacy_index)


def test_explicit_native_strands_support_other_reference_genomes():
    index = pd.DataFrame({"gene_id": ["another_gene"], "native_strand": ["-"]})
    assert resolve_native_strands(index).tolist() == ["-"]


@pytest.mark.parametrize("strand", ["", ".", "plus", None])
def test_invalid_explicit_native_strand_is_not_replaced_by_fallback(legacy_index, strand):
    legacy_index["native_strand"] = [strand, "-"]
    with pytest.raises(ValueError, match="native_strand values"):
        resolve_native_strands(legacy_index)
