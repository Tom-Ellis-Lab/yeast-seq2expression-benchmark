"""Read-only source audit for Gui et al. 2021 (issue #57).

This writes audit artifacts, not a benchmark distribution or an inferred cassette.
Inputs and download URLs are recorded in docs/benchmarks/gui_insertion_audit.md.
Requires the project's numpy/pandas/scipy/matplotlib dependencies; reads XLSX
with the standard library so that openpyxl is not an additional requirement.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import io
import itertools
import json
from pathlib import Path
import re
import tarfile
from urllib.parse import unquote
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

FLOOR = 0.0001347
NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def load_xlsx_table(data: bytes) -> pd.DataFrame:
    """Read first sheet of the source workbooks, whose header is row two."""
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        strings = []
        if "xl/sharedStrings.xml" in z.namelist():
            strings = ["".join(t.text or "" for t in e.findall(".//s:t", NS))
                       for e in ET.fromstring(z.read("xl/sharedStrings.xml"))]
        rows = []
        root = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
        for row in root.findall("s:sheetData/s:row", NS):
            cells = {}
            for c in row:
                v = c.find("s:v", NS)
                if v is None:
                    continue
                idx = 0
                for ch in re.sub(r"\d", "", c.attrib["r"]):
                    idx = idx * 26 + ord(ch) - 64
                cells[idx - 1] = (strings[int(v.text)] if c.get("t") == "s"
                                  else float(v.text))
            rows.append(cells)
    return pd.DataFrame(rows[2:]).rename(columns=rows[1]).dropna(axis=1, how="all")


def parse_fasta(text: str) -> dict[str, str]:
    chunks = {}
    for line in text.splitlines():
        if line.startswith(">"):
            chrom = re.search(r"\[chromosome=([^\]]+)\]", line)
            name = chrom.group(1) if chrom else line[1:].split()[0]
            if "NC_001224" in name or name in {"Mito", "MT", "chrmt"}:
                name = "Mito"
            name = name.removeprefix("chr")
            chunks[name] = []
        else:
            chunks[name].append(line.strip().upper())
    return {k: "".join(v) for k, v in chunks.items()}


def compute_reverse_complement(seq: str) -> str:
    return seq.translate(str.maketrans("ACGT", "TGCA"))[::-1]


def load_soft_samples(path: Path, labels: pd.DataFrame) -> pd.DataFrame:
    names = labels.reset_index().groupby("Gene.Name")["Gene.ID"].agg(list)
    rows = []
    for block in gzip.decompress(path.read_bytes()).decode().split("^SAMPLE = ")[1:]:
        attrs = defaultdict(list)
        for line in block.splitlines()[1:]:
            if line.startswith("!") and " = " in line:
                k, v = line.split(" = ", 1)
                attrs[k].append(v)
        source = attrs["!Sample_source_name_ch1"][0]
        symbol = source.split("/")[0]
        matches = ["WILD TYPE"] if symbol == "wild type" else names.get(symbol, [symbol])
        if len(matches) != 1:
            raise ValueError(f"Ambiguous genotype {symbol}: {matches}")
        strain = matches[0]
        if strain not in labels.columns:
            raise ValueError(f"Genotype {source} missing from Table S2")
        rows.append(dict(accession=block.splitlines()[0], strain=strain, source=source,
                         title=attrs["!Sample_title"][0],
                         file_url=attrs["!Sample_supplementary_file_1"][0],
                         relations=";".join(attrs["!Sample_relation"])))
    return pd.DataFrame(rows)


def build_deletion_designs(path: Path, labels: pd.DataFrame, genome: dict, aliases: dict) -> pd.DataFrame:
    designs = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    designs.columns = designs.columns.str.strip()
    designs = designs.map(str.strip)
    rows = []
    for strain in labels.columns[7:]:
        gene = labels.loc[strain]
        candidates = designs[designs.ORF_name.isin([strain, *aliases.get(strain, [])])]
        if candidates.empty:
            rows.append(dict(strain=strain, status="no_design_record"))
        for _, design in candidates.iterrows():
            row = dict(strain=strain, source_orf=design.ORF_name, design_id=design.rec_num,
                       uptag=design.UPTAG_sequence_20mer, dntag=design.DNTAG_sequence_20mer,
                       gene_start=int(gene.Start), gene_end=int(gene.End), strand=gene.Strand)
            up = design.UPstream45_primer_sequence
            down = design.DNstream45_primer_sequence
            # Older records append universal sequences and some use 55 bp arms.
            # Never blindly truncate to 45 bp; preserve all genomic homology.
            for tail in ("GATGTCCACGAGCTCTCT", "GATGTCCACGAGGTCTCT"):
                if len(up) > 45 and up.endswith(tail):
                    up = up[:-len(tail)]
                    break
            for tail in ("ATCGATGAATTCGAGCTCG", "CGGTGTCGGTCTCGTAG"):
                if len(down) > 45 and down.endswith(tail):
                    down = down[:-len(tail)]
                    break
            seq = genome[gene.Reference.removeprefix("chr")]
            oriented = seq if gene.Strand == "+" else compute_reverse_complement(seq)
            hits = []
            for arm in (up, compute_reverse_complement(down)):
                hits.append([m.start() for m in re.finditer(f"(?={re.escape(arm)})", oriented)]
                            if arm and set(arm) <= set("ACGT") else [])
            row.update(up_arm=up, down_arm=down, up_matches=len(hits[0]), down_matches=len(hits[1]))
            if len(hits[0]) == len(hits[1]) == 1 and hits[0][0] + len(up) < hits[1][0]:
                lo, hi = hits[0][0] + len(up), hits[1][0]
                if gene.Strand == "-":
                    lo, hi = len(seq) - hi, len(seq) - lo
                row.update(status="unique_exact_arms", deletion_start=lo + 1, deletion_end=hi,
                           start_delta=lo + 1 - int(gene.Start), end_delta=hi - int(gene.End))
            else:
                row["status"] = "unresolved_arms"
            rows.append(row)
    return pd.DataFrame(rows)


def run(args):
    source, out = args.source_dir, args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.archive) as z:
        tables = {i: load_xlsx_table(z.read(f"TableS{i}.xlsx")) for i in (2, 3, 5)}
    x = tables[2].set_index("Gene.ID")
    if not x.index.is_unique or not x.columns.is_unique:
        raise ValueError("Duplicate published gene/sample identifiers")
    strains = list(x.columns[7:])
    native = x.index[x.index != "GFP"]
    wt = x["WILD TYPE"]
    meta = load_soft_samples(source / "GSE142839_family.soft.gz", x)
    raw, raw_qc, raw_coordinates = {}, [], None
    with tarfile.open(source / "GSE142839_RAW.tar") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            accession = member.name.split("_")[0]
            d = pd.read_csv(io.BytesIO(gzip.decompress(tar.extractfile(member).read())), sep="\t")
            # Three distinct genes carry placeholder ID '.', not one gene to sum.
            named = d[d["Gene ID"] != "."].set_index("Gene ID")
            if not named.index.is_unique:
                raise ValueError(f"Duplicate real gene IDs in {member.name}")
            raw[accession] = named.TPM
            coords = named[["Reference", "Strand", "Start", "End"]].sort_index()
            if raw_coordinates is None:
                raw_coordinates = coords
            elif not coords.equals(raw_coordinates):
                raise ValueError(f"Inconsistent GEO annotations in {accession}")
            raw_qc.append(dict(accession=accession, rows=len(d), placeholder_ids=int((d["Gene ID"] == ".").sum()),
                               tpm_sum=float(d.TPM.sum()), zero_tpm=int((d.TPM == 0).sum())))
    r = pd.DataFrame(raw)
    if set(r.columns) != set(meta.accession):
        raise ValueError("GEO metadata and abundance archives differ")
    meta = meta.merge(pd.DataFrame(raw_qc), on="accession", validate="one_to_one")
    meta["gfp_tpm"] = meta.accession.map(r.loc["GFP"])
    meta.to_csv(out / "samples.tsv", sep="\t", index=False)
    matches, strain_qc, replicates = [], [], []
    for strain in ["WILD TYPE", *strains]:
        accessions = list(meta.loc[meta.strain == strain, "accession"])
        if not accessions:
            raise ValueError(f"No deposited samples for {strain}")
        mean = r.loc[x.index, accessions].mean(axis=1)
        reconstructed = mean.mask(mean == 0, FLOOR)
        diff = (reconstructed - x[strain]).abs()
        matches.append(dict(strain=strain, n_libraries=len(accessions),
                            mismatching_cells=int((diff > 1e-6).sum()), max_abs_error=float(diff.max())))
        if strain == "WILD TYPE":
            continue
        strain_qc.append(dict(strain=strain, n_libraries=len(accessions), gfp_tpm=x.loc["GFP", strain],
                              pearson_wt=pearsonr(wt.loc[native], x.loc[native, strain]).statistic,
                              deleted_gene_ratio=x.loc[strain, strain] / wt[strain],
                              median_log2fc=float(np.log2(x.loc[native, strain] / wt.loc[native]).median())))
        for a, b in itertools.combinations(accessions, 2):
            p, q = r.loc[native, a], r.loc[native, b]
            sel = wt.loc[native] >= 1
            # Shared WT denominator means this is NOT a noise-ceiling estimate.
            dp = np.log2((p[sel] + 1) / (wt.loc[native][sel] + 1))
            dq = np.log2((q[sel] + 1) / (wt.loc[native][sel] + 1))
            replicates.append(dict(strain=strain, sample_a=a, sample_b=b,
                                   pearson=pearsonr(p, q).statistic,
                                   spearman=spearmanr(p, q).statistic,
                                   shared_wt_log2fc_spearman=spearmanr(dp, dq).statistic,
                                   gfp_ratio=r.loc["GFP", a] / r.loc["GFP", b]))
    match = pd.DataFrame(matches)
    qc = pd.DataFrame(strain_qc)
    rep = pd.DataFrame(replicates)
    for name, frame in (("table_s2_reconciliation", match), ("strain_qc", qc), ("replicate_pairs", rep)):
        frame.to_csv(out / f"{name}.tsv", sep="\t", index=False)

    with tarfile.open(source / "R64-2-1.tgz") as tar:
        fsa = next(m for m in tar.getmembers() if m.name.endswith(".fsa"))
        gff = next(m for m in tar.getmembers() if m.name.endswith(".gff"))
        genome = parse_fasta(tar.extractfile(fsa).read().decode())
        annotation = tar.extractfile(gff).read().decode()
    genes, aliases = {}, {}
    for line in annotation.splitlines():
        if line.startswith("##FASTA"):
            break
        if line.startswith("#"):
            continue
        f = line.split("\t")
        attrs = {k: unquote(v) for k, v in (p.split("=", 1) for p in f[8].split(";") if "=" in p)}
        if f[2] == "gene":
            genes[attrs["ID"]] = (f[0], f[6], int(f[3]), int(f[4]))
            aliases[attrs["ID"]] = attrs.get("Alias", "").split(",")
    coord_bad = []
    for gene in native:
        row = x.loc[gene]
        observed = (row.Reference, row.Strand, int(row.Start), int(row.End))
        if observed != genes.get(gene):
            coord_bad.append(dict(gene=gene, published=observed, reference=genes.get(gene)))
    comparisons = {}
    for ref in args.reference:
        other = parse_fasta(ref.read_text())
        comparisons[str(ref)] = {chrom: genome[chrom] == other.get(chrom) for chrom in genome}
    gtf_bad = []
    if args.gtf:
        for line in args.gtf.read_text().splitlines():
            if line.startswith("#"):
                continue
            f = line.split("\t")
            if f[2] != "gene":
                continue
            gene = re.search(r'gene_id "([^"]+)"', f[8]).group(1)
            coords = ("chr" + f[0], f[6], int(f[3]), int(f[4]))
            if gene in strains and genes[gene] != coords:
                gtf_bad.append(dict(gene=gene, historical=genes[gene], current_gtf=coords))
    designs = build_deletion_designs(source / "Deletion_primers_PCR_sizes.txt", x, genome, aliases)
    designs.to_csv(out / "deletion_designs.tsv", sep="\t", index=False)
    # Export only the verified reference-derived promoter, explicitly not a full cassette.
    chrom, strand, lo, hi = genes["YPL131W"]
    assert strand == "+"
    promoter = genome[chrom.removeprefix("chr")][lo - 607:lo - 1]
    assert len(promoter) == 606 and promoter.startswith("AAAAGGAGCCTGAGATGAATG")
    reverse_primer_tail = "TGTTGCTGAGATTTGTGACGGTGTCTTTTATTTTGAAGTGTAGGAGAAAACAAAAGTAAA"
    assert promoter.endswith(compute_reverse_complement(reverse_primer_tail))
    (out / "rpl5_promoter_reference.fasta").write_text(
        f">RPL5_606bp_S288C_R64-2-1 chrXVI:{lo-606}-{lo-1}; reference-derived; not full cassette\n{promoter}\n")

    protein = load_xlsx_table((source / "chen2016_mmc2.xlsx").read_bytes())
    protein.columns = protein.columns.str.strip()
    protein = protein.set_index("Locus name").loc[strains, "Mean"]
    reporter = qc.set_index("strain").join(protein.rename("protein_relative_2016"))
    reporter.to_csv(out / "reporter_crosscheck.tsv", sep="\t")
    neighbor_rows = []
    nuclear = x[x.Reference.str.fullmatch(r"chr[IVX]+")]
    for strain in strains:
        g = x.loc[strain]
        ordered = nuclear[nuclear.Reference == g.Reference].sort_values(["Start", "End"])
        ids = list(ordered.index)
        pos = ids.index(strain)
        for rank in [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]:
            if not 0 <= pos + rank < len(ids):
                continue
            gene = ids[pos + rank]
            n = x.loc[gene]
            neighbor_rows.append(dict(strain=strain, gene=gene, rank=rank,
                                      gap_bp=max(g.Start - n.End, n.Start - g.End, 0),
                                      wt_tpm=wt[gene], edited_tpm=n[strain],
                                      log2fc=np.log2(n[strain] / wt[gene])))
    neighbors = pd.DataFrame(neighbor_rows)
    neighbors.to_csv(out / "neighbors.tsv", sep="\t", index=False)
    resolved = designs[designs.status == "unique_exact_arms"]
    design_counts = designs[designs.status != "no_design_record"].groupby("strain").size()
    boundary_options = resolved.groupby("strain").apply(
        lambda g: len(g[["deletion_start", "deletion_end"]].drop_duplicates()))
    summary = dict(
        n_strains=len(strains), n_endogenous_genes=len(native), n_libraries=len(meta),
        library_counts_per_genotype={str(k): int(v) for k, v in Counter(meta.groupby("strain").size()).items()},
        reconciled_cells=int(x.iloc[:, 6:].size), mismatching_cells=int(match.mismatching_cells.sum()),
        reconciliation_max_error=float(match.max_abs_error.max()),
        source_floor=FLOOR, raw_min_positive=float(r.where(r > 0).min().min()),
        published_min_above_floor=float(x.iloc[:, 6:].where(x.iloc[:, 6:] > FLOOR).min().min()),
        all_floor_genes=list(x.index[(x.iloc[:, 6:] == FLOOR).all(axis=1)]),
        historical_annotation_mismatches=coord_bad, reference_sequence_comparison=comparisons,
        current_gtf_insertion_gene_mismatches=gtf_bad,
        design_status_counts=designs.status.value_counts().to_dict(),
        multiple_design_loci=list(design_counts[design_counts > 1].index),
        multiple_resolved_boundary_loci=list(boundary_options[boundary_options > 1].index),
        missing_design_loci=list(designs.loc[designs.status == "no_design_record", "strain"]),
        unresolved_design_loci=sorted(set(designs.loc[designs.status == "unresolved_arms", "strain"])),
        standard_boundary_designs=int(((resolved.start_delta == 3) & (resolved.end_delta == -3)).sum()),
        n_neighbor_pairs=len(neighbors), n_neighbor_genes=neighbors.gene.nunique(),
        n_nearest_pairs=int((neighbors["rank"].abs() == 1).sum()),
        neighbor_wt_baseline_rho=float(spearmanr(neighbors.wt_tpm, neighbors.edited_tpm).statistic),
        neighbor_floor_pairs=int(((neighbors.wt_tpm == FLOOR) | (neighbors.edited_tpm == FLOOR)).sum()),
        reporter_rna_protein_rho=float(spearmanr(reporter.gfp_tpm, reporter.protein_relative_2016).statistic),
        reporter_outlier=reporter.loc["YNL075W"].to_dict(),
        replicate_pair_count=len(rep),
        replicate_pearson_range=[float(rep.pearson.min()), float(rep.pearson.max())],
        native_ura3_zero_in_all_geo=bool((r.loc["YEL021W"] == 0).all()),
        fitness_engineered_strains=int(tables[5]["Strain"].isin(strains).sum()),
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    inputs = [args.archive, *sorted(p for p in source.iterdir() if p.is_file()), *args.reference]
    if args.gtf:
        inputs.append(args.gtf)
    hashes = []
    for path in inputs:
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        hashes.append(dict(path=str(path), size=path.stat().st_size, sha256=digest))
    (out / "input_checksums.json").write_text(json.dumps(hashes, indent=2) + "\n")
    plot_audit(reporter, rep, neighbors, out)
    print(json.dumps(summary, indent=2))


def plot_audit(reporter, rep, neighbors, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(6, 5), layout="constrained")
    axis.hist(reporter.gfp_tpm, bins=35, color="#397b99")
    axis.set(xlabel="GFP TPM", ylabel="Strains", title="Reporter distribution")
    axis.annotate("YNL075W / IMP4", xy=(5129, 1), xytext=(2800, 20), arrowprops=dict(arrowstyle="->"))
    fig.savefig(out / "audit_reporter_distribution.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(6, 5), layout="constrained")
    axis.scatter(reporter.protein_relative_2016, reporter.gfp_tpm, s=15, alpha=.6, color="#397b99")
    row = reporter.loc["YNL075W"]
    axis.scatter([row.protein_relative_2016], [row.gfp_tpm], c="crimson", label="YNL075W")
    axis.set(xlabel="Relative GFP protein (2016)", ylabel="GFP TPM (2021)", xscale="log", yscale="log", title="Reporter RNA and protein abundance")
    axis.legend()
    fig.savefig(out / "audit_reporter_rna_protein.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(6, 5), layout="constrained")
    axis.scatter(rep.pearson, rep.shared_wt_log2fc_spearman, s=20, color="#397b99")
    axis.set(xlabel="Replicate expression Pearson r", ylabel="Replicate log₂ FC Spearman ρ", title="Shared WT: diagnostic, not a noise ceiling")
    fig.savefig(out / "audit_library_concordance.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(6, 5), layout="constrained")
    axis.scatter(np.log2(neighbors.wt_tpm + 1), np.log2(neighbors.edited_tpm + 1), s=5, alpha=.2, color="#397b99")
    axis.set(xlabel="WT log₂(TPM + 1)", ylabel="Edited log₂(TPM + 1)", title="Neighbor abundance mostly follows WT")
    fig.savefig(out / "audit_neighbor_abundance.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("archive/msab104_supplementary_data.zip"))
    parser.add_argument("--source-dir", type=Path, default=Path("data/tasks/gui_audit/source"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/gui_audit"))
    parser.add_argument("--reference", type=Path, action="append", default=[])
    parser.add_argument("--gtf", type=Path)
    run(parser.parse_args())
