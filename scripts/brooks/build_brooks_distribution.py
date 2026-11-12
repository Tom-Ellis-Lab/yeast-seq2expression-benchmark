"""Build the window-agnostic Brooks SCRaMBLE distribution (one task, all models).

This is the ONLY component that touches `gs://brooks-nanopore`. It resolves every
(gene, strain, copy) candidate and bakes a **window-agnostic** artifact the
benchmark slices at eval time — so the benchmark depends on these files alone (no
GCS, no per-strain genomes/GFF/BED). Instead of one pre-cut TSV per model
receptive field, it ships a generous gene-centred slice per construct; the
benchmark re-cuts the model's window (and reproduces membership + the per-window
dedup) at run time via `benchmarks.brooks.window_slice`.

Pipeline per SCRaMBLE strain S (control = JS94, 3 runs):
  1. per-strain genome FASTA + GFF + read BED from the bucket (cached).
  2. native-genome size factor per strain/run (native chromosomes are
     byte-identical across strains → removes depth + batch confounds).
  3. per-copy CDS coverage on `JS<S>_1`, gene-strand, size-factor normalised.
  4. `true_lfc = log2( normcov(strain copy) / mean normcov(JS94 gene) )`;
     JS94 per-run normalised coverages kept so the reproducibility ceiling is
     derivable from the same file.
  5. a generous gene-centred slice (±FLANK, clamped to the contig): sequence +
     per-base coverage for the alt construct (on `JS<S>_1`), and sequence only
     for the native baseline (on `JS94_1`; native coverage isn't scored). Native
     is identical across strains → deduped per gene.

No window-specific filtering at build time. The membership rules — alt/native
window fits, alt != native within the window, dedup byte-identical copies per
gene — are window-dependent and re-applied at eval time, so one artifact serves
any model with window ≤ FLANK.

Outputs under `data/tasks/brooks_scramble/`:
  - `brooks_constructs.fasta`  — `>alt~<sample_id>` / `>native~<gene_id>` slices
  - `brooks_cov.npz`           — per-base int32 coverage for the `alt~*` records
  - `brooks_index.tsv`         — one row per construct (schema in INDEX_COLUMNS)

Run:
  uv run python scripts/brooks/build_brooks_distribution.py --strains all
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from yeastbench.benchmarks.brooks import cov_key, gene_centre

ROOT = Path(__file__).resolve().parents[2]
BUCKET = "gs://brooks-nanopore"
CACHE = ROOT / "data" / "tasks" / "brooks_scramble" / "_cache"
OUT_DIR = ROOT / "data" / "tasks" / "brooks_scramble"
# Half-width of the stored gene-centred slice. Guarantees any model window
# W <= FLANK is contained (incl. contig-edge clamping); covers Yorzoi 4992 +
# Shorkie 16384. A larger model needs a rebuild with a bigger --flank.
FLANK = 16384

CONTROL = "JS94"
ROADMAP_STRAINS = ["JS606", "JS707", "JS711", "JS731", "JS732"]
NATIVE_CONTIGS = [
    f"chr{r}" for r in "I II III IV V VI VII VIII X XI XII XIII XIV XV XVI".split()
] + ["chrIXL"]
PSEUDOCOUNT = 1.0
MIN_READS = 10  # strain-side raw CDS read floor → low_support flag
# on the sample. Same threshold is used per-JS94-run
# inside the benchmark to decide which JS94 runs
# contribute to the per-replicate true_lfc set.
MIN_RUN_READS = 50_000  # per-run native library-size floor; failed/ultra-
# shallow runs (e.g. JS94 20180607=651, 20181122=
# 3878 reads) are dropped before they corrupt the
# control denominator / size factor / ceiling
SGD_RE = re.compile(r"^Y[A-P][LR]\d{3}[WC](?:-[A-Z])?$")

# One row per (gene, strain, copy) candidate. Sequence + per-base coverage for
# the alt/native slices live in the FASTA / npz keyed by `cov_key(...)`; this
# table carries the window-agnostic coords, provenance, and truth scalars the
# benchmark needs to re-cut each model's window and score it.
INDEX_COLUMNS = [
    "sample_id",
    "gene_id",
    "strain",
    "copy_idx",
    "n_copies",
    "strand",
    "rearr_class",
    # alt provenance + absolute coords (the strain's genome / synthetic contig)
    "alt_source_genome",
    "alt_contig",
    "alt_contig_len",
    "alt_cds_start",
    "alt_cds_end",
    "alt_slice_start",
    # native provenance + absolute coords (parental genome; shared per gene)
    "native_source_genome",
    "native_contig",
    "native_contig_len",
    "native_cds_start",
    "native_cds_end",
    "native_strand",
    "native_slice_start",
    # window-agnostic truth scalars (CDS-based; independent of the window)
    "strain_reads",
    "js94_reads_runs",  # raw counts (strain sum, JS94 per-run)
    "size_factor_strain",
    "norm_cov_strain",
    "norm_cov_js94_mean",
    "norm_cov_js94_runs",  # comma-list of all WT-run values (ceiling derivable)
    "true_lfc",
    # `low_support` is strain-side only: a (strain, gene, copy) whose strain raw
    # CDS-overlap count < MIN_READS. JS94 side is handled per-replicate inside
    # the benchmark (n_reps_supported derivable from `js94_reads_runs`).
    "low_support",
]


# ── bucket access (cached) ────────────────────────────────────


def _gs(*args: str) -> str:
    return subprocess.run(
        ["gcloud", "storage", *args], capture_output=True, text=True, check=True
    ).stdout


def _ls(prefix: str) -> list[str]:
    return [l for l in _gs("ls", f"{BUCKET}/{prefix}").splitlines() if l.strip()]


def fetch(remote: str) -> Path:
    """Download `BUCKET/remote` once into the cache, return local path."""
    local = CACHE / remote
    if not local.exists():
        local.parent.mkdir(parents=True, exist_ok=True)
        _gs("cp", f"{BUCKET}/{remote}", str(local))
    return local


def bed_paths(strain: str) -> list[str]:
    """Plain WT direct-RNA runs only. EXCLUDES tagged libraries — JS94
    has rrp6Δ/xrn1Δ RNA-decay-mutant runs (`..._20191017rrp6`, `xrn1`,
    `xrn1nc`) that are NOT the −SCRaMBLE WT baseline; the strict pattern
    (date immediately followed by `_porechopped`) drops them."""
    pat = re.compile(
        rf"^{strain}_\d{{8}}_porechopped_filtered_canuCorrected"
        r"_distinguished\.bed$"
    )
    return sorted(
        f"alignment/{l.rsplit('/', 1)[1]}"
        for l in _ls("alignment/")
        if pat.match(l.rsplit("/", 1)[1])
    )


# ── parsing ───────────────────────────────────────────────────


def read_fasta(path: Path) -> dict[str, str]:
    seqs: dict[str, list[str]] = {}
    cur = None
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                cur = line[1:].split()[0]
                seqs[cur] = []
            elif cur is not None:
                seqs[cur].append(line.strip())
    return {k: "".join(v).upper() for k, v in seqs.items()}


def parse_gff_cds(path: Path) -> pd.DataFrame:
    """One row per (contig, gene_id) with merged CDS span + strand.

    Uses CDS features where present, else the gene feature. Keeps only
    real SGD ORF ids.
    """
    rows = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9 or f[2] not in ("CDS", "gene"):
                continue
            m = re.search(r"ID=([^;]+)", f[8])
            if not m:
                continue
            gid = m.group(1)
            if not SGD_RE.match(gid):
                continue
            rows.append((f[0], gid, f[2], int(f[3]), int(f[4]), f[6]))
    df = pd.DataFrame(
        rows, columns=["contig", "gene_id", "ftype", "start", "end", "strand"]
    )
    if df.empty:
        return df
    cds = df[df.ftype == "CDS"]
    use = cds if not cds.empty else df
    g = (
        use.groupby(["contig", "gene_id", "strand"])
        .agg(start=("start", "min"), end=("end", "max"))
        .reset_index()
    )
    return g


def load_bed(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["chrom", "start", "end", "rid", "mapq", "strand"],
        usecols=[0, 1, 2, 4, 5],
        dtype={
            "chrom": str,
            "start": np.int64,
            "end": np.int64,
            "mapq": np.int16,
            "strand": str,
        },
    )


# ── coverage helpers ──────────────────────────────────────────


def count_overlaps(
    bed: pd.DataFrame, contig: str, lo: int, hi: int, strand: str
) -> int:
    """# reads on `contig`/`strand` overlapping 1-based inclusive [lo,hi]."""
    b = bed[(bed.chrom == contig) & (bed.strand == strand)]
    if b.empty:
        return 0
    # BED is 0-based half-open; gene coords 1-based inclusive
    return int(((b.start < hi) & (b.end > lo - 1)).sum())


def per_base_cov(
    bed: pd.DataFrame, contig: str, strand: str, contig_len: int
) -> np.ndarray:
    """Per-base read-depth array for one (contig, strand), via diff+cumsum
    over the BED intervals — O(N reads + contig_len), then `[w0:w0+W]`
    slicing per sample is O(W). Returns int32 array of length `contig_len`."""
    b = bed[(bed.chrom == contig) & (bed.strand == strand)]
    if b.empty:
        return np.zeros(contig_len, dtype=np.int32)
    starts = np.clip(b.start.to_numpy(), 0, contig_len)
    ends = np.clip(b.end.to_numpy(), 0, contig_len)
    delta = np.zeros(contig_len + 1, dtype=np.int32)
    np.add.at(delta, starts, 1)
    np.add.at(delta, ends, -1)
    return np.cumsum(delta[:contig_len]).astype(np.int32)


def native_total_reads(bed: pd.DataFrame) -> int:
    """Total reads on the native nuclear contigs (chrI–chrXVI minus the
    synthetic synIXR, plus chrIXL; chrMT and the synthetic contig
    excluded). The native genome is byte-identical across all strains, so
    this is a clean, simple library-size factor that removes the depth +
    sequencing-batch confound without per-gene modelling (median-of-ratios
    is a v2 refinement — see spec)."""
    return int(bed.chrom.isin(NATIVE_CONTIGS).sum())


def size_factor(strain_total: int, ref_total: float) -> float:
    """Strain/run library-size factor vs the JS94-mean native total."""
    return float(strain_total / max(ref_total, 1.0))


# ── generous gene-centred slice (window-agnostic) ─────────────


def gene_slice(
    contig_len: int, cds_start: int, cds_end: int, flank: int
) -> tuple[int, int]:
    """``[start, end)`` of the ±``flank`` gene-centred slice, clamped to the
    contig. Uses the shared ``gene_centre`` so the benchmark can re-cut any
    window <= ``flank`` from the stored slice. Absolute coords."""
    centre = gene_centre(cds_start, cds_end)
    return max(0, centre - flank), min(contig_len, centre + flank)


def cluster_copies(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Collapse near-coincident annotation entries (overlap artifacts);
    distinct copies are far-separated on the contig."""
    spans = sorted(spans)
    out: list[tuple[int, int]] = []
    for s, e in spans:
        if out and s <= out[-1][1] + 1000:  # same locus → keep longest
            ps, pe = out[-1]
            out[-1] = (min(ps, s), max(pe, e))
        else:
            out.append((s, e))
    return out


# ── main ──────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--strains", default="all", help="'roadmap' (5), 'all', or comma list"
    )
    ap.add_argument(
        "--flank",
        type=int,
        default=FLANK,
        help="Half-width of the stored gene-centred slice (bp); must "
        "be >= the largest model window to support. Default "
        f"{FLANK} (covers Yorzoi 4992 + Shorkie 16384).",
    )
    args = ap.parse_args()
    flank = int(args.flank)

    # Control = JS94 (−SCRaMBLE). The bucket has NO JS94 FASTA, but
    # JS96_1 is the parental synIXR sequence: identical 98,752 bp length
    # and byte-identical synIXR GFF coords to JS94_1 (a rearrangement
    # would change these) → JS96 supplies the parental *sequence*; JS94's
    # 3 runs supply the parental *expression* (shared coordinate system).
    par_fa = read_fasta(fetch("genomes/JS96_ERCC92.fasta"))
    par_syn = next(c for c in par_fa if c.startswith("JS96_"))
    js94_gff = parse_gff_cds(fetch("annotations/JS94.gff"))
    js94_read_contig = "JS94_1"  # JS94 BED chrom for control reads
    assert len(par_fa[par_syn]) == 98752, len(par_fa[par_syn])
    js94_genes = js94_gff[js94_gff.contig == js94_read_contig].set_index("gene_id")
    js94_bed_paths = bed_paths(CONTROL)
    kept = []
    for p in js94_bed_paths:
        b = load_bed(fetch(p))
        nt = native_total_reads(b)
        date = p.rsplit("/", 1)[1].split("_")[1]
        if nt >= MIN_RUN_READS:
            kept.append((date, b, nt))
        else:
            print(f"  JS94 run {date}: DROPPED (native reads {nt} < {MIN_RUN_READS})")
    if len(kept) < 2:
        raise RuntimeError("need ≥2 deep JS94 control runs for a ceiling")
    js94_beds = [b for _, b, _ in kept]
    js94_native_tot = [nt for _, _, nt in kept]
    ref_total = float(np.mean(js94_native_tot))
    js94_sf = [size_factor(t, ref_total) for t in js94_native_tot]
    print(
        f"JS94: {len(js94_genes)} synIXR genes (parental seq from "
        f"{par_syn}); {len(js94_beds)} deep WT runs "
        f"{[d for d, _, _ in kept]}"
    )

    # Length of the parental synIXR contig (JS94_1 = JS96_1 coord system);
    # used to clamp the native gene slices below.
    js94_syn_len = len(par_fa[par_syn])

    if args.strains == "roadmap":
        strains = ROADMAP_STRAINS
    elif args.strains == "all":
        strains = sorted(
            {
                re.match(r"(JS\d+)_", l.rsplit("/", 1)[1]).group(1)
                for l in _ls("genomes/")
                if re.search(r"/JS\d+_ERCC92\.fasta$", l)
            }
            - {CONTROL}
        )
    else:
        strains = args.strains.split(",")

    rows: list[dict] = []
    fasta_records: dict[str, str] = {}  # cov_key -> sequence slice
    cov_arrays: dict[str, np.ndarray] = {}  # alt cov_key -> per-base int32 coverage
    native_done: set[str] = set()  # genes whose native slice is written
    PAR_GENOME = "genomes/JS96_ERCC92.fasta"

    for S in strains:
        try:
            fa = read_fasta(fetch(f"genomes/{S}_ERCC92.fasta"))
            gff = parse_gff_cds(fetch(f"annotations/{S}.gff"))
            beds = [load_bed(fetch(p)) for p in bed_paths(S)]
        except Exception as e:  # noqa: BLE001
            print(f"  {S}: skip ({e})")
            continue
        beds = [b for b in beds if native_total_reads(b) >= MIN_RUN_READS]
        if not beds:
            print(f"  {S}: skip (no run ≥ {MIN_RUN_READS} native reads)")
            continue
        syn = next(c for c in fa if c.startswith(f"{S}_"))
        syn_len = len(fa[syn])
        bed = beds[0] if len(beds) == 1 else pd.concat(beds, ignore_index=True)
        s_sf = size_factor(native_total_reads(bed), ref_total)
        # Per-base alt coverage on JS<S>_1, both strands (one cumsum per strand,
        # then O(slice) extraction per construct).
        strain_alt_cov = {s: per_base_cov(bed, syn, s, syn_len) for s in ("+", "-")}
        g = gff[gff.contig == syn]
        n_kept = 0
        for gid, grp in g.groupby("gene_id"):
            if gid not in js94_genes.index:
                continue  # not in parental → no native baseline
            jrow = js94_genes.loc[gid]
            if isinstance(jrow, pd.DataFrame):
                jrow = jrow.iloc[0]
            nat_strand = jrow.strand
            nat_start, nat_end = int(jrow.start), int(jrow.end)
            # Native generous slice (parental; identical across strains → once).
            n0, n1 = gene_slice(js94_syn_len, nat_start, nat_end, flank)
            if gid not in native_done:
                nk = cov_key("native", gid)
                fasta_records[nk] = par_fa[par_syn][n0:n1]
                native_done.add(gid)
            # JS94 per-run normalised CDS coverage (gene strand), reads on JS94_1.
            j_raws = [
                count_overlaps(b, js94_read_contig, nat_start, nat_end, nat_strand)
                for b in js94_beds
            ]
            j_norm = [r / sf for r, sf in zip(j_raws, js94_sf)]
            j_mean = float(np.mean(j_norm))

            copies = cluster_copies(list(zip(grp.start, grp.end)))
            strand = grp.strand.iloc[0]
            for ci, (cs, ce) in enumerate(copies):
                cs, ce = int(cs), int(ce)
                a0, a1 = gene_slice(syn_len, cs, ce, flank)
                sample_id = f"{S}:{gid}:{ci}"
                ak = cov_key("alt", sample_id)
                fasta_records[ak] = fa[syn][a0:a1]
                cov_arrays[ak] = strain_alt_cov[strand][a0:a1].astype(np.int32)
                s_raw = count_overlaps(bed, syn, cs, ce, strand)
                s_norm = s_raw / s_sf
                true_lfc = float(
                    np.log2((s_norm + PSEUDOCOUNT) / (j_mean + PSEUDOCOUNT))
                )
                rows.append(
                    {
                        "sample_id": sample_id,
                        "gene_id": gid,
                        "strain": S,
                        "copy_idx": ci,
                        "n_copies": len(copies),
                        "strand": strand,
                        "rearr_class": "duplication"
                        if len(copies) > 1
                        else "context_change",
                        "alt_source_genome": f"genomes/{S}_ERCC92.fasta",
                        "alt_contig": syn,
                        "alt_contig_len": syn_len,
                        "alt_cds_start": cs,
                        "alt_cds_end": ce,
                        "alt_slice_start": a0,
                        "native_source_genome": PAR_GENOME,
                        "native_contig": par_syn,
                        "native_contig_len": js94_syn_len,
                        "native_cds_start": nat_start,
                        "native_cds_end": nat_end,
                        "native_strand": nat_strand,
                        "native_slice_start": n0,
                        "strain_reads": s_raw,
                        "js94_reads_runs": ",".join(str(int(v)) for v in j_raws),
                        "size_factor_strain": round(s_sf, 4),
                        "norm_cov_strain": round(s_norm, 3),
                        "norm_cov_js94_mean": round(j_mean, 3),
                        "norm_cov_js94_runs": ",".join(f"{v:.3f}" for v in j_norm),
                        "true_lfc": round(true_lfc, 4),
                        "low_support": bool(s_raw < MIN_READS),
                    }
                )
                n_kept += 1
        print(
            f"  {S}: {n_kept} candidate constructs (sf={s_sf:.3f}, {len(beds)} run(s))"
        )

    # ── write the 3-file window-agnostic artifact ──
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    idx = pd.DataFrame(rows, columns=INDEX_COLUMNS)
    idx_path = OUT_DIR / "brooks_index.tsv"
    idx.to_csv(idx_path, sep="\t", index=False)
    fasta_path = OUT_DIR / "brooks_constructs.fasta"
    with open(fasta_path, "w") as fh:
        for key in sorted(fasta_records):
            fh.write(f">{key}\n{fasta_records[key]}\n")
    cov_path = OUT_DIR / "brooks_cov.npz"
    np.savez_compressed(cov_path, **cov_arrays)
    total_mb = sum(p.stat().st_size for p in (idx_path, fasta_path, cov_path)) / 1e6
    print(
        f"\nwrote {idx_path.name} ({len(idx)} constructs, "
        f"{int(idx.low_support.sum()) if len(idx) else 0} low-support), "
        f"{fasta_path.name} ({len(fasta_records)} records), "
        f"{cov_path.name}  —  {total_mb:.1f} MB total"
    )


if __name__ == "__main__":
    main()
