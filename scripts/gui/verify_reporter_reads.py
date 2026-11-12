"""Check reconstructed GFP/KlURA3 coding sequences against public IMP4 RNA reads.

Default input is HTTP bytes 0-29999999 of SRR10815545 R1, a deliberately partial
gzip file. Complete FASTQ records are streamed from it without requiring a gzip
footer. This provides a consensus check, not a complete variant-calling pipeline
or physical verification of untranscribed cassette DNA.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Iterator
import hashlib
import json
from pathlib import Path
import zlib

import numpy as np

from yeastbench.storage.files import sha256_file

READ_RUNS = {
    "SRR10815545": ("IMP4 / YNL075W, GSM4241065, SRX7489025", "045",
                    258387382, "8ce6967e9ce8b90855b4bbb2c66b11c5"),
    "SRR10815486": ("SEC61 / YLR378C, GSM4241006, SRX7488966", "086",
                    315462997, "5167fe5abaeb14c8845e8002d093e5b3"),
    "SRR10815459": ("IME1 / YJR094C, GSM4240979, SRX7488939", "059",
                    334217256, "b9d1de1979d788ef6156ae996595f0e5"),
}
BASES = "ACGT"
FEATURES = {"GFP": (666, 1383), "KlURA3": (1932, 2736)}


def get_fastq_records(path: Path) -> Iterator[tuple[str, str]]:
    """Yield sequence and quality from complete records in a gzip prefix."""
    decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)
    pending = b""
    with path.open("rb") as handle:
        while compressed_block := handle.read(2**20):
            pending += decompressor.decompress(compressed_block)
            lines = pending.split(b"\n")
            complete_lines = (len(lines) - 1) // 4 * 4
            for index in range(0, complete_lines, 4):
                if not lines[index].startswith(b"@") or not lines[index + 2].startswith(b"+"):
                    raise ValueError("Invalid FASTQ record structure")
                sequence = lines[index + 1].decode()
                quality = lines[index + 3].decode()
                if len(sequence) != len(quality):
                    raise ValueError("Sequence and quality lengths differ")
                yield sequence, quality
            pending = b"\n".join(lines[complete_lines:])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reads", type=Path, default=Path(
        "data/tasks/gui_audit/source/SRR10815545_1.prefix30MB.fastq.gz.part"))
    parser.add_argument("--cassette", type=Path, default=Path(
        "results/gui_audit/cassette/gui_cassette_3144bp.fasta"))
    parser.add_argument("--out", type=Path, default=Path("results/gui_audit/cassette"))
    args = parser.parse_args()
    run = args.reads.name.split("_", 1)[0]
    sample, remote_directory, full_file_bytes, full_file_md5 = READ_RUNS[run]
    read_url = (f"https://ftp.sra.ebi.ac.uk/vol1/fastq/SRR108/{remote_directory}/"
                f"{run}/{run}_1.fastq.gz")
    reference = "".join(line.strip() for line in args.cassette.read_text().splitlines()
                        if not line.startswith(">"))
    if len(reference) != 3144:
        raise ValueError("Expected the 3144bp reconstructed PCR cassette")
    seed_positions: defaultdict[str, list[int]] = defaultdict(list)
    for position in range(len(reference) - 20):
        seed_positions[reference[position:position + 21]].append(position)
    base_counts = np.zeros((len(reference), 4), dtype=np.int64)
    reads_scanned = 0
    reads_aligned = 0
    alignment_starts: Counter[tuple[str, int]] = Counter()
    reads_per_feature: Counter[str] = Counter()
    marker_allele_starts: defaultdict[str, set[tuple[str, int]]] = defaultdict(set)
    marker_allele_strands: defaultdict[str, Counter[str]] = defaultdict(Counter)
    complement = str.maketrans("ACGTN", "TGCAN")
    for read, quality in get_fastq_records(args.reads):
        reads_scanned += 1
        candidates = []
        for strand, oriented, oriented_quality in (
            ("+", read, quality),
            ("-", read.translate(complement)[::-1], quality[::-1]),
        ):
            candidate_starts: Counter[int] = Counter()
            for read_position in range(0, len(oriented) - 20, 15):
                for reference_position in seed_positions.get(oriented[read_position:read_position + 21], []):
                    candidate_starts[reference_position - read_position] += 1
            for start, _ in candidate_starts.most_common(2):
                low = max(0, start)
                high = min(len(reference), start + len(oriented))
                overlap = high - low
                overlaps_cds = any(
                    min(high, end) - max(low, begin) >= 25
                    for begin, end in FEATURES.values()
                )
                if overlap < 50 or not overlaps_cds:
                    continue
                matches = sum(reference[position] == oriented[position - start]
                              for position in range(low, high))
                if matches / overlap >= 0.9:
                    candidates.append((matches, strand, start, oriented, oriented_quality, low, high))
        if not candidates:
            continue
        _, strand, start, oriented, oriented_quality, low, high = max(candidates)
        reads_aligned += 1
        alignment_starts[(strand, start)] += 1
        for name, (begin, end) in FEATURES.items():
            if min(high, end) - max(low, begin) >= 25:
                reads_per_feature[name] += 1
        for position in range(low, high):
            base = oriented[position - start]
            if base in BASES and ord(oriented_quality[position - start]) - 33 >= 30:
                base_counts[position, BASES.index(base)] += 1
                if position == 2595:
                    marker_allele_starts[base].add((strand, start))
                    marker_allele_strands[base][strand] += 1

    depth = base_counts.sum(axis=1)
    winner = base_counts.argmax(axis=1)
    consensus = "".join(BASES[index] if coverage else "N"
                        for index, coverage in zip(winner, depth, strict=True))
    feature_checks = {}
    for name, (begin, end) in FEATURES.items():
        mismatches = [
            {"cassette_position_0based": position,
             "reference": reference[position], "consensus": consensus[position],
             "depth_q30": int(depth[position]),
             "counts": dict(zip(BASES, map(int, base_counts[position]), strict=True))}
            for position in range(begin, end)
            if depth[position] and reference[position] != consensus[position]
        ]
        feature_checks[name] = {
            "cds_length_bp": end - begin,
            "reads_aligned": reads_per_feature[name],
            "minimum_q30_depth": int(depth[begin:end].min()),
            "median_q30_depth": float(np.median(depth[begin:end])),
            "positions_q30_depth_ge10": int((depth[begin:end] >= 10).sum()),
            "consensus_mismatches": mismatches,
        }
    summary = {
        "sample": f"{sample}, {run}",
        "reads_url": read_url,
        "reads_local_path": str(args.reads),
        "reads_input_bytes": args.reads.stat().st_size,
        "reads_input_sha256": sha256_file(args.reads),
        "default_download_range": "HTTP bytes=0-29999999 (partial file)",
        "full_remote_file_bytes": full_file_bytes,
        "full_remote_file_md5_from_ENA": full_file_md5,
        "cassette_path": str(args.cassette),
        "cassette_sequence_sha256": hashlib.sha256(reference.encode()).hexdigest(),
        "method": "21nt seeds every15nt in both orientations; at least50nt ungapped "
        "overlap, 90% identity and25nt CDS overlap; consensus uses bases with Phred>=30",
        "reads_scanned": reads_scanned,
        "reads_aligned": reads_aligned,
        "distinct_start_orientation_pairs": len(alignment_starts),
        "features": feature_checks,
        "marker_thesis_difference_site": {
            "cassette_position_0based": 2595,
            "consequence": "KlURA3 coding T141A, CTT-to-CTA, synonymous Leu47",
            "allele_distinct_start_orientation_pairs": {
                base: len(starts) for base, starts in marker_allele_starts.items()
            },
            "allele_read_strand_counts": dict(marker_allele_strands),
        },
        "limits": "RNA consensus from one strain supports transcribed coding sequence only; "
        "it does not establish the complete integration sequence in every strain.",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "reporter_read_sequence_check.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (args.out / "reporter_read_base_counts.tsv").open("w") as handle:
        handle.write("cassette_position_0based\treference\tconsensus\tdepth_q30\tA\tC\tG\tT\n")
        for position, base in enumerate(reference):
            handle.write(f"{position}\t{base}\t{consensus[position]}\t{depth[position]}\t"
                         + "\t".join(map(str, base_counts[position])) + "\n")
    (args.out / "reporter_read_consensus.fasta").write_text(
        f">{run}_R1_prefix30MB_Q30_consensus\n" + consensus + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
