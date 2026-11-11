"""Reconstruct the Chen 2016 pRPL5-GFP-KlURA3 PCR cassette from public sources.

This is an as-designed reconstruction, not DNA sequencing of the Gui strains.
The 3,144 bp product excludes retained deletion-library barcodes and outer arms.
Run from the repository root; requires the external ``pdftotext`` executable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tarfile
import textwrap


SOURCE_RECORDS = {
    "AJ002682.1.embl": {
        "url": "https://www.ebi.ac.uk/ena/browser/api/embl/AJ002682.1?download=true",
        "sha256": "ce8987428cbcb5bcfb27373481a4eacf02925f221f4aa54c2828e13e7fb5708c",
        "description": "Original pFA6a-GFPS65T-kanMX6 sequence, AJ002682.1",
    },
    "pgsku_adelaide_thesis.pdf": {
        "url": "https://digital.library.adelaide.edu.au/server/api/core/bitstreams/"
        "de3fefeb-c096-4ed1-993b-91f3c1e82903/content",
        "sha256": "1d98cd3c0e198356109bf1d46e016c50ba1d0b109b88499423ea5f3aeb0b0195",
        "description": "Sylvie Callegari (2011), Elucidating the biochemical and genetic "
        "basis of statin toxicity using yeast as a model, Appendix D, printed pp130-131. "
        "Full pGSKU sequence; methods identify plasmid as a gift from Francesca Storici.",
    },
    "chen2016_mmc1.pdf": {
        "url": "https://ars.els-cdn.com/content/image/1-s2.0-S2405471216301077-mmc1.pdf",
        "sha256": "dd861e22450268b69c849173a41b9ca71abaa70adc8e322e7cdb00f3d60834cf",
        "description": "Chen and Zhang 2016 supplemental Table S1 construction primers",
    },
    "R64-2-1.tgz": {
        "url": "https://downloads.yeastgenome.org/sequence/S288C_reference/genome_releases/"
        "S288C_reference_genome_R64-2-1_20150113.tgz",
        "sha256": "de3749dde095c1191a90c66d0da2b60d15cf2ccb8e14e0562f4f7b7d52a7c639",
        "description": "S288C reference, pRPL5 chrXVI:302515-303120, 1-based inclusive",
    },
    "Y00454.embl": {
        "url": "https://www.ebi.ac.uk/ena/browser/api/embl/Y00454?download=true",
        "sha256": "e9bc95c8988e060f5959d4d6fdb60b7fed8c992738e28b38ebb823f426eaee68",
        "description": "Original Kluyveromyces lactis URA3 sequence, Y00454.1; "
        "its entire CDS matches the thesis pGSKU source, which differs from Addgene at one synonymous base",
    },
}

ADDGENE_RECORDS = {
    "136113": "b6af9d1fc5e0fde834529d13b0d5fa200476fa13890d692e9135f85346f7d8ba",
    "136112": "486f541ed707d0b8bca59876afdf3e347aa508e050e9e73ddc435b0db3035637",
    "136111": "53e993fd56e101839ea38653ffc18b4d31c2a4f9b8902d4c40d0f5f87e8ce259",
}

# Names and sequences are transcribed from Chen 2016 Table S1, including its
# RPL5_GFP_forward name for the reverse primer and KI spelling for KlURA3.
PRIMERS = {
    "RPL5_kanMX_forward": (
        "CGTACGCTGCAGGTCGACGGATCCCCGGGTTAATTAAGGCGCGCCAGATCTGTTTAGCTT"
        "AAAAGGAGCCTGAGATGAATG"
    ),
    "RPL5_GFP_forward": (
        "AGTGAAAAGTTCTTCTCCTTTACTCAT"
        "TGTTGCTGAGATTTGTGACGGTGTCTTTTATTTTGAAGTGTAGGAGAAAACAAAAGTAAA"
    ),
    "GFP_forward": "ATGAGTAAAGGAGAAGAACTTTTCACT",
    "GFP_reverse": "GGTGTGGTCAATAAGAGCGAC",
    "KIURA3_forward": (
        "TTCCTGTAGGTCAGGTTGCTTTCTCAGGTATAGTATGAGGTCGCTCTTATTGACCACACC"
        "CTTGTTGTTCCTTACCATTAAGTTG"
    ),
    "KIURA3_kanMX_reverse": (
        "ATCGATGAATTCGAGCTCGTTTTCGACACTGGATGGCGGCGTTAGTATCGAATCGACAGC"
        "CGGAGACAATCATATGGGAGA"
    ),
}


def get_reverse_complement(sequence: str) -> str:
    return sequence.translate(str.maketrans("ACGT", "TGCA"))[::-1]


def write_fasta(path: Path, name: str, sequence: str) -> None:
    path.write_text(f">{name}\n{textwrap.fill(sequence, 80)}\n")


def get_unique_position(template: str, sequence: str) -> int:
    if template.count(sequence) != 1:
        raise ValueError(f"Expected one exact template match for {sequence}")
    return template.index(sequence)


def get_embl_sequence(path: Path) -> str:
    sequence_text = path.read_text().split("\nSQ ", 1)[1].split("\n", 1)[1]
    return re.sub("[^acgt]", "", sequence_text).upper()


def get_pgsku_sequence(pdf_path: Path) -> str:
    thesis = subprocess.check_output(
        ["pdftotext", "-layout", str(pdf_path), "-"], text=True
    )
    appendix = thesis.rsplit("Sequence of pGSKU plasmid used to amplify the", 1)[1]
    sequence_text = appendix.split("5’-", 1)[1].split("Appendix E", 1)[0]
    sequence = "".join(
        line.strip() for line in sequence_text.splitlines()
        if re.fullmatch("[ACGTacgt]+", line.strip())
    ).upper()
    if len(sequence) != 6965:
        raise ValueError(f"pGSKU extraction has {len(sequence)} bp; expected 6965")
    return sequence


def get_promoter_sequence(archive_path: Path) -> str:
    with tarfile.open(archive_path) as archive:
        fasta_member = next(
            member for member in archive.getmembers()
            if member.name.endswith(".fsa") and "reference_sequence" in member.name
        )
        fasta_handle = archive.extractfile(fasta_member)
        if fasta_handle is None:
            raise ValueError("Reference FASTA member is not readable")
        chromosomes = {}
        chromosome = ""
        for line in fasta_handle.read().decode().splitlines():
            if line.startswith(">"):
                match = re.search(r"\[chromosome=([^\]]+)\]", line)
                chromosome = match.group(1) if match else line[1:].split()[0]
                chromosomes[chromosome] = ""
            else:
                chromosomes[chromosome] += line.strip().upper()
    return chromosomes["XVI"][302514:303120]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/tasks/gui_audit/source"))
    parser.add_argument("--out", type=Path, default=Path("results/gui_audit/cassette"))
    parser.add_argument("--archive", type=Path, default=Path("archive"),
                        help="Directory containing the three user-downloaded Addgene pGSKU GenBank files")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sources = []
    for filename, record in SOURCE_RECORDS.items():
        path = args.source / filename
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != record["sha256"]:
            raise ValueError(f"Source checksum differs: {path}")
        sources.append({"path": str(path), **record})
    addgene_sequences = {}
    for sequence_id, expected_digest in ADDGENE_RECORDS.items():
        path = args.archive / f"addgene-plasmid-72243-sequence-{sequence_id}.gbk"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected_digest:
            raise ValueError(f"Addgene source checksum differs: {path}")
        sequence_text = path.read_text().split("\nORIGIN", 1)[1].split("//", 1)[0]
        addgene_sequences[sequence_id] = re.sub("[^acgt]", "", sequence_text.lower()).upper()
        sources.append({"path": str(path), "sha256": digest,
                        "url": "https://www.addgene.org/72243/sequences/",
                        "sequence_id": sequence_id,
                        "provenance": "User-downloaded Addgene GenBank file",
                        "description": ("Full pGSKU sequence" if sequence_id == "136113"
                                        else "Partial sequence; checked against full pGSKU")})

    # Check that every transcribed primer is present in the PDF text. Removing
    # spaces joins sequences split across lines without joining adjacent cells.
    primer_text = subprocess.check_output(
        ["pdftotext", "-raw", str(args.source / "chen2016_mmc1.pdf"), "-"], text=True
    )
    compact_text = re.sub(r"\s+", "", primer_text)
    for name, primer in PRIMERS.items():
        if primer not in compact_text:
            raise ValueError(f"Primer transcription not found in source PDF: {name}")

    vector = get_embl_sequence(args.source / "AJ002682.1.embl")
    if len(vector) != 4882:
        raise ValueError("Unexpected AJ002682.1 vector length")
    gfp_start = get_unique_position(vector, PRIMERS["GFP_forward"][3:])
    gfp_end = get_unique_position(vector, get_reverse_complement(PRIMERS["GFP_reverse"]))
    gfp_end += len(PRIMERS["GFP_reverse"])
    gfp_amplicon = "ATG" + vector[gfp_start:gfp_end]
    assert (gfp_start, gfp_end, len(gfp_amplicon)) == (62, 976, 917)
    gfp_cds_end = next(
        position + 3 for position in range(0, len(gfp_amplicon), 3)
        if gfp_amplicon[position:position + 3] in {"TAA", "TAG", "TGA"}
    )
    assert gfp_cds_end == 717

    thesis_pgsku = get_pgsku_sequence(args.source / "pgsku_adelaide_thesis.pdf")
    assert thesis_pgsku[622:1426] == get_embl_sequence(args.source / "Y00454.embl")[177:981]
    pgsku = addgene_sequences["136113"]
    assert len(pgsku) == 6968
    assert pgsku[241:1242] == addgene_sequences["136112"]
    assert pgsku[2949:3963] == get_reverse_complement(addgene_sequences["136111"])
    # The selected PCR interval excludes the backbone indels in the thesis.
    thesis_marker = thesis_pgsku[274:1775]
    addgene_marker = pgsku[277:1778]
    marker_differences = [(position, source, deposited) for position, (source, deposited)
                          in enumerate(zip(thesis_marker, addgene_marker, strict=True))
                          if source != deposited]
    assert marker_differences == [(488, "T", "A")]
    marker_template = get_reverse_complement(pgsku)
    marker_forward = PRIMERS["KIURA3_forward"]
    marker_reverse = PRIMERS["KIURA3_kanMX_reverse"]
    marker_start = get_unique_position(marker_template, marker_forward[60:])
    marker_end = get_unique_position(marker_template, get_reverse_complement(marker_reverse[60:]))
    marker_end += len(marker_reverse[60:])
    assert (marker_start, marker_end) == (5190, 6691)
    marker_amplicon = (marker_forward[:60] + marker_template[marker_start:marker_end]
                       + get_reverse_complement(marker_reverse[:60]))
    assert len(marker_amplicon) == 1621
    assert gfp_amplicon[-60:] == marker_amplicon[:60]
    fusion = gfp_amplicon + marker_amplicon[60:]

    promoter = get_promoter_sequence(args.source / "R64-2-1.tgz")
    assert len(promoter) == 606
    assert promoter.startswith(PRIMERS["RPL5_kanMX_forward"][60:])
    gfp_overlap_length = len(PRIMERS["GFP_forward"])
    assert promoter.endswith(get_reverse_complement(PRIMERS["RPL5_GFP_forward"][gfp_overlap_length:]))
    assert get_reverse_complement(PRIMERS["RPL5_GFP_forward"][:gfp_overlap_length]) == gfp_amplicon[:gfp_overlap_length]
    cassette = PRIMERS["RPL5_kanMX_forward"][:60] + promoter + fusion
    assert len(cassette) == 3144
    assert cassette[666:1383] == gfp_amplicon[:717]
    assert cassette[1932:2736] == get_reverse_complement(pgsku[625:1429])

    # Preserve the independently recovered thesis version as a comparison only.
    assert cassette[2595] == "T"
    thesis_cassette = cassette[:2595] + "A" + cassette[2596:]
    sequences = {
        "pgsku_callegari2011": thesis_pgsku,
        "pgsku_addgene_136113": pgsku,
        "gfp_aj002682_primer_amplicon": gfp_amplicon,
        "klura3_primer_amplicon": marker_amplicon,
        "gfp_klura3_fusion_reconstructed": fusion,
        "gui_cassette_thesis_3144bp": thesis_cassette,
        "gui_cassette_3144bp": cassette,
    }
    outputs = []
    for name, sequence in sequences.items():
        path = args.out / f"{name}.fasta"
        write_fasta(path, name, sequence)
        outputs.append({"path": str(path), "length_bp": len(sequence),
                        "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest()})
    provenance = {
        "sequence_status": "source_reconstructed_as_designed; not strain DNA verified",
        "primary_marker_template": "Addgene pGSKU sequence 136113 (6968bp)",
        "thesis_comparison": {
            "marker_interval_differences": [{"marker_position_0based": 488, "thesis": "T", "Addgene": "A"}],
            "cassette_position_0based": 2595,
            "consequence": "KlURA3 coding T141A; CTT-to-CTA, synonymous Leu47",
            "interpretation": "The Addgene allele is independently supported by RNA from three Gui strains; "
            "no sequence base was inferred from RNA during construction.",
        },
        "sources": sources,
        "primers": PRIMERS,
        "outputs": outputs,
        "coordinate_convention": "0-based half-open on GFP-forward PCR product",
        "features": {
            "forward_kanMX_homology": [0, 60, "+"],
            "RPL5_promoter": [60, 666, "+"],
            "GFP_CDS": [666, 1383, "+"],
            "GFP_downstream_ADH1_fragment": [1383, 1583, "+"],
            "KlURA3_CDS": [1932, 2736, "-"],
            "reverse_kanMX_homology": [3084, 3144, "-"],
        },
        "template_intervals": {
            "GFP": {"template": "AJ002682.1", "start": 62, "end": 976,
                    "strand": "+", "primer_added_prefix": "ATG"},
            "KlURA3": {"template": "Addgene pGSKU 136113", "start": 277, "end": 1778, "strand": "-"},
        },
        "limits": [
            "Retained deletion-library barcode/outer-primer sequences are not part of this PCR product.",
            "Exact GFP variant is independently supported by Gui IMP4, SEC61 and IME1 RNA reads; see verify_reporter_reads.py.",
            "An as-designed reconstruction cannot establish absence of strain-specific synthesis/PCR mutations.",
        ],
    }
    (args.out / "cassette_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"cassette_length_bp": len(cassette), "GFP_CDS": [666, 1383],
                      "output_directory": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
