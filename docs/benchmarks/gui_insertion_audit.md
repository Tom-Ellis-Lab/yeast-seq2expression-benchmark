# Gui et al. 2021: source and sequence audit

Audit updated: 2026-11-11. Related to [issue #57](https://github.com/Tom-Ellis-Lab/yeast-seq2expression-benchmark/issues/57).
**All expression labels reconcile, and 205 of 240 replacement designs now have
source-resolved sequences.** The remaining 35 are explicitly excluded from the
first GFP RNA benchmark. Sequence reconstruction describes the documented
construct design; it does not certify the genome of each physical clone.
See the [benchmark specification](gui_gfp_rna.md) for scoring and model results.

The archived filename says 2026, but the RNA-seq study is
[Gui et al. 2021](https://doi.org/10.1093/molbev/msab104). The strain construction
comes from [Chen and Zhang 2016](https://doi.org/10.1016/j.cels.2016.03.009).
These are heterozygous replacements in BY4743, not intergenic insertions.

## Expression provenance

Downloaded the complete [GSE142839 SOFT metadata](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE142nnn/GSE142839/soft/GSE142839_family.soft.gz)
and [processed abundance archive](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE142nnn/GSE142839/suppl/GSE142839_RAW.tar).
Despite its name, `RAW.tar` contains StringTie abundance tables, not sequencing
reads: gene coordinates, coverage, FPKM and TPM. Raw reads are separately in SRA.

| Check | Result |
| --- | --- |
| Deposited libraries | 269, covering 240 engineered genotypes and WT |
| Engineered genotypes with one library | 222/240 |
| Engineered genotypes with multiple libraries | 18: thirteen with 2, two with 3, two with 4, one with 6 |
| WT libraries | One: GSM4241069, titled `BY4743-2` |
| Published matrix | 5,107 endogenous genes + GFP; 240 engineered profiles + WT |
| Reconciled values | All **1,231,028** published expression values |
| Discrepancies above absolute tolerance 1e-6 TPM | **Zero**; largest floating-point difference 3.64e-11 |
| Genotype mapping | Every GEO genotype maps unambiguously to a Table S2 column |
| Duplicate deposited expression profiles | None in the preliminary full-matrix check |

The recipe is: group libraries by genotype, take their arithmetic mean TPM,
then replace a zero mean with 0.0001347. This reproduces the entire table.
The comparison establishes the mean-and-floor recipe, but cannot distinguish
all conceivable equivalent preprocessing orders for entries that were zero in
every contributing library. The library metadata do not explicitly establish
the biological/technical independence of every repeated library. Do not
advertise 240 replicated measurements, or infer a second WT library from its title.

All 5,107 endogenous rows correspond to the historical annotation's verified
genes (including four annotated as verified/silenced), not every annotated
feature. This explains why Table S2's column sums are only about 0.658–0.933
million TPM. Do not renormalize this subset to one million: that would change
the published labels. The full deposited tables sum to about 0.994–0.999
million and have 7,124 rows each. Three unrelated features have the literal
gene ID `.` (SDL1, FLO8, CRS5); these must not be merged as one biological gene.
They are absent from Table S2.

All engineered profiles have endogenous-gene Pearson correlation >0.9 with
WT, reproducing that aspect of the paper's QC. The deposit contains the final
240 engineered genotypes, so it does not let us reconstruct the rejected
genotypes or independently rerun the original selection procedure.

## The GFP outlier is supported, not a spreadsheet typo

`YNL075W / IMP4`, library **GSM4241065**, has **5,129.369629 GFP TPM** in both
GEO and Table S2. Other engineered strains span 261.576–1,244.047 TPM.
Its endogenous transcriptome correlates with WT at Pearson r=0.9766 and its
deleted-gene expression ratio is 0.377, so it passes those basic checks.

The independently measured GFP protein abundance in the original
[2016 Data S1](https://ars.els-cdn.com/content/image/1-s2.0-S2405471216301077-mmc2.xlsx)
is also unusually high: **4.71045 relative units**, the maximum among these
240 strains and second-highest among the original 482. All 240 strain IDs
exist in that older dataset. Across them, RNA–protein Spearman rho is
**0.21428, p=0.000834**, reproducing the 2021 paper's rho=0.21 and p<0.0009.
Without IMP4, rho is 0.20438.

Recommendation: retain IMP4, use rank correlation as primary, and report
leave-one-out sensitivity for magnitude-sensitive metrics. Agreement across
assays supports the phenotype but does not prove cassette copy number or
exclude an unusual strain construction. The prose claim of an approximately
fivefold RNA range does not describe the full supplied table; it does describe
the range after omitting this one point. The older protein table likewise
has a wider full range than the approximately 2.5-fold description in the
2021 text. Do not silently trim either dataset to reproduce the prose.

## Replication and low-expression cautions

The 18 repeated genotypes yield 46 pairwise library comparisons, which are
not 46 independent biological observations. Endogenous-gene expression
Pearson r ranges from 0.9007 to 0.9987 (median 0.9787).

Absolute-expression agreement can mask disagreement about changes. For
`YGR089W / NNF2`, GSM4241041 versus GSM4241042 has absolute-expression
Pearson r=0.9030, but only **rho=0.0199** between log2 fold changes. This
diagnostic uses WT TPM >=1 and log2((sample TPM+1)/(WT TPM+1)). These change
correlations are **not noise ceilings**: both sides share the same noisy WT
denominator. One WT library prevents estimating WT biological variance.
For `YPR107C / YTH1`, the two GFP measurements differ by **2.23-fold**.
That is substantial compared with the central 90th/10th percentile reporter
range of about 2.25-fold across strains.

There are **10,186 floor-valued cells** in Table S2. Its smallest value above
the floor is 0.001347, consistent with dividing the minimum positive retained
value by ten. However, the deposited full matrices contain positive values as
small as 0.000015: the paper's phrase about the minimum in all transcriptional
profiles should not be interpreted as a literal rule over every deposited row.

Five endogenous rows are floor-valued in every profile: YCL058W-A, YCL018W,
YEL021W, YER014C-A, YKR035W-A. The native URA3 row (YEL021W) is exactly zero
in every deposited library. That is not evidence that the selectable marker
failed: the marker is **KlURA3**, not native S. cerevisiae URA3 (see below).
No separately named KlURA3 feature is present in the deposited abundance
tables; the supplementary URA3 analysis cannot be reproduced from the native
URA3 row.

The five-neighbors-per-side panel contains 2,392 pairs (1,531 distinct genes),
including 15 with a floor-valued WT or engineered measurement. Unregularized
log2 fold changes span **-21.48 to +17.69**. A notable high-to-zero example is
HAC1 (YFL031W) in the MOB2 replacement strain: 394.664 WT TPM versus a zero
deposited estimate, about 11 kb away. This needs read-level/quantification
inspection before interpreting it as strong distal repression.

Other checks worth preserving:

- The deleted-gene/WT expression ratios range from 0.1485 to 1.4302; twelve
  exceed one. Compensation, compositional effects, strain differences and
  quantification are possible explanations. This is not proof of mislabeled
  samples, and a fixed 0.5 label should not be imposed.
- Genome-wide median log2 fold changes vary from -0.675 to +0.347 across
  strains. Local effects must be distinguished from broad shifts and TPM
  composition. These medians are descriptive, not identified trans effects.
- Reusing measured WT expression predicts neighbor abundance at
  **rho=0.88488**. This is an experimental-reference diagnostic, not a
  sequence-only baseline, and motivates evaluating changes separately.
- Table S5 covers only **229** engineered strains plus WT.
- The original library selection deliberately balanced essential-gene
  neighborhoods and emphasized chromosomes I and VI. This is not a uniformly
  random sample of the yeast genome.

## Reference and integration coordinates

Downloaded SGD's historical
[R64-2-1 release](https://downloads.yeastgenome.org/sequence/S288C_reference/genome_releases/S288C_reference_genome_R64-2-1_20150113.tgz).
**Every endogenous Table S2 chromosome, strand, start and end matches that
GFF exactly.** All 17 reference sequences, after normalizing mitochondrial
names, are identical to the local R64-1-1 and R64-5-1 FASTAs. This is an
explicit sequence comparison of these files, not a general guarantee about
all releases bearing those labels.

The local Ensembl GTF differs at one insertion gene, **COX5B / YIL111W**:
historical start 155222 versus GTF start 155311 on chrIX. The historical
annotation includes a one-base first CDS segment at 155222 and an intron at
155223–155310. This is an annotation difference, not an assembly sequence
shift. Use the historical annotation for reproducing the labels, with explicit
CDS/exon handling for gene-level readouts.

Gene coordinates are not necessarily deletion junctions. Audited the existing
SGTC `Deletion_primers_PCR_sizes.txt` against the historical genome, retaining
all alternative designs and historical aliases. Genomic homology was separated
from appended universal primer sequences; some records use 55 bp rather than
45 bp arms. An exact match means both arms uniquely match the expected
chromosome in gene orientation. It does not verify which physical strain the
authors used.

- **253 candidate design records** map to the 240 genes through canonical IDs
  and historical aliases; 248 records have exact unique arm matches.
- **236 loci** have at least one exact design candidate.
- Four loci have no exact candidate yet: **YAL022C, YAR019C, YCL004W,
  YCL028W**. Five records across them have unresolved arms.
- Ten loci have multiple design records. Four have multiple different exact
  deletion spans: **YAR042W, YFL034C-B, YJL012C, YJL020C**. Historical aliases
  can represent earlier partial/split gene designs; they are alternatives to
  investigate, not interchangeable strain identifiers.
- 229 exact records imply deletion of `[gene_start+3, gene_end-3]`, retaining
  the terminal codons. Other designs have larger differences, sometimes
  hundreds or thousands of bases. A full-ORF replacement would be inaccurate.
- Some designs lack a down-tag; others differ in barcode sequence even where
  their genomic deletion spans agree. Do not synthesize missing barcode
  sequences for a supposedly verified benchmark reconstruction.

For example, YAR042W candidates replace 192622–193387, 192622–196182, or
193609–196182 on chrI. For MOB2, the historical YFL035C and YFL035C-A
designs imply different endpoints (63792 versus 63990). We need the original
strain/design identity to resolve these. The audit exports every candidate
rather than selecting the last row in the primer table.

## Cassette reconstruction

The [Chen and Zhang construction supplement](https://ars.els-cdn.com/content/image/1-s2.0-S2405471216301077-mmc1.pdf)
provides the PCR primers and a diagram with GFP and KlURA3 in convergent
orientations. The complete 3,144 bp PCR product can be reconstructed from:

- The 606 bp RPL5 promoter at chrXVI:302515–303120 (+), with exact primer matches.
- [pFA6a-GFPS65T-kanMX6, AJ002682.1](https://www.ebi.ac.uk/ena/browser/view/AJ002682.1),
  including the 717 bp GFP CDS and its downstream terminator.
- [pGSKU, Addgene #72243](https://www.addgene.org/72243/), full sequence record
  136113, for the KlURA3 module. Both marker primers match uniquely.

Callegari's [2011 thesis, Appendix D](https://digital.library.adelaide.edu.au/server/api/core/bitstreams/de3fefeb-c096-4ed1-993b-91f3c1e82903/content)
provides an independent pGSKU sequence. It differs from Addgene at one base
within this amplicon, a synonymous KlURA3 Leu47 codon (CTT versus CTA).
The Addgene allele is also present in its partial sequence record 136112 and
in public RNA reads from three Gui strains. The final cassette uses the
Addgene sequence directly; RNA data validate its sequence, rather than supply
an inferred sequence edit. The thesis's other length differences lie outside
the amplified marker module.

For GFP, all 717 CDS bases agree with the RNA consensus in the three inspected
libraries: IMP4/SRR10815545, SEC61/SRR10815486 and IME1/SRR10815459
(minimum Q30 coverage per base: 74, 19 and 8). These checks support
transcribed sequence identity. They cannot establish retained barcode identity,
copy number, native junctions or all untranscribed DNA in every strain.
The deposited GFP feature at positions 667–1399 spans 733 bp; it includes
sequence beyond the actual CDS. The model readout uses the 717 bp CDS,
`[666, 1383)` within the PCR core, including the stop codon.

## Conservative sequence cohort

The [deletion-collection architecture](https://pmc.ncbi.nlm.nih.gov/articles/PMC4063906/)
and the individual SGTC primer records determine the retained tags. In GFP
transcriptional orientation, the complete inserted DNA is:

```text
U1 – UPTAG – [3,144 bp core, already containing U2 and RC(D2)] – RC(DNTAG) – RC(D1)
```

The downstream oligos are reverse primers: their retained sequence must be
reverse-complemented. Native genomic homology arms belong to the flanking
genome and are not duplicated in the payload. Each accepted payload is
**3,219 bp**, with GFP CDS offsets **[704, 1421)**. Its orientation follows
the replaced gene, and the audited deletion bounds determine the junctions.

| Sequence decision | Genotypes |
| --- | ---: |
| One exact design, canonical two-tag structure, consistent two-stage PCR overlaps | **205** |
| Older outer upstream primer inconsistent with the canonical inner U1 sequence | 22 |
| Multiple candidate deletion-design records | 10 |
| Single design with unresolved genomic arms | 3 |
| Total | 240 |

The exclusion categories are applied in order: multiple-design loci are
excluded before testing individual candidates. This explains why the broad
coordinate audit finds four loci without an exact candidate, whereas only
three receive the final unresolved-arms reason. Older UP-only records sometimes
explicitly specify no downstream tag; no tag is invented for these designs.
The older upstream-primer discrepancy is sufficient to exclude them from v1.

The retained 205 include IMP4. Selection depends on source sequence evidence,
not expression magnitude or model agreement. `sequence_status=verified` means
that this source reconstruction and geometry passed validation, not that each
clone was resequenced. Actual barcode deviations are known in the
[physical deletion collection](https://pmc.ncbi.nlm.nih.gov/articles/PMC491991/).
All 240 labels remain available, including exclusions, to make the cohort
selection inspectable.

## Reproduction

Run in the installed project environment from the repository root. The
reconstruction additionally needs the `pdftotext` executable. Place the Gui
supplementary ZIP in `archive/msab104_supplementary_data.zip` and the downloaded
source files in `data/tasks/gui_audit/source/`. The three user-downloaded
Addgene GenBank files belong in `archive/`, named
`addgene-plasmid-72243-sequence-136113.gbk`,
`addgene-plasmid-72243-sequence-136112.gbk`, and
`addgene-plasmid-72243-sequence-136111.gbk`.

The source directory needs `GSE142839_family.soft.gz`, `GSE142839_RAW.tar`,
`R64-2-1.tgz`, `chen2016_mmc1.pdf`, `chen2016_mmc2.xlsx`,
`Deletion_primers_PCR_sizes.txt`, `AJ002682.1.embl`,
`pgsku_adelaide_thesis.pdf`, and `Y00454.embl`.

`scripts/gui/sources.json` records the expression, reference, construct and
RNA-read source URLs and SHA-256 hashes. `SOURCE_RECORDS` and `ADDGENE_RECORDS` in
`scripts/gui/reconstruct_cassette.py` records the sequence sources and expected
hashes; the reconstruction rejects changed files. The SGTC primer table was
copied from the existing Wu source download, whose documented upstream source
is [SGTC](http://yeastdeletion-sgtc.yeastgenome.org/Deletion_primers_PCR_sizes.txt).
The source documents and downloaded sequences are inputs, not runtime data.

```bash
# Reconcile all source labels and audit the deletion-library records.
python scripts/gui/audit_data.py --reference data/tasks/R64-5-1.fa

# Reconstruct the core, then add only source-resolved retained tags.
python scripts/gui/reconstruct_cassette.py
python scripts/gui/build_sequence_map.py

# Export labels and eligible full payloads for the runtime benchmark.
python scripts/gui/build_gui_distribution.py \
  --sequence-map data/tasks/gui_audit/resolved_loci.tsv \
  --constructs data/tasks/gui_audit/resolved_constructs.fasta

ybench data lock --artifacts gui
ybench data verify --tasks gui_gfp_rna
ybench run --config configs/gui_gfp_rna.yaml
```

The shared reference is `data/tasks/R64-5-1.fa`; historical-reference comparison
is optional in the audit but explicit genomic-arm validation is mandatory in
the sequence-map build. `audit_data.py` also accepts an additional `--reference`
or `--gtf` for checking other local references/annotations.

| Output | Purpose |
| --- | --- |
| `results/gui_audit/summary.json`, `input_checksums.json` | Original source checks and exact input hashes |
| `samples.tsv`, `table_s2_reconciliation.tsv` in that folder | All library mappings, GFP TPM and label errors |
| `deletion_designs.tsv` in that folder | Every candidate design, retained tag and inferred deletion interval |
| `results/gui_audit/cassette/cassette_provenance.json` | Sequence sources, primer checks and reconstructed features |
| `data/tasks/gui_audit/resolved_loci.tsv` | All 240 sequence decisions and exclusion reasons |
| `resolved_constructs.fasta`, `sequence_map_provenance.json` in that folder | 205 full payloads and reproducible input/output hashes |
| `data/tasks/gui/gui_gfp_rna_v1.tsv`, `constructs.fasta` | Compact runtime distribution |

Optional read validation uses `scripts/gui/verify_reporter_reads.py` with
`--reads`, `--cassette` and `--out`. Its JSON report records the public read URL,
exact downloaded prefix hash, scanned-read count and per-feature coverage.
For example, after downloading the documented 30 MB read prefix:

```bash
python scripts/gui/verify_reporter_reads.py \
  --reads data/tasks/gui_audit/source/SRR10815486_1.prefix30MB.fastq.gz.part \
  --out results/gui_audit/cassette/SRR10815486
```

This targeted sequence check is not a replacement for genomic sequencing.
Benchmark scatter and comparison figures are single-panel PNG/SVG files; no
PDF is written by the benchmark.

## Earlier exploratory analysis

The earlier exploration covers all 240 genotypes and is separate from the
205-genotype sequence benchmark. It uses bulk RNA libraries, not independently
measured haplotypes. Its neighbor-distance figures used native replaced-gene
midpoints as a provisional proxy for GFP position; they have not been recomputed
using the resolved cassette geometry. They should not be reported as exact
GFP-center distances or as a validated neighboring-gene benchmark.

The blue genome-expression heatmap columns do not imply that total RNA fell
for every gene. In IRC18/YJL037W, 4,254 of 5,001 plotted genes have lower TPM
than WT, losing 147,654 TPM in total, while 747 genes gain 175,920 TPM. The
plot weights genes equally, whereas TPM totals weight their abundance. The
plotted subset therefore gains summed TPM despite appearing predominantly
blue. This identifies a compositional pattern, not whether its cause was
biological or technical; median-centering would impose an additional assumption.
