# Benchmarks

| Benchmark | Probes | Task | Primary metric |
| --- | --- | --- | --- |
| [Caudal eQTL](caudal_eqtl.md) | *cis*-regulatory variants | binary classification | area under the receiver operating characteristic curve (AUROC) / area under the precision–recall curve (AUPRC), mean over 4 negative sets |
| [Kita eQTL](kita_eqtl.md) | *cis*-regulatory variants (independent panel) | binary classification | AUROC / AUPRC (mean over 4 negative sets) |
| [Rafi / deBoer MPRA (promoter)](rafi_mpra_promoter.md) | promoter grammar | regression (marginalized logSED — log-scaled expression difference) | Pearson / Spearman |
| [Shalem MPRA (terminator)](shalem_mpra_terminator.md) | 3′-end / termination grammar | regression (marginalized logSED) | Pearson *r* (Spearman alongside) |
| [Chen synonymous MPRA](chen_synonymous.md) | coding-sequence / codon-usage effect on mRNA | regression, 3 libraries | Pearson *r* + Spearman ρ on `log2(R/D)` (R = RNA, D = DNA read counts) |
| [Wu RFP insertions](wu_rfpins.md) | genomic *position* effect at an open reading frame (ORF) deletion locus | regression | Pearson *r* + Spearman ρ (+ tail AUROC) |
| [Hong IGR insertions](hong_igr.md) | genomic *position* effect (intergenic) | regression | Spearman ρ on the held-out test loci (n = 52) |
| [Brooks SCRaMBLE](brooks_scramble.md) | genome-architecture rearrangement | coverage-track log-fold-change (LFC) | direction balanced accuracy, then Spearman / Pearson — reported per JS94 parental run |
| [Cuperus 5′-UTR](cuperus_mpra_5utr.md) | 5′ untranslated region (5′-UTR) / translational grammar | regression (HIS3 reporter) | Spearman/Pearson + partial correlation over Kozak features |
| [Lee YTK promoters](ytk_promoter.md) | integrated promoter dynamic range across two reporters | regression (reporter RNA range) | two-reporter consensus dynamic-range recovery + Spearman/Pearson |
| [Meneu foreign DNA](meneu_foreign_dna.md) | foreign / out-of-distribution (OOD) sequence (whole bacterial chromosomes in yeast) | zero-shot coverage-track prediction | per-window Pearson + Jensen–Shannon (JS) divergence (+ fold-change error) |

## Probing seven model capabilities across 11 benchmark tasks

**eQTL classification — [Caudal](caudal_eqtl.md), [Kita](kita_eqtl.md).** 
This benchmark asks the model to correctly predict the effect single-nucleotide changes on gene expression. Adapted from [Hao et al.](https://www.biorxiv.org/content/10.1101/2025.09.19.677475v1): We create mixed sets of eQTLs, i.e. (eVariant, eGene)-pairs, and distance-matched (variant, gene)-pairs that are not eQTLs, i.e. the variant does not have a statistical association with the gene's expression in the population. The model is then asked to classify each (variant, gene)-pair as either "eQTL" or "no-eQTL". This is done by making two predictions, with and without the respective variant, to calculate a log-change score between REF and ALT gene expression. Thresholding the magnitude of that score creates the aforementioned labels. Caudal and Kita differ only in where the positives come from: Caudal takes any eQTL within 25 kb of the gene body, Kita takes promoter, UTR, and ORF variants within 8 kb of the transcription start site.

**cis-element MPRA — [Rafi](rafi_mpra_promoter.md) (promoter),
[Shalem](shalem_mpra_terminator.md) (terminator),
[Chen](chen_synonymous.md) (synonymous CDS).** Each task uses a designed library of thousands of multi-bp variants with a reporter readout to probe a different regulatory layer: Rafi the promoter, Shalem the 3′-end and termination, Chen the coding sequence. We score an insert by measuring its logSED
effect and averaging that across many native host-gene contexts. The averaging
keeps the sequence in-distribution for the model and cancels the miscalibration
that comes from any single scaffold.

**Position effect — [Wu](wu_rfpins.md), [Hong](hong_igr.md).** Among others, [Brooks](https://www.science.org/doi/10.1126/science.abg0162) discovered that the genetic neighborhood of a gene has an influence on its transcription. Thus, instead of varying the sequence within an expression cassette (cf. _cis-element MPRAs_) this task keeps the expression cassette (promoter, UTRs, CDS, terminator) constant, and only the insertion site in the genome moves. The question is whether a model tracks where a gene sits or only reads the local promoter. Wu inserts at a deleted-ORF locus with a ~3.5 kb cassette; Hong inserts at an untouched intergenic region with a ~1.6 kb one.

**Structural rearrangement — [Brooks SCRaMBLE](brooks_scramble.md).** This is the
genome-scale companion to the position-effect benchmarks. SCRaMBLE shuffles the
layout of the genome but leaves each gene's coding sequence and promoter intact, so
what changes for a gene is the context downstream of it. The model reads sequence
and predicts coverage. We score it as the log-fold-change of coverage over the
coding sequence against an unscrambled parental control, and read that against a
leave-one-out reproducibility ceiling. The control was sequenced three times, and
we report the score against each of those runs separately rather than pooling
them: a gene only counts for a run it was sequenced deeply enough in, so the
three runs cover different genes and carry different ceilings.

**5′-UTR grammar — [Cuperus](cuperus_mpra_5utr.md).** A gene's 5′-untranslated
region influences the stability of mRNA and it's coding sequence is translated into protein. Cuperus put ~489k random 50 bp
5′-UTRs (plus 11,856 native yeast fragments) in front of a native histidine gene, a gene the cell
needs to grow, and grew the pool under histidine starvation: a 5′-UTR that
translates `HIS3` well makes more protein, so its cell divides faster and gains
share in the pool. Sequencing the pool before and after ~6 doublings turns each
variant's change in read count into a growth score. We score each UTR zero-shot by
the model's predicted `HIS3` mRNA coverage in the real reporter
(`CYC1`pr–`HIS3`–`CYC1`term), one forward pass per UTR, plus a second metric for
what the model adds beyond hand-crafted Kozak features. The ceiling to keep in mind:
the growth score is a protein readout, and an RNA-seq model can never predict
protein abundance — it can only explain the share of that signal's variance that
comes from mRNA-level differences (a uORF triggering decay, structure changing
stability), never the purely translational part.

**Integrated promoter range — [Lee YTK](ytk_promoter.md).** Nineteen promoters
drive mRuby2 or Venus from the same chromosomal locus. The task asks whether
predicted raw per-base reporter CDS coverage recovers the broad max/min range in
Figure 3A, with a geometric-mean consensus across both reporters and rank
correlation alongside it.

**Foreign-DNA coverage — [Meneu](meneu_foreign_dna.md).** Whole bacterial
chromosomes (*M. pneumoniae* and *M. mycoides*, linearised) are put into yeast and
RNA-sequenced. We tile the foreign chromosome into 5 kb windows and predict the
RNA-seq coverage of each one zero-shot. This is the far out-of-distribution end of
the suite — foreign sequence neither model could have trained on. We score each
chromosome two ways: the shape of the coverage within a window (median Pearson plus
JS divergence) and its overall magnitude across windows (fold-change error).

## Planned / v2

Check out future benchmarks in [`../ROADMAP.md`](../ROADMAP.md).

## Abbreviations
- eQTL: expression quantitative loci
- CDS: coding sequence
- UTR: untranslated regions
- CNN: convolutional neural network
