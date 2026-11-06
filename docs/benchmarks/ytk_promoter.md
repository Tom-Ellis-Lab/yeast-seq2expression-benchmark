# Lee et al. YTK promoter panel

| | |
| --- | --- |
| **Question** | Do predicted reporter RNA levels span the broad range measured for the 19 Yeast Toolkit promoters, or collapse into a narrow band? |
| **Experiment** | Each promoter drives either mRuby2 or Venus from a construct integrated at `URA3` in BY4741: 19 promoters × 2 reporters = 38 constructs. The paper reports four biological replicates as a median and range. |
| **Truth** | Digitised Figure 3A fluorescence divided by the fluorescence of non-fluorescent cells. v1 treats these values as truth. |
| **Readout** | Sum predicted RNA coverage over the exact reporter CDS bases in raw, untransformed, unbinned per-base space. |
| **Headline** | Two-reporter consensus dynamic-range recovery, with range fidelity, Spearman ρ, and log10 Pearson *r*. mRuby2 and Venus metrics are reported alongside. |
| **Task name** | `ytk_promoter` |
| **Paper** | [Lee et al. 2015, DOI 10.1021/sb500366v](https://doi.org/10.1021/sb500366v) |

## Experimental panel

Lee et al. placed 19 native constitutive promoters upstream of three fluorescent
reporters. Figure 3A contains the mRuby2 and Venus measurements used here. Each
point is the median of four biological replicates; its horizontal and vertical
bars show the replicate ranges. The cells grew in defined dextrose medium and
the constructs were integrated at `URA3` under Zeocin selection.

The digitised median values span:

| Panel | Weakest | Strongest | Max/min | Log10 span |
| --- | ---: | ---: | ---: | ---: |
| mRuby2 | 1.53× | 101× | 66.0× | 1.820 decades |
| Venus | 1.70× | 818× | 481.2× | 2.682 decades |
| Geometric-mean consensus | 1.77× | 287.4× | 162.1× | 2.210 decades |

The benchmark uses the medians as labels. The digitised minimum and maximum
replicate values remain in the task data for the Figure 3A-style plot.

## Results

Zero-shot Modal A10 run, 2026-08-25. Both models scored all 38 constructs.
The cross-model comparison is a 2 × 2 set of log–log scatter plots: one
column per model and one row per reporter CDS. Each panel plots the raw,
untransformed, unbinned predicted CDS coverage sum against measured fluorescence;
horizontal error bars show the digitised Figure 3A fluorescence range.

| Consensus metric | Shorkie | Yorzoi |
| --- | ---: | ---: |
| Predicted max/min | **2.315×** | 1.739× |
| Dynamic-range recovery | **0.1650** | 0.1088 |
| Range fidelity | **0.1650** | 0.1088 |
| Spearman ρ | **0.6526** | 0.5702 |
| Log10 Pearson *r* | **0.6660** | 0.5358 |

Both models recover some promoter ordering but compress the 162.079× observed
consensus range into less than 2.4×. Shorkie recovers more of the span and ranks
the promoters better.

| Reporter | Observed range | Shorkie predicted / recovery / ρ | Yorzoi predicted / recovery / ρ |
| --- | ---: | ---: | ---: |
| mRuby2 | 66.013× | **2.627× / 0.2306 / 0.5246** | 1.875× / 0.1501 / 0.4667 |
| Venus | 481.176× | **2.187× / 0.1267 / 0.7561** | 2.151× / 0.1240 / 0.4632 |

## Construct reconstruction

The paper and its supporting files do not contain the final 38 plasmid records.
v1 reconstructs them from the [official ACS sequence archive](https://acs.figshare.com/articles/dataset/A_Highly_Characterized_Yeast_Toolkit_for_Modular_Multipart_Assembly/2054064)
and the assembly rules in the [official supporting information](https://acs.figshare.com/articles/journal_contribution/A_Highly_Characterized_Yeast_Toolkit_for_Modular_Multipart_Assembly/2054070):

```text
URA3 5′ arm — barcode — ConLS — promoter — reporter — tTDH1 — ConRE
             — ZeocinR — barcode — URA3 3′ arm
```

The variable promoters are pYTK009–pYTK027. Venus is pYTK033, mRuby2 is
pYTK034, tTDH1 is pYTK056, and ZeocinR is pYTK080. Golden Gate assembly keeps
one copy of each matching four-base junction. The processed payload replaces
R64-1-1 chromosome V bases 115946–117045 (1-based, inclusive), the interval
between the inner edges of the two `URA3` homology arms.

tTDH1 is the standard terminator used with the toolkit's fluorescent reporters
and is shown in the paper's reporter designs, but the final promoter-test
plasmids were not released. This part choice and the connector layout are
therefore a documented v1 reconstruction, not a claim that the final plasmid
sequences were directly observed.

Run the deterministic build with:

```bash
python scripts/ytk/build_distribution.py \
  --input-dir tmp-ytk-implementation \
  --reference-fasta data/tasks/R64-1-1.fa \
  --output-dir data/tasks/ytk_promoter
```

The build checks every BsaI overhang, aligns both homology arms, checks each CDS
against its assembled payload, and records hashes for all raw inputs.

## Exact CDS readout

YTK Type-3 reporters omit their start codon and stop codon. The Type-2/Type-3
`TATG` junction supplies `ATG`; the Type-3 part then supplies the annotated
reporter body and a `GGATCC` Gly-Ser linker. The Type-4 terminator supplies the
stop codon.

This task defines the reporter CDS as the junction `ATG` plus the annotated
reporter body. It excludes the linker and the terminator-supplied stop:

| Reporter | Exact readout length |
| --- | ---: |
| Venus | 714bp |
| mRuby2 | 711bp |

For promoter (p) and reporter (r), the adapter returns

\[
s_{p,r}=\sum_{b\in\mathrm{CDS}_{p,r}} \widehat{C}_{p,r,b},
\]

where \(\widehat{C}\) is raw predicted per-base RNA coverage. Shorkie uses the
mean of its T0 RNA-seq tracks. Yorzoi applies its nonlinear inverse transform to
each track and model pass before reverse-complement averaging, unbins to bases,
and then averages the 81 plus-strand tracks. Neither adapter sums transformed
values or whole model bins.

## Metrics

For a panel of 19 positive scores, define its fold range and span as

\[
F=\frac{\max_i x_i}{\min_i x_i}, \qquad D=\log_{10}F.
\]

The main diagnostic is

\[
R_{\mathrm{range}}=\frac{D_{\mathrm{pred}}}{D_{\mathrm{obs}}}.
\]

`0` means complete collapse, `1` means the observed number of decades was
recovered, and values above `1` mean an exaggerated range. The bounded range
fidelity

\[
Q_{\mathrm{range}}=\min(R_{\mathrm{range}},1/R_{\mathrm{range}})
\]

has its optimum at `1` and falls towards `0` for either compression or
exaggeration. The output also reports absolute span error in decades, Spearman
ρ, and Pearson *r* on log10 values.

The headline consensus is computed per promoter as the geometric mean of the
two reporter scores. Reporter-specific multiplicative scales cancel from all
range and correlation metrics:

\[
s_{p,\mathrm{consensus}}=\sqrt{s_{p,\mathrm{mRuby2}}s_{p,\mathrm{Venus}}}.
\]

The same operation is applied to the measured fluorescence labels. mRuby2 and
Venus results stay visible so disagreement between reporters is not hidden.

## Limits and v2 work

- Fluorescence is a protein-level readout; the models predict RNA coverage.
  Rank and dynamic-range agreement can still be useful, but absolute units do
  not match.
- The experimental background is non-fluorescent-cell signal, not an RNA
  quantity. The model scores therefore have no promoterless-background
  subtraction.
- v1 treats the digitised Figure 3A values as truth. For v2, ask the authors for
  the original per-replicate values and rebuild the labels from those data.
- For v2, also ask for the final promoter-test plasmid sequences and replace the
  documented tTDH1/connector reconstruction if they differ.
- Max/min is deliberately sensitive to the two endpoint promoters. This is the
  chosen definition because endpoint compression is the biological question.
- The experiment used BY4741, while sequence models use R64-1-1. The task keeps
  native R64-1-1 homology-arm sequence and replaces the native `URA3` interval
  with the reconstructed payload.

## Runtime data

| File | Contents |
| --- | --- |
| `figure3a.tsv` | Digitised median and replicate range for 19 promoters and two reporters. |
| `constructs.tsv` | Construct identity, URA3 replacement coordinates, payload length, and exact reporter CDS span. |
| `constructs.fasta` | The 38 post-integration replacement payloads. |

The official supporting files are licensed CC BY-NC 4.0. The processed task is
declared under the same licence in the data manifest.
