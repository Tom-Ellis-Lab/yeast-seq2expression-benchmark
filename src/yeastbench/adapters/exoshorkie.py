"""Task adapters for individual distilled ExoShorkie students.

Scaffolds own experimental sequence construction. The model owns decoding,
nonnegative coverage, strand alignment, bin geometry and region sums.
"""

from __future__ import annotations

import numpy as np
from tqdm import tqdm

from yeastbench.adapters._chen_predictor import ChenMarginalizedBase
from yeastbench.adapters._marginalized_mpra import MPRAMarginalizedBase
from yeastbench.adapters._shalem_scaffold import ShalemMarginalizedBase
from yeastbench.adapters._variant_scorer import VariantScorerBase


def _geometry(model):
    return dict(
        seq_len=model.seq_len,
        crop_bp_each_side=model.crop_bp_each_side,
        bin_width=model.bin_width,
        output_bins=model.output_bins,
    )


_DNA_ENCODING = np.zeros((256, 4), dtype=np.float32)
_DNA_ENCODING[np.frombuffer(b"ACGT", dtype=np.uint8)] = np.eye(4, dtype=np.float32)


def _one_hot(seq):
    """Vectorized equivalent of the shared uppercase ACGT encoder."""
    return _DNA_ENCODING[np.frombuffer(seq.encode("ascii"), dtype=np.uint8)].T


def _encode(sequences, model):
    import torch

    if any(len(seq) != model.seq_len for seq in sequences):
        raise ValueError(f"ExoShorkie inputs must have length {model.seq_len}")
    return torch.from_numpy(
        np.stack([_one_hot(seq) for seq in sequences])
    ).to(model.device)


class ExoShorkieCoverage:
    """Bridge the marginalized engine to model-owned binned region readouts."""

    def __init__(self, model):
        self.model = model
        self.device = model.device

    def one_hot(self, seq):
        return _one_hot(seq)

    def splice(self, alt, row, start, block_oh, length):
        alt[row, :, start : start + length] = block_oh

    def splice_batch(self, alt, start, blocks, length):
        alt[:, :, start : start + length] = blocks

    def forward(self, batch_oh):
        return self.model.forward_strands_binned(batch_oh)

    def readout(self, out, row, bins_t, strand):
        return self.model.sum_region(out[row], bins_t, strand)

    def readout_batch(self, out, bins_t, strand):
        return self.model.sum_region(out, bins_t, strand)


class ExoShorkieMPRAPredictor(MPRAMarginalizedBase):
    def __init__(
        self, model, fasta_path, gtf_path, batch_size=8, n_sample=None, seed=42
    ):
        self._setup(
            model=model,
            cov=ExoShorkieCoverage(model),
            fasta_path=fasta_path,
            gtf_path=gtf_path,
            **_geometry(model),
            batch_size=batch_size,
            n_sample=n_sample,
            seed=seed,
            desc=f"ExoShorkie {model.variant} marginalized",
        )


class ExoShorkieShalemPredictor(ShalemMarginalizedBase):
    def __init__(
        self,
        model,
        fasta_path,
        gtf_path,
        batch_size=8,
        host_genes_json=None,
        n_sample=None,
        seed=42,
    ):
        self._setup(
            model=model,
            cov=ExoShorkieCoverage(model),
            fasta_path=fasta_path,
            gtf_path=gtf_path,
            host_genes_json=host_genes_json,
            **_geometry(model),
            batch_size=batch_size,
            n_sample=n_sample,
            seed=seed,
            desc=f"ExoShorkie {model.variant} Shalem",
        )


class ExoShorkieVariantScorer(VariantScorerBase):
    def __init__(self, model, fasta_path, gtf_path, batch_size=8):
        super().__init__(model, fasta_path, gtf_path, batch_size)
        self.__dict__.update(_geometry(model))

    def score_variants(self, variants):
        import torch

        jobs = self._prepare_jobs(variants)
        scores = np.empty(len(jobs), dtype=np.float64)
        for start in range(0, len(jobs), self.batch_size):
            batch = jobs[start : start + self.batch_size]
            pairs = [self._build_one_hot_pair(job) for job in batch]
            x = torch.from_numpy(
                np.stack([ref for ref, _ in pairs] + [alt for _, alt in pairs])
            ).to(self.model.device)
            coverage = self.model.forward_strands_binned(x)
            for i, job in enumerate(batch):
                ref = self.model.sum_region(
                    coverage[i], job.base_idx, job.strand
                ).item()
                alt = self.model.sum_region(
                    coverage[i + len(batch)],
                    job.base_idx,
                    job.strand,
                ).item()
                scores[start + i] = np.log2(alt + 1) - np.log2(ref + 1)
        return scores


class ExoShorkieChenPredictor(ChenMarginalizedBase):
    def __init__(self, model, fasta_path, hosts_path, data_dir, batch_size=4):
        super().__init__(model, fasta_path, hosts_path, data_dir, batch_size)
        self.__dict__.update(_geometry(model))
        self.description = f"ExoShorkie {model.variant}"

    def _predict_exon_sums(self, batch_oh, contexts):
        import torch

        n = len(contexts)
        if not n or batch_oh.shape[0] % n:
            raise ValueError("batch size must be a multiple of the nonzero host count")
        # Chen batches variants × hosts; chunk model forwards independently so
        # a normal task batch size does not multiply GPU memory by 20 hosts.
        coverage = torch.cat(
            [
                self.model.forward_strands_binned(
                    batch_oh[start : start + self.batch_size]
                )
                for start in range(0, len(batch_oh), self.batch_size)
            ]
        )
        sums = coverage.new_empty(len(batch_oh))
        for h, ctx in enumerate(contexts):
            bases = torch.arange(
                ctx.cds_base_lo, ctx.cds_base_hi, device=coverage.device
            )
            sums[h::n] = self.model.sum_region(coverage[h::n], bases, ctx.host.strand)
        return sums


class _ReporterPredictor:
    """Streaming context batches; subclasses specify the experimental readout."""

    def __init__(self, model, fasta_path, batch_size=8):
        import pysam

        self.model = model
        self.fasta = pysam.FastaFile(str(fasta_path))
        self.batch_size = batch_size

    def _score_contexts(self, contexts, n, description):
        scores = np.full(n, np.nan, dtype=np.float64)

        def flush(batch):
            values = self.model.predict_region_sums(
                _encode([ctx.window_seq for _, ctx, _, _ in batch], self.model),
                [bases for _, _, bases, _ in batch],
                [strand for _, _, _, strand in batch],
            )
            for j, (index, _, bases, _) in enumerate(batch):
                if len(bases):
                    scores[index] = values[j].item()

        batch = []
        for item in tqdm(contexts, total=n, desc=description):
            batch.append(item)
            if len(batch) == self.batch_size:
                flush(batch)
                batch = []
        if batch:
            flush(batch)
        return scores


class ExoShorkieWuPredictor(_ReporterPredictor):
    def __init__(
        self,
        model,
        fasta_path,
        gtf_path,
        batch_size=8,
        cassette_fasta=None,
        barcodes_path=None,
    ):
        from yeastbench.adapters._wu_scaffold import (
            DEFAULT_BARCODES_TSV,
            DEFAULT_CASSETTE_FASTA,
            load_barcodes,
            load_cassette_payload,
        )

        super().__init__(model, fasta_path, batch_size)
        # Loci already carry the gene annotations resolved by the benchmark.
        self.payload = load_cassette_payload(cassette_fasta or DEFAULT_CASSETTE_FASTA)
        self.barcodes = load_barcodes(barcodes_path or DEFAULT_BARCODES_TSV)

    def predict_expressions(self, loci):
        from yeastbench.adapters._wu_scaffold import (
            build_insertion_context,
            inject_barcodes,
        )

        def contexts():
            for i, locus in enumerate(loci):
                payload = inject_barcodes(self.payload, *self.barcodes[locus.gene_id])
                ctx = build_insertion_context(
                    locus, payload, self.fasta, **_geometry(self.model)
                )
                if ctx is not None:
                    yield i, ctx, ctx.rfp_base_positions, locus.strand

        return self._score_contexts(contexts(), len(loci), "ExoShorkie Wu")


class ExoShorkieHongPredictor(_ReporterPredictor):
    def __init__(self, model, fasta_path, batch_size=8, cassette_fasta=None):
        from yeastbench.adapters._hong_scaffold import (
            DEFAULT_CASSETTE_FASTA,
            load_cassette_payload,
        )

        super().__init__(model, fasta_path, batch_size)
        self.payload = load_cassette_payload(cassette_fasta or DEFAULT_CASSETTE_FASTA)

    def predict_expressions(self, loci):
        from yeastbench.adapters._hong_scaffold import build_insertion_context

        def contexts():
            for i, locus in enumerate(loci):
                ctx = build_insertion_context(
                    locus,
                    self.payload,
                    self.fasta,
                    **_geometry(self.model),
                )
                if ctx is not None:
                    yield i, ctx, ctx.mcherry_base_positions, "+"

        # The cassette is always integrated on the genomic + strand.
        return self._score_contexts(contexts(), len(loci), "ExoShorkie Hong")

    def predict_diagnostic_readouts(self, loci):
        from yeastbench.adapters._hong_scaffold import (
            DIAGNOSTIC_REGION_NAMES,
            build_insertion_context,
            readout_region_base_positions,
        )

        # One RNA-seq output, so the candidate space varies only by region.
        readouts = {
            f"RNA-seq + × {name}": np.full(len(loci), np.nan)
            for name in DIAGNOSTIC_REGION_NAMES
        }
        for start in range(0, len(loci), self.batch_size):
            batch = []
            for i in range(start, min(start + self.batch_size, len(loci))):
                ctx = build_insertion_context(
                    loci[i],
                    self.payload,
                    self.fasta,
                    **_geometry(self.model),
                )
                if ctx is not None:
                    batch.append((i, ctx))
            if not batch:
                continue
            coverage = self.model.forward_strands_binned(
                _encode([ctx.window_seq for _, ctx in batch], self.model),
            )
            for j, (i, ctx) in enumerate(batch):
                regions = readout_region_base_positions(
                    ctx,
                    self.model.crop_bp_each_side,
                    self.model.output_bins * self.model.bin_width,
                )
                for name, bases in regions.items():
                    if len(bases):
                        readouts[f"RNA-seq + × {name}"][i] = self.model.sum_region(
                            coverage[j],
                            bases,
                            "+",
                        ).item()
        readouts["primary"] = readouts["RNA-seq + × cassette CDS"]
        return readouts


class ExoShorkieYTKPredictor(_ReporterPredictor):
    def predict_reporter_expressions(self, constructs):
        from yeastbench.adapters._ytk_scaffold import build_context

        def contexts():
            for i, construct in enumerate(constructs):
                ctx = build_context(construct, self.fasta, **_geometry(self.model))
                if ctx is not None:
                    yield i, ctx, ctx.readout_base_positions, construct.strand

        return self._score_contexts(contexts(), len(constructs), "ExoShorkie YTK")


class ExoShorkieCuperusPredictor(_ReporterPredictor):
    def __init__(self, model, fasta_path, batch_size=8, construct_json=None):
        from yeastbench.adapters._cuperus_scaffold import CuperusConstruct

        super().__init__(model, fasta_path, batch_size)
        self.construct = (
            CuperusConstruct.from_json(construct_json)
            if construct_json
            else CuperusConstruct.from_reference(self.fasta)
        )

    def predict_utr_expressions(self, utrs):
        from yeastbench.adapters._cuperus_scaffold import HIS3_BACKGROUND, build_context

        def contexts():
            for i, utr in enumerate(utrs):
                ctx = build_context(
                    self.construct,
                    utr,
                    HIS3_BACKGROUND,
                    self.fasta,
                    **_geometry(self.model),
                )
                if ctx is not None:
                    yield i, ctx, ctx.readout_base_positions, "+"

        return self._score_contexts(contexts(), len(utrs), "ExoShorkie Cuperus")


class ExoShorkieBrooksPredictor:
    varies_by_strain = False

    def __init__(self, model, batch_size=8):
        self.model = model
        self.batch_size = batch_size
        self.seq_len = model.seq_len
        self.crop_bp_each_side = model.crop_bp_each_side

    def predict_coverage_batch(self, seqs, strands, strains=None):
        import torch

        if len(seqs) != len(strands) or any(s not in ("+", "-") for s in strands):
            raise ValueError("Each sequence needs a '+' or '-' strand")
        if not len(seqs):
            return np.empty((0, self.seq_len - 2 * self.crop_bp_each_side))
        coverage = self.model.forward_strands_perbase(_encode(seqs, self.model))
        strand_ids = torch.tensor(
            [0 if s == "+" else 1 for s in strands], device=coverage.device
        )
        return (
            coverage[torch.arange(len(seqs), device=coverage.device), strand_ids]
            .cpu()
            .numpy()
        )


class ExoShorkieMeneuPredictor(ExoShorkieBrooksPredictor):
    def predict_coverage_batch(self, seqs, strands, strains=None):
        if not len(seqs):
            return np.empty((0, self.seq_len - 2 * self.crop_bp_each_side))
        return (
            self.model.forward_unstranded_perbase(_encode(seqs, self.model))
            .cpu()
            .numpy()
        )
