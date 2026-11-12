"""Shared Chen context construction, splicing, reference caching and logSED scoring.

Subclasses supply only model-specific CDS readouts. Preserves the existing
variant-by-host batching and reference-log precision.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np
from tqdm import tqdm

from yeastbench.adapters._chen_marginalized import (
    VAR_LEN,
    ChenHostContext,
    alt_block_oh,
    build_cassette,
    build_host_contexts,
    load_hosts,
)
from yeastbench.adapters._genome import one_hot_encode_channels_first
from yeastbench.adapters.protocols import LocalCodingVariantPredictor

if TYPE_CHECKING:
    import torch
log = logging.getLogger(__name__)


@dataclass
class _Bundle:
    contexts: list[ChenHostContext]
    ref_oh_gpu: torch.Tensor
    ref_log: np.ndarray


class ChenMarginalizedBase(LocalCodingVariantPredictor):
    description = "Model"

    def __init__(self, model, fasta_path, hosts_path, data_dir, batch_size=4):
        import pysam

        self.model = model
        self.batch_size = batch_size
        self.fasta = pysam.FastaFile(str(fasta_path))
        self.hosts = load_hosts(hosts_path)
        self.data_dir = Path(data_dir)
        self._bundles = {}

    def _predict_exon_sums(self, batch_oh, contexts):
        raise NotImplementedError

    def _bundle(self, library: str) -> _Bundle:
        bundle = self._bundles.get(library)
        if bundle is not None:
            return bundle
        import torch as _torch

        cassette = build_cassette(library, self.fasta, self.data_dir)
        contexts = build_host_contexts(
            library=library,
            hosts=self.hosts,
            fasta=self.fasta,
            cassette=cassette,
            seq_len=self.seq_len,
            crop_bp_each_side=self.crop_bp_each_side,
            bin_width=self.bin_width,
            output_bins=self.output_bins,
        )
        log.info("%s: built %d host contexts (model geometry)", library, len(contexts))

        # Cache REF one-hots on GPU. Shape: (n_hosts, 4, self.seq_len).
        n = len(contexts)
        ref_np = np.zeros((n, 4, self.seq_len), dtype=np.float32)
        for i, ctx in enumerate(contexts):
            ref_np[i] = one_hot_encode_channels_first(ctx.window_seq)
        ref_oh_gpu = _torch.from_numpy(ref_np).to(self.model.device)

        ref_exon_sums = self._predict_exon_sums(ref_oh_gpu, contexts).cpu().numpy()
        ref_log = np.log2(ref_exon_sums + 1.0)
        log.info(
            "%s: REF exon-sums per host (min / max / median): %.2f / %.2f / %.2f",
            library,
            float(ref_exon_sums.min()),
            float(ref_exon_sums.max()),
            float(np.median(ref_exon_sums)),
        )
        bundle = _Bundle(contexts=contexts, ref_oh_gpu=ref_oh_gpu, ref_log=ref_log)
        self._bundles[library] = bundle
        return bundle

    def _splice_batch(
        self,
        variant_seqs: Sequence[str],
        bundle: _Bundle,
    ) -> "torch.Tensor":
        """Returns one-hot tensor of shape (B_variants * n_hosts, 4, self.seq_len)
        with each variant spliced into all 20 host REF windows."""
        import torch as _torch

        B = len(variant_seqs)
        contexts = bundle.contexts
        n = len(contexts)
        # Pre-compute per-variant ALT blocks: one-hot, RC where needed per host.
        # Resulting layout: per (var, host) pair, splice the variant block (RC'd
        # if host is - strand) at var_start_in_window.
        alt = (
            bundle.ref_oh_gpu.unsqueeze(0)
            .expand(B, -1, -1, -1)
            .reshape(B * n, 4, self.seq_len)
            .clone()
        )

        for v_idx, seq in enumerate(variant_seqs):
            seq_upper = seq.upper()
            if len(seq_upper) != VAR_LEN:
                raise ValueError(f"variant_seq {v_idx} has {len(seq_upper)} nt")
            # Build both fwd and rc one-hots once per variant (cheap, 36 nt).
            fwd = _torch.from_numpy(alt_block_oh(seq_upper, needs_revcomp=False)).to(
                self.model.device
            )
            rc = _torch.from_numpy(alt_block_oh(seq_upper, needs_revcomp=True)).to(
                self.model.device
            )
            for h_idx, ctx in enumerate(contexts):
                row = v_idx * n + h_idx
                s = ctx.var_start_in_window
                alt[row, :, s : s + VAR_LEN] = rc if ctx.var_needs_revcomp else fwd
        return alt

    def _score_library(
        self,
        variant_seqs: Sequence[str],
        bundle: _Bundle,
        library: str,
    ) -> np.ndarray:
        import torch as _torch

        n_variants = len(variant_seqs)
        n_hosts = len(bundle.contexts)
        scores = np.empty(n_variants, dtype=np.float64)
        ref_log_t = (
            _torch.from_numpy(bundle.ref_log).to(self.model.device).to(_torch.float32)
        )

        for batch_start in tqdm(
            range(0, n_variants, self.batch_size),
            desc=f"{self.description} Chen marginalized {library}",
        ):
            batch_end = min(batch_start + self.batch_size, n_variants)
            batch = variant_seqs[batch_start:batch_end]
            B = len(batch)

            alt_batch = self._splice_batch(
                batch, bundle
            )  # (B*n_hosts, 4, self.seq_len)
            alt_sums = self._predict_exon_sums(
                alt_batch, bundle.contexts
            )  # (B*n_hosts,)
            alt_sums = alt_sums.view(B, n_hosts)
            alt_log = _torch.log2(alt_sums + 1.0)
            logsed = alt_log - ref_log_t.unsqueeze(0)  # (B, n_hosts)
            scores[batch_start:batch_end] = logsed.mean(dim=1).cpu().numpy()

        return scores

    def predict_local_variants(
        self,
        library_ids: Sequence[str],
        variant_seqs: Sequence[str],
    ) -> np.ndarray:
        library_ids = list(library_ids)
        variant_seqs = list(variant_seqs)
        scores = np.empty(len(variant_seqs), dtype=np.float64)

        # Group by library (stable order) so each library is scored against its
        # own host contexts; a single-library call (the benchmark's path) is one
        # group in original order, bit-identical to a standalone run.
        groups: dict[str, list[int]] = {}
        for i, lib in enumerate(library_ids):
            groups.setdefault(lib, []).append(i)

        for library, idxs in groups.items():
            bundle = self._bundle(library)
            sub_scores = self._score_library(
                [variant_seqs[i] for i in idxs],
                bundle,
                library,
            )
            for j, i in enumerate(idxs):
                scores[i] = sub_scores[j]
        return scores
