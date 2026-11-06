"""Yorzoi adapter for the Chen synonymous-mutation MPRA benchmark.

Marginalised version — see ``shorkie_chen_marginalized.py`` for the
motivation. Differs from the Shorkie analog in the model geometry and
in **strand-aware track aggregation**: a + strand host uses Yorzoi's
plus-strand tracks [0:81]; a − strand host uses [81:162].

One adapter serves all Chen libraries: the per-library host contexts +
REF caches (a "bundle") are built on demand and cached, so calling for a
given library reproduces exactly what a single-library run would compute.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np
from tqdm import tqdm

from yeastbench.adapters._chen_marginalized import (
    VAR_LEN, ChenHostContext, alt_block_oh, build_cassette, build_host_contexts,
    load_hosts,
)
from yeastbench.adapters._genome import one_hot_encode_channels_first
from yeastbench.adapters._yorzoi_constants import (
    BIN_WIDTH,
    CROP_BP_EACH_SIDE,
    OUTPUT_BINS,
    SEQ_LEN,
)
from yeastbench.adapters.protocols import LocalCodingVariantPredictor
from yeastbench.models.yorzoi import Yorzoi

if TYPE_CHECKING:
    import torch

log = logging.getLogger(__name__)


@dataclass
class _Bundle:
    """Per-library precomputed state (host contexts + REF caches)."""
    contexts: list[ChenHostContext]
    ref_oh_gpu: "torch.Tensor"          # (n_hosts, SEQ_LEN, 4)
    ref_log: np.ndarray                 # (n_hosts,) log2(ref_sum + 1)
    track_slices: list[tuple[int, int]]  # per-host strand-matched (lo, hi)


class YorzoiChenPredictor(LocalCodingVariantPredictor):
    def __init__(
        self,
        model: Yorzoi,
        fasta_path: str | Path,
        hosts_path: str | Path,
        data_dir: str | Path,
        batch_size: int = 8,
    ) -> None:
        import pysam

        self.model = model
        self.batch_size = batch_size

        self.fasta = pysam.FastaFile(str(fasta_path))
        self.hosts = load_hosts(hosts_path)
        self.data_dir = Path(data_dir)
        self._bundles: dict[str, _Bundle] = {}

    @classmethod
    def from_pretrained(
        cls,
        hf_repo: str,
        fasta_path: str | Path,
        hosts_path: str | Path,
        data_dir: str | Path,
        device: str = "cuda",
        batch_size: int = 8,
        use_rc: bool = True,
        autocast: bool = True,
        hf_revision: str | None = None,
    ) -> "YorzoiChenPredictor":
        return cls(
            Yorzoi.from_pretrained(
                hf_repo, device=device, use_rc=use_rc, autocast=autocast,
                hf_revision=hf_revision,
            ),
            fasta_path=fasta_path,
            hosts_path=hosts_path,
            data_dir=data_dir,
            batch_size=batch_size,
        )

    # ── Per-library bundle ────────────────────────────────────────────

    def _bundle(self, library: str) -> _Bundle:
        bundle = self._bundles.get(library)
        if bundle is not None:
            return bundle
        import torch as _torch

        cassette = build_cassette(library, self.fasta, self.data_dir)
        contexts = build_host_contexts(
            library=library, hosts=self.hosts, fasta=self.fasta,
            cassette=cassette,
            seq_len=SEQ_LEN, crop_bp_each_side=CROP_BP_EACH_SIDE,
            bin_width=BIN_WIDTH, output_bins=OUTPUT_BINS,
        )
        log.info("%s: built %d host contexts (Yorzoi geometry)",
                 library, len(contexts))

        n = len(contexts)
        ref_np = np.zeros((n, SEQ_LEN, 4), dtype=np.float32)
        for i, ctx in enumerate(contexts):
            ref_np[i] = one_hot_encode_channels_first(ctx.window_seq).T
        ref_oh_gpu = _torch.from_numpy(ref_np).to(self.model.device)

        # Per-host strand-matched track index slice.
        track_slices = [
            (0, 81) if c.host.strand == "+" else (81, 162)
            for c in contexts
        ]

        ref_exon_sums = self._predict_exon_sums(
            ref_oh_gpu, contexts, track_slices,
        ).cpu().numpy()
        ref_log = np.log2(ref_exon_sums + 1.0)
        log.info(
            "%s: Yorzoi REF exon-sums per host (min / max / median): %.2f / %.2f / %.2f",
            library,
            float(ref_exon_sums.min()), float(ref_exon_sums.max()),
            float(np.median(ref_exon_sums)),
        )
        bundle = _Bundle(
            contexts=contexts, ref_oh_gpu=ref_oh_gpu, ref_log=ref_log,
            track_slices=track_slices,
        )
        self._bundles[library] = bundle
        return bundle

    # ── Internals ────────────────────────────────────────────────────

    def _predict_exon_sums(
        self,
        batch_oh: "torch.Tensor",
        contexts: list[ChenHostContext],
        track_slices: list[tuple[int, int]],
    ) -> "torch.Tensor":
        """``batch_oh`` shape (B, SEQ_LEN, 4). B must be a multiple of n_hosts;
        row i maps to host ``i % n_hosts``. Returns (B,) with the
        strand-matched cross-track-mean CDS-bin sum per row."""
        import torch as _torch

        n = len(contexts)
        B = batch_oh.shape[0]
        if B % n != 0:
            raise ValueError(f"batch size {B} must be a multiple of n_hosts={n}")

        with _torch.no_grad():
            # Per-base raw counts: the Borzoi inverse is applied per pass
            # before RC-averaging inside the wrapper, then unbinned.
            pred = self.model.forward_tracks_perbase(batch_oh)  # (B, 162, OUT_BINS*BIN_WIDTH)

        out = _torch.zeros(B, device=pred.device, dtype=pred.dtype)
        for h, ctx in enumerate(contexts):
            ts, te = track_slices[h]
            rows = _torch.arange(h, B, n, device=pred.device)
            # Sum over CDS base positions, mean across strand-matched tracks
            # (raw counts; the nonlinear inverse already applied upstream).
            cds = pred[rows][:, ts:te, ctx.cds_base_lo:ctx.cds_base_hi].sum(dim=2)  # (R, 81)
            out[rows] = cds.mean(dim=1)
        return out

    def _splice_batch(
        self, variant_seqs: Sequence[str], bundle: _Bundle,
    ) -> "torch.Tensor":
        """Returns (B_variants * n_hosts, SEQ_LEN, 4) one-hot tensor."""
        import torch as _torch

        B = len(variant_seqs)
        contexts = bundle.contexts
        n = len(contexts)
        alt = bundle.ref_oh_gpu.unsqueeze(0).expand(B, -1, -1, -1).reshape(B * n, SEQ_LEN, 4).clone()

        for v_idx, seq in enumerate(variant_seqs):
            seq_upper = seq.upper()
            if len(seq_upper) != VAR_LEN:
                raise ValueError(f"variant_seq {v_idx} has {len(seq_upper)} nt")
            fwd = _torch.from_numpy(alt_block_oh(seq_upper, needs_revcomp=False).T).to(self.model.device)
            rc  = _torch.from_numpy(alt_block_oh(seq_upper, needs_revcomp=True).T).to(self.model.device)
            for h_idx, ctx in enumerate(contexts):
                row = v_idx * n + h_idx
                s = ctx.var_start_in_window
                alt[row, s : s + VAR_LEN, :] = rc if ctx.var_needs_revcomp else fwd
        return alt

    def _score_library(
        self, variant_seqs: Sequence[str], bundle: _Bundle, library: str,
    ) -> np.ndarray:
        import torch as _torch

        n_variants = len(variant_seqs)
        n_hosts = len(bundle.contexts)
        scores = np.empty(n_variants, dtype=np.float64)
        ref_log_t = _torch.from_numpy(bundle.ref_log).to(self.model.device).to(_torch.float32)

        for batch_start in tqdm(
            range(0, n_variants, self.batch_size),
            desc=f"Yorzoi Chen marginalized {library}",
        ):
            batch_end = min(batch_start + self.batch_size, n_variants)
            batch = variant_seqs[batch_start:batch_end]
            B = len(batch)

            alt_batch = self._splice_batch(batch, bundle)
            alt_sums = self._predict_exon_sums(
                alt_batch, bundle.contexts, bundle.track_slices,
            )
            alt_sums = alt_sums.view(B, n_hosts)
            alt_log = _torch.log2(alt_sums + 1.0)
            logsed = (alt_log - ref_log_t.unsqueeze(0))
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

        # Group by library (stable order); a single-library call (the
        # benchmark's path) is one group in original order, bit-identical to a
        # standalone run.
        groups: dict[str, list[int]] = {}
        for i, lib in enumerate(library_ids):
            groups.setdefault(lib, []).append(i)

        for library, idxs in groups.items():
            bundle = self._bundle(library)
            sub_scores = self._score_library(
                [variant_seqs[i] for i in idxs], bundle, library,
            )
            for j, i in enumerate(idxs):
                scores[i] = sub_scores[j]
        return scores
