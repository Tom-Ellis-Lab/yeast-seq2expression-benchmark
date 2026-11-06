"""Shorkie adapter for the Lee et al. YTK promoter panel."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from tqdm import tqdm

from yeastbench.adapters._genome import one_hot_encode_channels_first
from yeastbench.adapters._shorkie_constants import (
    BIN_WIDTH,
    CROP_BP_EACH_SIDE,
    OUTPUT_BINS,
    SEQ_LEN,
    SHORKIE_T0_RNA_SEQ_TRACK_IDS,
)
from yeastbench.adapters._ytk_scaffold import YTKConstruct, build_context
from yeastbench.adapters.protocols import IntegratedPromoterPanelPredictor
from yeastbench.models.shorkie import Shorkie


class ShorkieYTKPredictor(IntegratedPromoterPanelPredictor):
    def __init__(
        self,
        model: Shorkie,
        fasta_path: str | Path,
        track_subset: list[int] = SHORKIE_T0_RNA_SEQ_TRACK_IDS,
        batch_size: int = 16,
    ) -> None:
        import pysam
        import torch

        self.model = model
        self.fasta = pysam.FastaFile(str(fasta_path))
        self.track_subset = list(track_subset)
        self.batch_size = batch_size
        self._track_idx_t = torch.tensor(
            self.track_subset, device=self.model.device, dtype=torch.long
        )

    @classmethod
    def from_checkpoints(
        cls,
        params_path: str | Path,
        checkpoint_paths: Sequence[str | Path],
        fasta_path: str | Path,
        track_subset: list[int] = SHORKIE_T0_RNA_SEQ_TRACK_IDS,
        device: str = "cuda",
        batch_size: int = 16,
        use_rc: bool = True,
    ) -> "ShorkieYTKPredictor":
        return cls(
            Shorkie.from_checkpoints(
                params_path,
                checkpoint_paths,
                device=device,
                use_rc=use_rc,
            ),
            fasta_path=fasta_path,
            track_subset=list(track_subset),
            batch_size=batch_size,
        )

    def predict_reporter_expressions(
        self,
        constructs: Sequence[YTKConstruct],
    ) -> np.ndarray:
        import torch

        scores = np.full(len(constructs), np.nan, dtype=np.float64)
        contexts: list[tuple[int, object]] = []
        for index, construct in enumerate(constructs):
            context = build_context(
                construct,
                self.fasta,
                SEQ_LEN,
                CROP_BP_EACH_SIDE,
                BIN_WIDTH,
                OUTPUT_BINS,
            )
            if context is not None and context.readout_base_positions.size:
                contexts.append((index, context))

        for start in tqdm(
            range(0, len(contexts), self.batch_size),
            desc="Shorkie YTK",
        ):
            batch = contexts[start : start + self.batch_size]
            x = torch.from_numpy(
                np.stack(
                    [
                        one_hot_encode_channels_first(context.window_seq)
                        for _, context in batch
                    ]
                )
            ).to(self.model.device)
            with torch.no_grad():
                # The wrapper returns raw, unbinned per-base coverage. Sum
                # only the exact reporter CDS bases; never score model bins.
                coverage = self.model.forward_track_mean_perbase(x, self._track_idx_t)
            for batch_index, (row_index, context) in enumerate(batch):
                base_index = torch.as_tensor(
                    context.readout_base_positions,
                    device=self.model.device,
                    dtype=torch.long,
                )
                scores[row_index] = float(
                    coverage[batch_index].index_select(0, base_index).sum().item()
                )
        return scores
