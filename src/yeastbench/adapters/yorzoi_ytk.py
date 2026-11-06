"""Yorzoi adapter for the Lee et al. YTK promoter panel."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from tqdm import tqdm

from yeastbench.adapters._genome import one_hot_encode_channels_first
from yeastbench.adapters._yorzoi_constants import (
    BIN_WIDTH,
    CROP_BP_EACH_SIDE,
    OUTPUT_BINS,
    SEQ_LEN,
)
from yeastbench.adapters._ytk_scaffold import YTKConstruct, build_context
from yeastbench.adapters.protocols import IntegratedPromoterPanelPredictor
from yeastbench.models.yorzoi import Yorzoi


PLUS_TRACKS = (0, 81)


class YorzoiYTKPredictor(IntegratedPromoterPanelPredictor):
    def __init__(
        self,
        model: Yorzoi,
        fasta_path: str | Path,
        batch_size: int = 32,
    ) -> None:
        import pysam

        self.model = model
        self.fasta = pysam.FastaFile(str(fasta_path))
        self.batch_size = batch_size

    @classmethod
    def from_pretrained(
        cls,
        hf_repo: str,
        fasta_path: str | Path,
        device: str = "cuda",
        batch_size: int = 32,
        use_rc: bool = True,
        autocast: bool = True,
    ) -> "YorzoiYTKPredictor":
        return cls(
            Yorzoi.from_pretrained(
                hf_repo,
                device=device,
                use_rc=use_rc,
                autocast=autocast,
            ),
            fasta_path=fasta_path,
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
            desc="Yorzoi YTK",
        ):
            batch = contexts[start : start + self.batch_size]
            x = torch.from_numpy(
                np.stack(
                    [
                        one_hot_encode_channels_first(context.window_seq).T
                        for _, context in batch
                    ]
                )
            ).to(self.model.device)
            with torch.no_grad():
                # The wrapper applies the nonlinear inverse to each track and
                # pass before RC averaging, then unbins to raw per-base counts.
                perbase = self.model.forward_tracks_perbase(x)
            for batch_index, (row_index, context) in enumerate(batch):
                base_index = torch.as_tensor(
                    context.readout_base_positions,
                    device=self.model.device,
                    dtype=torch.long,
                )
                per_track = perbase[batch_index].index_select(1, base_index).sum(dim=1)
                start_track, end_track = PLUS_TRACKS
                scores[row_index] = float(
                    per_track[start_track:end_track].mean().item()
                )
        return scores
