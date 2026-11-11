"""Yorzoi GFP RNA scores with per-locus reporter-strand routing."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Mapping, Sequence

import numpy as np
from tqdm import tqdm

from yeastbench.adapters._genome import one_hot_encode_channels_first
from yeastbench.adapters._gui_scaffold import (
    GuiLocus,
    build_gui_contexts,
    compute_gui_context_metadata,
    load_gui_constructs,
)
from yeastbench.adapters._yorzoi_constants import (
    BIN_WIDTH,
    CROP_BP_EACH_SIDE,
    OUTPUT_BINS,
    SEQ_LEN,
    YORZOI_MINUS_TRACK_IDS,
    YORZOI_PLUS_TRACK_IDS,
)
from yeastbench.adapters.protocols import GuiReporterExpressionPredictor

if TYPE_CHECKING:
    from yeastbench.models.yorzoi import Yorzoi


class YorzoiGuiPredictor(GuiReporterExpressionPredictor):
    def __init__(
        self,
        model: Yorzoi,
        fasta_path: str | Path,
        constructs_path: str | Path,
        batch_size: int = 32,
    ) -> None:
        import pysam

        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.model = model
        self.constructs = load_gui_constructs(constructs_path)
        self.fasta = pysam.FastaFile(str(fasta_path))
        self.batch_size = batch_size
        self._prediction_exclusions: dict[str, str] = {}
        self._prediction_metadata: dict[str, dict[str, int | str]] = {}

    @property
    def prediction_exclusions(self) -> Mapping[str, str]:
        return self._prediction_exclusions

    @property
    def prediction_metadata(self) -> Mapping[str, Mapping[str, int | str]]:
        """Per-locus window hashes, CDS coverage and visible native flanks."""
        return self._prediction_metadata

    @classmethod
    def from_pretrained(
        cls,
        hf_repo: str,
        fasta_path: str | Path,
        constructs_path: str | Path,
        device: str = "cuda",
        batch_size: int = 32,
        use_rc: bool = True,
        autocast: bool = True,
        hf_revision: str | None = None,
    ) -> YorzoiGuiPredictor:
        from yeastbench.models.yorzoi import Yorzoi

        return cls(
            Yorzoi.from_pretrained(
                hf_repo, device=device, use_rc=use_rc, autocast=autocast,
                hf_revision=hf_revision,
            ),
            fasta_path=fasta_path, constructs_path=constructs_path, batch_size=batch_size,
        )

    def predict_expressions(self, loci: Sequence[GuiLocus]) -> np.ndarray:
        import torch

        scores = np.full(len(loci), np.nan, dtype=np.float64)
        contexts, self._prediction_exclusions = build_gui_contexts(
            loci, self.constructs, self.fasta,
            SEQ_LEN, CROP_BP_EACH_SIDE, BIN_WIDTH, OUTPUT_BINS,
        )
        self._prediction_metadata = {
            locus.locus_id: compute_gui_context_metadata(locus, context, len(self.constructs[locus.construct_id]))
            for _, locus, context in contexts
        }
        for batch_start in tqdm(range(0, len(contexts), self.batch_size), desc="Yorzoi Gui GFP RNA"):
            batch = contexts[batch_start:batch_start + self.batch_size]
            sequences = np.stack([
                one_hot_encode_channels_first(context.window_seq).T for _, _, context in batch
            ])
            inputs = torch.from_numpy(sequences).to(self.model.device)
            with torch.no_grad():
                # Inverse transform is applied before RC averaging by the
                # wrapper; these are raw per-base counts, already strand-aligned.
                coverage = self.model.forward_tracks_perbase(inputs)
            if tuple(coverage.shape) != (len(batch), 162, OUTPUT_BINS * BIN_WIDTH):
                raise ValueError("Unexpected Yorzoi per-base output shape")
            for batch_index, (row_index, locus, context) in enumerate(batch):
                base_indices = torch.as_tensor(
                    context.readout_base_positions, device=self.model.device, dtype=torch.long,
                )
                track_ids = YORZOI_PLUS_TRACK_IDS if locus.cassette_strand == "+" else YORZOI_MINUS_TRACK_IDS
                track_indices = torch.as_tensor(track_ids, device=self.model.device, dtype=torch.long)
                reporter = coverage[batch_index].index_select(0, track_indices).index_select(1, base_indices)
                score = float(reporter.sum(dim=1).mean().item())
                if np.isfinite(score) and score >= 0:
                    scores[row_index] = score
                else:
                    self._prediction_exclusions[locus.locus_id] = "invalid_model_score"
        return scores
