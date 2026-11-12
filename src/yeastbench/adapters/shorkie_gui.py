"""Shorkie GFP RNA scores for validated Gui replacement genotypes."""
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
from yeastbench.adapters._shorkie_constants import (
    BIN_WIDTH,
    CROP_BP_EACH_SIDE,
    OUTPUT_BINS,
    SEQ_LEN,
    SHORKIE_1000_RNA_SEQ_TRACK_IDS,
)
from yeastbench.adapters.protocols import GuiReporterExpressionPredictor

if TYPE_CHECKING:
    from yeastbench.models.shorkie import Shorkie


class ShorkieGuiPredictor(GuiReporterExpressionPredictor):
    def __init__(
        self,
        model: Shorkie,
        fasta_path: str | Path,
        constructs_path: str | Path,
        batch_size: int = 16,
        track_subset: Sequence[int] = SHORKIE_1000_RNA_SEQ_TRACK_IDS,
    ) -> None:
        import pysam
        import torch

        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not track_subset or any(track < 0 for track in track_subset):
            raise ValueError("RNA track indices must be nonempty and nonnegative")
        self.model = model
        self.constructs = load_gui_constructs(constructs_path)
        self.fasta = pysam.FastaFile(str(fasta_path))
        self.batch_size = batch_size
        self.track_subset = tuple(track_subset)
        self._track_indices = torch.tensor(self.track_subset, device=model.device, dtype=torch.long)
        self._prediction_exclusions: dict[str, str] = {}
        self._prediction_metadata: dict[str, dict[str, int | str]] = {}

    @property
    def prediction_exclusions(self) -> Mapping[str, str]:
        return self._prediction_exclusions

    @property
    def prediction_metadata(self) -> Mapping[str, Mapping[str, int | str]]:
        """Per-locus window hashes, CDS coverage and visible native flanks."""
        return self._prediction_metadata

    @property
    def readout_metadata(self) -> Mapping[str, object]:
        """Resolved RNA track selection and aggregation, for saved results."""
        return {
            "group": "1000-RNA-Seq" if self.track_subset == tuple(SHORKIE_1000_RNA_SEQ_TRACK_IDS) else "custom",
            "n_tracks": len(self.track_subset),
            "track_ids": list(self.track_subset),
            "strand": "unstranded",
            "aggregation": "arithmetic mean across tracks of linear RNA coverage summed over GFP CDS",
        }

    @classmethod
    def from_checkpoints(
        cls,
        params_path: str | Path,
        checkpoint_paths: Sequence[str | Path],
        fasta_path: str | Path,
        constructs_path: str | Path,
        device: str = "cuda",
        batch_size: int = 16,
        use_rc: bool = True,
        track_subset: Sequence[int] = SHORKIE_1000_RNA_SEQ_TRACK_IDS,
    ) -> ShorkieGuiPredictor:
        from yeastbench.models.shorkie import Shorkie

        return cls(
            Shorkie.from_checkpoints(params_path, checkpoint_paths, device=device, use_rc=use_rc),
            fasta_path=fasta_path, constructs_path=constructs_path,
            batch_size=batch_size, track_subset=track_subset,
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
        for batch_start in tqdm(range(0, len(contexts), self.batch_size), desc="Shorkie Gui GFP RNA"):
            batch = contexts[batch_start:batch_start + self.batch_size]
            sequences = np.stack([
                one_hot_encode_channels_first(context.window_seq) for _, _, context in batch
            ])
            inputs = torch.from_numpy(sequences).to(self.model.device)
            with torch.no_grad():
                # The wrapper unbins raw predictions and handles its ensemble
                # and RC averaging. Shorkie's RNA tracks are unstranded.
                coverage = self.model.forward_track_mean_perbase(inputs, self._track_indices)
            if tuple(coverage.shape) != (len(batch), OUTPUT_BINS * BIN_WIDTH):
                raise ValueError("Unexpected Shorkie per-base output shape")
            for batch_index, (row_index, locus, context) in enumerate(batch):
                base_indices = torch.as_tensor(
                    context.readout_base_positions, device=self.model.device, dtype=torch.long,
                )
                score = float(coverage[batch_index].index_select(0, base_indices).sum().item())
                if np.isfinite(score) and score >= 0:
                    scores[row_index] = score
                else:
                    self._prediction_exclusions[locus.locus_id] = "invalid_model_score"
        return scores
