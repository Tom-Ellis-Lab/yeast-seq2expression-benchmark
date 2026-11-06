"""CodonTransformer baseline for the Chen synonymous-mutation benchmark.

Loads Fallahpour et al. 2025's BigBird masked-LM (Nat Commun 16:3205;
https://github.com/Adibvafa/CodonTransformer), runs one forward pass per
library over an all-`*_unk` merged input (protein only, no codons), and
scores each variant by summing per-position log-probabilities over the
12 variable codons.

See ``docs/benchmarks/chen_synonymous.md`` for the math and caveats.

The convenience ``predict_dna_sequence()`` in the upstream package throws
away the logits we need, so we call the underlying HuggingFace model
directly. No upstream patch required.
"""
from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np
import torch

from yeastbench.adapters._chen_gfp_reference import GFP_PROTEIN
from yeastbench.adapters.protocols import LocalCodingVariantPredictor

log = logging.getLogger(__name__)

# TDH3 (YGR192C from S. cerevisiae R64-1-1), 332 aa.
TDH3_PROTEIN = (
    "MVRVAINGFGRIGRLVMRIALSRPNVEVVALNDPFITNDYAAYMFKYDSTHGRYAGEVSHDDKHIIVD"
    "GKKIATYQERDPANLPWGSSNVDIAIDSTGVFKELDTAQKHIDAGAKKVVITAPSSTAPMFVMGVNEE"
    "KYTSDLKIVSNASCTTNCLAPLAKVINDAFGIEEGLMTTVHSLTATQKTVDGPSHKDWRGGRTASGNI"
    "IPSSTGAAKAVGKVLPELQGKLTGMAFRVPTVDVSVVDLTVKLNKETTYDEIKKVVKAAAEGKLKGVL"
    "GYTEDAVVSSDFLGDSHSSIFDASAGIQLSPKFVKLVSWYDNEYGYSTRVVDLVEHVAKA"
)

# Per-library protein + 0-based slice [start, start+12) of the 12-codon
# variable region inside that protein, plus the published 12-aa peptide.
LIBRARY_CONTEXTS: dict[str, dict] = {
    "gfp_r1": dict(protein=GFP_PROTEIN, var_start=41, expected_peptide="LTLKFICTTGKL"),
    "gfp_r2": dict(protein=GFP_PROTEIN, var_start=156, expected_peptide="QKNGIKVNFKIR"),
    "tdh3":   dict(protein=TDH3_PROTEIN, var_start=56, expected_peptide="EVSHDDKHIIVD"),
}

ORGANISM = "Saccharomyces cerevisiae"


class CodonTransformerBaselinePredictor(LocalCodingVariantPredictor):
    """Score Chen synonymous variants under CodonTransformer's BigBird MLM.

    Load model + tokenizer once in ``from_task``. The per-library
    log-probabilities (one forward pass over that library's all-``*_unk``
    merged input) are built lazily on first use and cached, so one adapter
    serves all three libraries. ``predict_local_variants`` indexes into the
    cached per-position log-probabilities for each variant's 12 codons.
    """

    def __init__(
        self,
        *,
        model: "BigBirdForMaskedLM",          # noqa: F821 - lazy import type
        tokenizer: Any,
        token2index: dict[str, int],
        device: str | torch.device,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.token2index = token2index
        self.device = device
        # library_id -> (protein, var_start, log_p[L, vocab])
        self._cache: dict[str, tuple[str, int, np.ndarray]] = {}

    @classmethod
    def from_task(
        cls,
        task,
        device: str | torch.device = "cuda",
        hf_repo: str = "adibvafa/CodonTransformer",
        hf_revision: str | None = None,
        **_ignored,
    ) -> "CodonTransformerBaselinePredictor":
        from CodonTransformer.CodonUtils import TOKEN2INDEX
        from transformers import AutoTokenizer, BigBirdForMaskedLM
        from yeastbench.data.model_sources import resolve_model_source

        source = resolve_model_source("codon_transformer", hf_repo, hf_revision)
        log.info("loading CodonTransformer from %s", source)
        tokenizer = AutoTokenizer.from_pretrained(source)
        model = BigBirdForMaskedLM.from_pretrained(source)
        model = model.eval().to(device)
        return cls(
            model=model,
            tokenizer=tokenizer,
            token2index=dict(TOKEN2INDEX),
            device=device,
        )

    def _log_p_for(self, library_id: str) -> tuple[str, int, np.ndarray]:
        """Per-library per-position log-probabilities, built + cached on
        first use (one forward pass over the all-``*_unk`` merged input)."""
        cached = self._cache.get(library_id)
        if cached is not None:
            return cached

        from CodonTransformer.CodonData import get_merged_seq

        if library_id not in LIBRARY_CONTEXTS:
            raise ValueError(f"unknown Chen library: {library_id!r}")
        ctx = LIBRARY_CONTEXTS[library_id]
        protein, var_start, expected_peptide = (
            ctx["protein"], ctx["var_start"], ctx["expected_peptide"],
        )
        if protein[var_start : var_start + 12] != expected_peptide:
            raise ValueError(
                f"{library_id}: hard-coded protein slice "
                f"{protein[var_start:var_start+12]!r} != {expected_peptide!r}"
            )

        # Use the upstream helper to build the merged "protein + dna" string
        # (with dna="" → every codon collapses to "{aa}_unk"), so the token
        # format exactly matches what the tokenizer expects at inference.
        merged = get_merged_seq(protein=protein, dna="")
        inputs = self.tokenizer(
            merged, return_tensors="pt", padding=True, truncation=False,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs, return_dict=True).logits[0, 1:-1, :]
        log_p_all = torch.log_softmax(logits, dim=-1).cpu().numpy().astype(np.float64)
        # The merged-seq helper appends a stop-codon token (`__UNK`) so we
        # see len(protein)+1 non-special positions. Slice to the protein
        # positions only; we never index into the stop position.
        if log_p_all.shape[0] not in (len(protein), len(protein) + 1):
            raise ValueError(
                f"{library_id}: tokenizer returned {log_p_all.shape[0]} non-special "
                f"positions; expected {len(protein)} or {len(protein) + 1}"
            )
        result = (protein, var_start, log_p_all[: len(protein)])
        self._cache[library_id] = result
        return result

    def predict_local_variants(
        self,
        library_ids: Sequence[str],
        variant_seqs: Sequence[str],
    ) -> np.ndarray:
        out = np.empty(len(library_ids), dtype=float)
        for i, (lib, seq) in enumerate(zip(library_ids, variant_seqs)):
            protein, var_start, log_p = self._log_p_for(lib)
            var_positions = list(range(var_start, var_start + 12))
            if len(seq) != 36:
                raise ValueError(
                    f"variant_seq has {len(seq)} nt, expected 36"
                )
            codons = [seq[j : j + 3].lower() for j in range(0, 36, 3)]
            try:
                token_ids = [
                    self.token2index[
                        f"{protein[var_positions[j]].lower()}_{codons[j]}"
                    ]
                    for j in range(12)
                ]
            except KeyError as e:
                raise ValueError(
                    f"CodonTransformer vocab missing token {e.args[0]!r} "
                    f"for variant {seq!r}"
                ) from e
            out[i] = float(sum(
                log_p[var_positions[j], token_ids[j]] for j in range(12)
            ))
        return out
