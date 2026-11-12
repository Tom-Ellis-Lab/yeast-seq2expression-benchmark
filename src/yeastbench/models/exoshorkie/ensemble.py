"""Equal-weight ensemble in decoded, nonnegative CPM coverage space."""
from __future__ import annotations

from yeastbench.models.exoshorkie.readouts import CoverageReadouts


class ExoShorkieEnsemble(CoverageReadouts):
    """Decode and clamp each student before averaging, then apply readouts.

    Students run sequentially on the same device to bound activation memory.
    No normalized outputs, logarithmic effects or correlations are averaged.
    """

    def __init__(self, members):
        self.members = tuple(members)
        if not self.members:
            raise ValueError("An ensemble requires at least one student")
        self.variants = tuple(member.variant for member in self.members)
        if len(set(self.variants)) != len(self.variants):
            raise ValueError("Ensemble students must be unique")
        self.variant = "ensemble[" + ",".join(self.variants) + "]"
        for attr in ("device", "seq_len", "output_bins", "bin_width", "crop_bp_each_side"):
            value = getattr(self.members[0], attr)
            if any(getattr(member, attr) != value for member in self.members):
                raise ValueError(f"Ensemble students must share {attr}")
            setattr(self, attr, value)

    def forward_coverage_binned(self, x):
        total = self.members[0].forward_coverage_binned(x)
        for member in self.members[1:]:
            total = total + member.forward_coverage_binned(x)
        return total / len(self.members)

    def coverage_statistics(self):
        per_student = {member.variant: member.coverage_statistics() for member in self.members}
        count = sum(d["coverage_bins_processed"] for d in per_student.values())
        negative = sum(d["negative_bins_clipped"] for d in per_student.values())
        return {
            "coverage_bins_processed": count,
            "negative_bins_clipped": negative,
            "negative_bin_fraction": negative / count if count else 0.0,
            "per_student_coverage_statistics": per_student,
        }
