"""Whole-dataset (mu, sigma) used by ExoShorkie's distilled-student notebooks.

These describe ln(1 + summed 16 bp CPM coverage), pooled across both strands
and all overlapping, cropped windows; sigma uses ddof=0. They differ from the
per-CV-fold training statistics. The authors invert with expm1(z * sigma + mu).

To recompute, load the normalized coverage and FASTAs from Figshare 31075375.
Use upstream build_windows_per_source(win_bp=16384, target_wins=10000,
train_ratio=0.8), then compute_global_stats(crop_bp=1024, bin_size_bp=16).
Pool both chr7 parts. For genomic, pool chrI–chrXVI and instead use
build_windows_full_genome(target_windows=10000, win_bp=16384) followed by
compute_logz_stats_multi with the same crop/bin sizes. Round to four decimals.

Source: https://github.com/OrensteinLab/ExoShorkie/tree/v1.0.1/Motif_visualization
"""

# Released checkpoints predate upstream's change to 109. Across the six
# students, only species channel 114 has different first-convolution weights,
# agreeing with the v1.0.1 training loader. Keep Shorkie's default independent.
RELEASED_SPECIES_INDEX = 114

# Keys match the upstream distilled-student names; values are (mu, sigma).
NORMALIZATION_STATS: dict[str, tuple[float, float]] = {
    "Mpneumo": (3.5228, 1.7232),
    "Mmmyco": (1.2864, 1.5034),
    "HPRT1": (4.4673, 1.8032),
    "HPRT1R": (4.5184, 1.7808),
    "chr7": (2.0900, 1.5280),
    "dChr": (4.1016, 1.9481),
    "genomic": (1.7390, 1.8168),
}
