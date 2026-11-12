"""Shared Shorkie architecture + track constants.

Lives here (rather than inside any task-specific adapter) so other
adapters don't have to import from a sibling — removing one
task-adapter must not break another. See PR #1 review.
"""
from __future__ import annotations

# Architecture (derived from data/models/shorkie/params.json):
# 16,384 bp input → ResTower (pool 2, repeat 7) → 128× downsample, then
# 3 UNet upsamples (2× each) → 16× net downsample, then Cropping1D
# (64 bins each side) → 896 output bins covering the central 14,336 bp.
SEQ_LEN: int = 16384
OUTPUT_BINS: int = 896
BIN_WIDTH: int = 16
CROP_BP_EACH_SIDE: int = 1024  # 64 bins × 16 bp/bin

# Output-track indices for the '1000-RNA-Seq' group from the Shorkie
# canonical targets sheet. The spec calls this set '1011 RNA-seq tracks';
# the released targets sheet has 1014 contiguous indices 4201..5214.
SHORKIE_1000_RNA_SEQ_TRACK_IDS: list[int] = list(range(4201, 5215))
# Backward-compatible name used by the eQTL adapter; this contains 1014 tracks.
SHORKIE_1011_RNA_SEQ_TRACK_IDS: list[int] = SHORKIE_1000_RNA_SEQ_TRACK_IDS

# 384 T0 RNA-seq track indices extracted from data/models/shorkie/
# targets.txt — tracks whose identifier contains "_T0_". Shorkie
# RNA-seq tracks are unstranded coverage bigwigs, so a single subset
# works for both positive- and negative-strand host genes.
SHORKIE_T0_RNA_SEQ_TRACK_IDS: list[int] = [
    1148, 1156, 1163, 1168, 1173, 1175, 1178, 1186, 1193, 1201,
    1208, 1212, 1219, 1227, 1234, 1240, 1248, 1256, 1263, 1271,
    1279, 1294, 1302, 1310, 1317, 1323, 1331, 1339, 1347, 1355,
    1363, 1371, 1377, 1385, 1393, 1400, 1408, 1416, 1424, 1432,
    1440, 1448, 1456, 1464, 1471, 1479, 1487, 1504, 1515, 1520,
    1525, 1536, 1549, 1576, 1586, 1594, 1599, 1607, 1615, 1631,
    1646, 1652, 1657, 1662, 1664, 1672, 1680, 1687, 1692, 1698,
    1711, 1720, 1728, 1729, 1739, 1740, 1759, 1760, 1770, 1799,
    1808, 1811, 1817, 1818, 1845, 1857, 1858, 1876, 1883, 1884,
    1894, 1900, 1908, 1909, 1918, 1919, 1929, 1930, 1938, 1939,
    1947, 1948, 1960, 1963, 1979, 1980, 1993, 1994, 2005, 2006,
    2020, 2028, 2035, 2036, 2049, 2050, 2062, 2066, 2067, 2075,
    2076, 2109, 2117, 2125, 2133, 2141, 2149, 2157, 2165, 2173,
    2181, 2189, 2197, 2205, 2213, 2221, 2229, 2237, 2245, 2253,
    2261, 2269, 2277, 2285, 2293, 2301, 2309, 2317, 2325, 2333,
    2341, 2349, 2357, 2365, 2373, 2381, 2389, 2397, 2405, 2413,
    2421, 2429, 2437, 2445, 2453, 2461, 2469, 2477, 2485, 2497,
    2504, 2516, 2523, 2530, 2538, 2550, 2558, 2566, 2574, 2581,
    2589, 2597, 2604, 2612, 2618, 2626, 2634, 2640, 2647, 2655,
    2661, 2673, 2685, 2693, 2701, 2706, 2713, 2720, 2726, 2731,
    2743, 2750, 2758, 2762, 2768, 2775, 2783, 2791, 2799, 2807,
    2815, 2823, 2831, 2839, 2847, 2854, 2861, 2868, 2876, 2884,
    2892, 2899, 2907, 2915, 2923, 2931, 2939, 2947, 2954, 2962,
    2970, 2977, 2985, 2993, 3001, 3009, 3017, 3025, 3033, 3041,
    3049, 3057, 3065, 3073, 3081, 3088, 3096, 3104, 3112, 3120,
    3128, 3136, 3144, 3152, 3160, 3168, 3176, 3184, 3192, 3200,
    3208, 3216, 3224, 3232, 3240, 3248, 3256, 3264, 3270, 3278,
    3285, 3292, 3300, 3307, 3314, 3322, 3330, 3338, 3346, 3354,
    3362, 3370, 3378, 3386, 3393, 3401, 3409, 3417, 3421, 3429,
    3437, 3445, 3453, 3461, 3469, 3475, 3482, 3490, 3498, 3506,
    3514, 3522, 3527, 3535, 3543, 3551, 3559, 3567, 3575, 3583,
    3591, 3599, 3607, 3615, 3623, 3637, 3645, 3653, 3661, 3669,
    3677, 3685, 3693, 3701, 3709, 3717, 3725, 3733, 3741, 3749,
    3757, 3765, 3773, 3781, 3788, 3796, 3804, 3812, 3820, 3827,
    3835, 3842, 3850, 3858, 3866, 3874, 3882, 3890, 3904, 3912,
    3919, 3926, 3934, 3942, 3949, 3957, 3965, 3978, 3984, 3994,
    4002, 4009, 4017, 4022, 4030, 4037, 4045, 4053, 4063, 4074,
    4082, 4087, 4095, 4103, 4111, 4119, 4130, 4138, 4146, 4154,
    4166, 4184, 4188, 4193,
]
