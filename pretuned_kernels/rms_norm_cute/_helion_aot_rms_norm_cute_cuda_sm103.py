"""Exact GB300 (SM103) BF16 RMSNorm presets.

The thirteen configurations were selected with full LFBO tuning and verified
with CUDA graphs and cold L2. The 2048x8192 preset uses global reloads to retain
the originally generated program after rebasing. The kernel specializes shapes
explicitly; unlisted shapes, dtypes, epsilon values or layouts use defaults.
"""

from __future__ import annotations


_SHAPES = [
    (2048, 1024),
    (2048, 4096),
    (2048, 8192),
    (2048, 16384),
    (2048, 32768),
    (4096, 3584),
    (4096, 7168),
    (16384, 8192),
    (32768, 256),
    (32768, 4096),
    (32768, 65536),
    (16384, 131072),
    (8192, 262144),
]
_CONFIG_INDEX = {shape: index for index, shape in enumerate(_SHAPES)}

# Explicit collection for the AOT collect/measure/compile workflow.
CONFIGS = [
    {
        "block_sizes": [2],
        "num_threads": [2, 32],
        "reduction_loops": [512],
        "load_eviction_policies": [
            "l2_last",
            "l2_last",
            "l2_last",
            "l2_last",
            "l2_last",
        ],
        "cute_vector_widths": [16, 8],
        "cute_lane_layouts": ["strided", "blocked"],
        "cute_reduction_reloads": ["register"],
        "cute_cluster_n": 1,
        "cute_min_blocks_per_mp": 0,
    },
    {
        "block_sizes": [4],
        "num_threads": [4, 64],
        "reduction_loops": [2048],
        "load_eviction_policies": [
            "l2_last",
            "l2_last",
            "l2_last",
            "l2_last",
            "l2_last",
        ],
        "cute_vector_widths": [8, 16],
        "cute_lane_layouts": ["strided", "blocked"],
        "cute_reduction_reloads": ["auto"],
        "cute_cluster_n": 1,
        "cute_min_blocks_per_mp": 6,
    },
    {
        "block_sizes": [1],
        "num_threads": [1, 256],
        "reduction_loops": [4096],
        "load_eviction_policies": [
            "l2_last",
            "l2_last",
            "l2_last",
            "l2_last",
            "l2_last",
        ],
        "cute_vector_widths": [8, 1],
        "cute_lane_layouts": ["strided", "blocked"],
        "cute_reduction_reloads": ["gmem"],
        "cute_cluster_n": 1,
        "cute_min_blocks_per_mp": 0,
    },
    {
        "block_sizes": [1],
        "num_threads": [1, 256],
        "reduction_loops": [8192],
        "load_eviction_policies": ["", "", "l2_last", "", ""],
        "cute_vector_widths": [8, 16],
        "cute_lane_layouts": ["strided", "blocked"],
        "cute_reduction_reloads": ["register"],
        "cute_cluster_n": 1,
        "cute_min_blocks_per_mp": 0,
    },
    {
        "block_sizes": [1],
        "num_threads": [1, 256],
        "reduction_loops": [8192],
        "load_eviction_policies": [
            "l2_last",
            "l2_last",
            "l2_last",
            "l2_last",
            "l2_last",
        ],
        "cute_vector_widths": [16, 1],
        "cute_lane_layouts": ["strided", "blocked"],
        "cute_reduction_reloads": ["gmem"],
        "cute_cluster_n": 2,
        "cute_min_blocks_per_mp": 2,
    },
    {
        "block_sizes": [2],
        "num_threads": [2, 64],
        "reduction_loops": [512],
        "load_eviction_policies": ["l2_last", "streaming", "l2_last", "", "l2_last"],
        "cute_vector_widths": [8, 1],
        "cute_lane_layouts": ["strided", "blocked"],
        "cute_reduction_reloads": ["gmem"],
        "cute_cluster_n": 1,
        "cute_min_blocks_per_mp": 0,
    },
    {
        "block_sizes": [1],
        "num_threads": [1, 224],
        "reduction_loops": [3584],
        "load_eviction_policies": [
            "l2_last",
            "l2_last",
            "l2_last",
            "l2_last",
            "l2_last",
        ],
        "cute_vector_widths": [16, 1],
        "cute_lane_layouts": ["strided", "blocked"],
        "cute_reduction_reloads": ["gmem"],
        "cute_cluster_n": 1,
        "cute_min_blocks_per_mp": 0,
    },
    {
        "block_sizes": [4],
        "num_threads": [0, 512],
        "reduction_loops": [2048],
        "load_eviction_policies": ["last", "l2_last", "l2_last", "last", "l2_last"],
        "cute_vector_widths": [8, 2],
        "cute_lane_layouts": ["strided", "blocked"],
        "cute_reduction_reloads": ["auto"],
        "cute_cluster_n": 2,
        "cute_min_blocks_per_mp": 0,
    },
    {
        "block_sizes": [64],
        "num_threads": [64, 256],
        "reduction_loops": [128],
        "load_eviction_policies": ["", "", "", "l2_last", "streaming"],
        "cute_vector_widths": [8, 4],
        "cute_lane_layouts": ["strided", "blocked"],
        "cute_reduction_reloads": ["register"],
        "cute_cluster_n": 2,
        "cute_min_blocks_per_mp": 2,
    },
    {
        "block_sizes": [2],
        "num_threads": [2, 128],
        "reduction_loops": [2048],
        "load_eviction_policies": [
            "l2_last",
            "l2_last",
            "l2_last",
            "l2_last",
            "l2_last",
        ],
        "cute_vector_widths": [16, 1],
        "cute_lane_layouts": ["strided", "blocked"],
        "cute_reduction_reloads": ["gmem"],
        "cute_cluster_n": 1,
        "cute_min_blocks_per_mp": 6,
    },
    {
        "block_sizes": [1],
        "num_threads": [1, 512],
        "reduction_loops": [32768],
        "load_eviction_policies": ["l2_last", "first", "", "l2_last", "l2_last"],
        "cute_vector_widths": [8, 16],
        "cute_lane_layouts": ["strided", "strided"],
        "cute_reduction_reloads": ["gmem"],
        "cute_cluster_n": 2,
        "cute_min_blocks_per_mp": 3,
    },
    {
        "block_sizes": [1],
        "num_threads": [1, 0],
        "reduction_loops": [65536],
        "load_eviction_policies": ["last", "streaming", "", "", "last"],
        "cute_vector_widths": [8, 16],
        "cute_lane_layouts": ["strided", "strided"],
        "cute_reduction_reloads": ["gmem"],
        "cute_cluster_n": 2,
        "cute_min_blocks_per_mp": 0,
    },
    {
        "block_sizes": [1],
        "reduction_loops": [65536],
        "load_eviction_policies": ["last", "first", "first", "streaming", "streaming"],
        "cute_vector_widths": [16, 4],
        "cute_lane_layouts": ["strided", "strided"],
        "cute_reduction_reloads": ["auto"],
        "cute_cluster_n": 4,
        "cute_min_blocks_per_mp": 4,
    },
]


def key_rms_norm_cute(
    m: int,
    n: int,
    x_dtype: str,
    weight_dtype: str,
    eps: float,
    x_contiguous: bool,
    weight_contiguous: bool,
) -> int:
    """Select only the measured contiguous BF16 input envelope."""
    if (
        x_dtype != "torch.bfloat16"
        or weight_dtype != "torch.bfloat16"
        or eps != 1e-5
        or not x_contiguous
        or not weight_contiguous
    ):
        return -1
    return _CONFIG_INDEX.get((m, n), -1)


def autotune_rms_norm_cute(
    m: int,
    n: int,
    x_dtype: str,
    weight_dtype: str,
    eps: float,
    x_contiguous: bool,
    weight_contiguous: bool,
) -> dict:
    """Return the exact preset or the backend's default configuration."""
    index = key_rms_norm_cute(
        m, n, x_dtype, weight_dtype, eps, x_contiguous, weight_contiguous
    )
    return CONFIGS[index] if index >= 0 else {}
