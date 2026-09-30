"""GB300 configs for contiguous BF16 top-k, with optional selected-logit softmax.

Each override records a separately tuned shape and softmax mode. The shared
values below only remove repetition from the saved configurations.
"""

from __future__ import annotations

import torch

import helion

# Recorded policy is independent of process environment and autotune effort.
STRUCTURAL_POLICY = helion.CuteStructuralPolicy(
    cute_region_fission=True,
    cute_full_slice_matmul_tiling=True,
    cute_segmented_matmul_tiling=True,
    cute_flatten_nested_reductions=True,
    cute_materialize_transformed_operands=True,
)

_DEFAULT_CONFIG = {
    "block_sizes": [32],
    "reduction_loops": [32],
    "cute_topk_lanes_per_row": 8,
    "cute_topk_rows_per_block": 64,
    "cute_topk_vector_width": 8,
    "cute_topk_output_vector_width": 2,
    "cute_topk_value_mode": "decode",
    "cute_topk_key_dtype": "int32",
    "cute_topk_rank_mode": "ordinal",
    "cute_topk_selection_layout": "distributed",
    "cute_topk_sort_network": "compact_pruned",
    "cute_topk_key_encoder": "paired",
    "cute_topk_defer_value_gathers": True,
    "cute_topk_merge_schedule": "sequential",
    "cute_vector_widths": [1, 1, 1],
    "cute_lane_layouts": ["blocked", "blocked", "blocked"],
    "cute_reduction_reloads": ["auto", "auto"],
}

_CONFIG_OVERRIDES = {
    (65536, 64, 8, False): {
        "cute_proven_bounds": False,
        "cute_topk_lanes_per_row": 2,
        "cute_topk_rows_per_block": 128,
        "cute_topk_output_vector_width": 4,
        "cute_topk_selection_layout": "replicated",
        "cute_topk_sort_network": "batcher",
        "cute_topk_merge_schedule": "balanced",
    },
    (65536, 64, 16, False): {
        "cute_proven_bounds": False,
        "cute_topk_lanes_per_row": 2,
        "cute_topk_selection_layout": "replicated",
    },
    (65536, 64, 32, False): {
        "cute_proven_bounds": False,
        "cute_topk_lanes_per_row": 2,
        "cute_topk_rows_per_block": 32,
        "cute_topk_output_vector_width": 8,
        "cute_topk_sort_network": "batcher",
        "cute_topk_defer_value_gathers": False,
        "cute_topk_merge_schedule": "balanced",
    },
    (65536, 128, 8, False): {
        "cute_proven_bounds": False,
        "cute_topk_lanes_per_row": 4,
        "cute_topk_output_vector_width": 1,
        "cute_topk_key_dtype": "float32_bits",
        "cute_topk_sort_network": "batcher",
        "cute_topk_defer_value_gathers": False,
        "cute_topk_merge_schedule": "balanced",
    },
    (65536, 128, 16, False): {
        "cute_proven_bounds": False,
        "cute_topk_lanes_per_row": 2,
        "cute_topk_output_vector_width": 4,
        "cute_topk_sort_network": "compact",
    },
    (65536, 128, 32, False): {
        "cute_proven_bounds": False,
        "cute_topk_lanes_per_row": 2,
        "cute_topk_rows_per_block": 32,
        "cute_topk_output_vector_width": 4,
        "cute_topk_key_dtype": "float32_bits",
        "cute_topk_selection_layout": "replicated",
    },
    (65536, 256, 8, False): {
        "block_sizes": [16],
        "reduction_loops": [64],
        "cute_proven_bounds": False,
        "cute_topk_lanes_per_row": 4,
        "cute_topk_vector_width": 4,
        "cute_topk_key_dtype": "float32_bits",
        "cute_topk_sort_network": "compact",
    },
    (65536, 256, 16, False): {
        "block_sizes": [16],
        "reduction_loops": [64],
        "cute_proven_bounds": False,
        "cute_topk_lanes_per_row": 4,
        "cute_topk_rows_per_block": 32,
        "cute_topk_sort_network": "compact",
    },
    (65536, 256, 32, False): {
        "block_sizes": [16],
        "reduction_loops": [64],
        "cute_proven_bounds": False,
        "cute_topk_lanes_per_row": 4,
        "cute_topk_rows_per_block": 32,
    },
    (65536, 512, 8, False): {
        "block_sizes": [1],
        "reduction_loops": [None],
        "cute_proven_bounds": False,
        "cute_topk_rows_per_block": 16,
        "cute_topk_key_dtype": "float32_bits",
        "cute_topk_sort_network": "batcher",
        "cute_topk_merge_schedule": "balanced",
    },
    (65536, 512, 16, False): {
        "block_sizes": [1],
        "reduction_loops": [None],
        "cute_proven_bounds": False,
        "cute_topk_rows_per_block": 16,
        "cute_topk_vector_width": 2,
        "cute_topk_output_vector_width": 1,
        "cute_topk_value_mode": "gather",
        "cute_topk_defer_value_gathers": False,
    },
    (65536, 512, 32, False): {
        "block_sizes": [1],
        "reduction_loops": [None],
        "cute_proven_bounds": False,
        "cute_topk_rows_per_block": 4,
    },
    (65536, 1024, 8, False): {
        "block_sizes": [1],
        "reduction_loops": [None],
        "cute_proven_bounds": False,
        "cute_topk_rows_per_block": 8,
        "cute_topk_key_dtype": "float32_bits",
        "cute_topk_merge_schedule": "balanced",
    },
    (65536, 1024, 16, False): {
        "block_sizes": [1],
        "reduction_loops": [None],
        "cute_proven_bounds": False,
        "cute_topk_rows_per_block": 8,
        "cute_topk_output_vector_width": 8,
        "cute_topk_sort_network": "compact",
    },
    (65536, 1024, 32, False): {
        "block_sizes": [1],
        "reduction_loops": [None],
        "cute_proven_bounds": False,
        "cute_topk_lanes_per_row": 16,
        "cute_topk_rows_per_block": 4,
        "cute_topk_value_mode": "gather",
        "cute_topk_defer_value_gathers": False,
    },
    (65536, 64, 8, True): {
        "cute_topk_lanes_per_row": 2,
        "cute_topk_output_vector_width": 4,
        "cute_topk_selection_layout": "replicated",
    },
    (65536, 64, 16, True): {
        "cute_topk_lanes_per_row": 2,
        "cute_topk_sort_network": "compact",
    },
    (65536, 64, 32, True): {
        "cute_topk_lanes_per_row": 2,
        "cute_topk_rows_per_block": 16,
        "cute_topk_output_vector_width": 8,
        "cute_topk_defer_value_gathers": False,
        "cute_topk_merge_schedule": "balanced",
    },
    (65536, 128, 8, True): {
        "cute_topk_lanes_per_row": 4,
        "cute_topk_output_vector_width": 1,
        "cute_topk_sort_network": "compact",
        "cute_topk_defer_value_gathers": False,
        "cute_topk_merge_schedule": "balanced",
    },
    (65536, 128, 16, True): {
        "cute_topk_lanes_per_row": 4,
        "cute_topk_output_vector_width": 8,
        "cute_topk_key_dtype": "float32_bits",
    },
    (65536, 128, 32, True): {
        "cute_topk_lanes_per_row": 4,
        "cute_topk_rows_per_block": 16,
        "cute_topk_output_vector_width": 4,
        "cute_topk_sort_network": "compact",
        "cute_topk_merge_schedule": "balanced",
    },
    (65536, 256, 8, True): {
        "block_sizes": [16],
        "reduction_loops": [64],
        "cute_topk_lanes_per_row": 4,
        "cute_topk_output_vector_width": 1,
        "cute_topk_sort_network": "batcher",
        "cute_topk_defer_value_gathers": False,
        "cute_topk_merge_schedule": "balanced",
    },
    (65536, 256, 16, True): {
        "block_sizes": [16],
        "reduction_loops": [64],
        "cute_topk_lanes_per_row": 2,
        "cute_topk_rows_per_block": 32,
        "cute_topk_sort_network": "compact",
    },
    (65536, 256, 32, True): {
        "block_sizes": [16],
        "reduction_loops": [64],
        "cute_topk_lanes_per_row": 4,
        "cute_topk_rows_per_block": 32,
        "cute_topk_merge_schedule": "balanced",
    },
    (65536, 512, 8, True): {
        "block_sizes": [1],
        "reduction_loops": [None],
        "cute_topk_output_vector_width": 4,
        "cute_topk_sort_network": "compact",
    },
    (65536, 512, 16, True): {
        "block_sizes": [1],
        "reduction_loops": [None],
        "cute_topk_rows_per_block": 16,
        "cute_topk_output_vector_width": 1,
        "cute_topk_defer_value_gathers": False,
        "cute_topk_merge_schedule": "balanced",
    },
    (65536, 512, 32, True): {
        "block_sizes": [1],
        "reduction_loops": [None],
        "cute_topk_rows_per_block": 16,
        "cute_topk_output_vector_width": 1,
        "cute_topk_value_mode": "gather",
        "cute_topk_defer_value_gathers": False,
    },
    (65536, 1024, 8, True): {
        "block_sizes": [1],
        "reduction_loops": [None],
        "cute_topk_rows_per_block": 16,
        "cute_topk_output_vector_width": 8,
        "cute_topk_value_mode": "gather",
        "cute_topk_key_dtype": "float32_bits",
        "cute_topk_merge_schedule": "balanced",
    },
    (65536, 1024, 16, True): {
        "block_sizes": [1],
        "reduction_loops": [None],
        "cute_topk_lanes_per_row": 16,
        "cute_topk_rows_per_block": 2,
        "cute_topk_output_vector_width": 4,
        "cute_topk_key_dtype": "float32_bits",
        "cute_topk_defer_value_gathers": False,
    },
    (65536, 1024, 32, True): {
        "block_sizes": [1],
        "reduction_loops": [None],
        "cute_topk_rows_per_block": 32,
        "cute_topk_value_mode": "gather",
        "cute_topk_key_dtype": "float32",
    },
}

_CONFIGS = {
    shape: {**_DEFAULT_CONFIG, **overrides}
    for shape, overrides in _CONFIG_OVERRIDES.items()
}
# Standalone AOT compilation can inspect the complete config set.
CONFIGS = [
    helion.CuteStructuralConfig(config, STRUCTURAL_POLICY)
    for config in _CONFIGS.values()
]
_HELION_AOT_STRUCTURAL_MANIFEST = {
    "schema": "helion.aot.structural_model",
    "version": 1,
    "policy": STRUCTURAL_POLICY.to_dict(),
    "policy_id": STRUCTURAL_POLICY.identity(),
    "configs": {"topk": [config.to_dict() for config in CONFIGS]},
}


def key_topk(
    x: torch.Tensor,
    k: int,
    softmax: bool = False,
) -> tuple[int, int, int, bool]:
    """Keep shape, K, and the selected-value epilogue in the specialization key."""
    if x.ndim != 2 or x.dtype != torch.bfloat16 or not x.is_contiguous():
        raise ValueError(
            "Pretuned top-k requires a contiguous two-dimensional BF16 input"
        )
    key = (x.size(0), x.size(1), k, softmax)
    if key not in _CONFIGS:
        raise ValueError(
            f"No pretuned top-k config for {key}; use HELION_AOT_MODE=collect to tune"
        )
    return key


def autotune_topk(
    x: torch.Tensor,
    k: int,
    softmax: bool = False,
) -> helion.CuteStructuralConfig:
    """Select the measured knobs and structural policy without online tuning."""
    return helion.CuteStructuralConfig(
        _CONFIGS[key_topk(x, k, softmax)], STRUCTURAL_POLICY
    )
