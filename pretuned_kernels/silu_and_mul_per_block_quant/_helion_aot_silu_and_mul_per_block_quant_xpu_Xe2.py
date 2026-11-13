"""
Auto-generated heuristic for kernel: silu_and_mul_per_block_quant
Backend: decision_tree

Provides:
- key_silu_and_mul_per_block_quant(*args): Returns config index (cache key)
- autotune_silu_and_mul_per_block_quant(*args): Returns config dict for the given arguments
"""

import torch


def key_silu_and_mul_per_block_quant(*args) -> int:
    """Select config index for the given arguments (also serves as cache key)."""
    # No features needed
    return 0


def autotune_silu_and_mul_per_block_quant(*args) -> dict:
    """Select the optimal config for the given arguments."""
    _C = [
        {'block_sizes': [4], 'loop_orders': [[1, 2, 0]], 'l2_groupings': [8], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['', 'last'], 'num_warps': 4, 'num_stages': 1, 'indexing': ['pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    ]
    return _C[key_silu_and_mul_per_block_quant(*args)]
