"""
Auto-generated heuristic for kernel: rope_fwd
Backend: decision_tree

Provides:
- key_rope_fwd(*args): Returns config index (cache key)
- autotune_rope_fwd(*args): Returns config dict for the given arguments
"""

import torch


def key_rope_fwd(*args) -> int:
    """Select config index for the given arguments (also serves as cache key)."""
    _key_0 = args[0]
    if _key_0 <= 512.0:
        return 1
    else:
        return 0


def autotune_rope_fwd(*args) -> dict:
    """Select the optimal config for the given arguments."""
    _C = [
        {'block_sizes': [1, 1], 'loop_orders': [[1, 0]], 'l2_groupings': [1], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['', '', '', ''], 'num_warps': 4, 'num_stages': 1, 'indexing': ['pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [1, 16], 'loop_orders': [[0, 1]], 'l2_groupings': [64], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['last', 'last', 'last', 'last'], 'num_warps': 16, 'num_stages': 7, 'indexing': ['pointer', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
    ]
    return _C[key_rope_fwd(*args)]
