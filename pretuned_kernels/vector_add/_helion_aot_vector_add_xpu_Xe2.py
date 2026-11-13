"""
Auto-generated heuristic for kernel: vector_add
Backend: decision_tree

Provides:
- key_vector_add(*args): Returns config index (cache key)
- autotune_vector_add(*args): Returns config dict for the given arguments
"""

import torch


def key_vector_add(*args) -> int:
    """Select config index for the given arguments (also serves as cache key)."""
    # No features needed
    return 0


def autotune_vector_add(*args) -> dict:
    """Select the optimal config for the given arguments."""
    _C = [
        {'block_sizes': [64], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['', ''], 'num_warps': 1, 'num_stages': 3, 'indexing': ['pointer', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
    ]
    return _C[key_vector_add(*args)]
