"""
Auto-generated heuristic for kernel: cross_entropy
Backend: decision_tree

Provides:
- key_cross_entropy(*args): Returns config index (cache key)
- autotune_cross_entropy(*args): Returns config dict for the given arguments
"""

import torch


def key_cross_entropy(*args) -> int:
    """Select config index for the given arguments (also serves as cache key)."""
    # No features needed
    return 0


def autotune_cross_entropy(*args) -> dict:
    """Select the optimal config for the given arguments."""
    _C = [
        {'block_sizes': [1], 'reduction_loops': [8192], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['last', 'last', 'first', 'last', 'last'], 'num_warps': 32, 'num_stages': 6, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
    ]
    return _C[key_cross_entropy(*args)]
