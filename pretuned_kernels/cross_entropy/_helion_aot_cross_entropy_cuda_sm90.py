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
    _arg0_dim1 = int(args[0].shape[1]) if len(args) > 0 and isinstance(args[0], torch.Tensor) and args[0].ndim > 1 else 0
    if _arg0_dim1 <= 129280.0:
        return 1
    else:
        return 0


def autotune_cross_entropy(*args) -> dict:
    """Select the optimal config for the given arguments."""
    _C = [
        {'block_sizes': [1], 'reduction_loops': [32768], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['last', 'first', 'last', 'first', 'first'], 'num_warps': 32, 'num_stages': 1, 'indexing': ['pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [1], 'reduction_loops': [None], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['first', 'first', 'first', 'first', ''], 'num_warps': 16, 'num_stages': 1, 'indexing': ['pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    ]
    return _C[key_cross_entropy(*args)]
