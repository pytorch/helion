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
    _arg0_dim0 = int(args[0].shape[0]) if len(args) > 0 and isinstance(args[0], torch.Tensor) and args[0].ndim > 0 else 0
    if _arg0_dim0 <= 4194304.0:
        if _arg0_dim0 <= 1048576.0:
            return 0
        else:
            return 1
    else:
        return 0


def autotune_vector_add(*args) -> dict:
    """Select the optimal config for the given arguments."""
    _C = [
        {'block_sizes': [256], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['last', 'last'], 'num_warps': 1, 'num_stages': 7, 'indexing': ['pointer', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [4096], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['', 'last'], 'num_warps': 8, 'num_stages': 3, 'indexing': ['pointer', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
    ]
    return _C[key_vector_add(*args)]
