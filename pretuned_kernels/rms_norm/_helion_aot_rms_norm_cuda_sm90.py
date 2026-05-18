"""
Auto-generated heuristic for kernel: rms_norm
Backend: decision_tree

Provides:
- key_rms_norm(*args): Returns config index (cache key)
- autotune_rms_norm(*args): Returns config dict for the given arguments
"""

import torch


def key_rms_norm(*args) -> int:
    """Select config index for the given arguments (also serves as cache key)."""
    _arg0_dim0 = int(args[0].shape[0]) if len(args) > 0 and isinstance(args[0], torch.Tensor) and args[0].ndim > 0 else 0
    _arg0_dim1 = int(args[0].shape[1]) if len(args) > 0 and isinstance(args[0], torch.Tensor) and args[0].ndim > 1 else 0
    if _arg0_dim1 <= 4096.0:
        if _arg0_dim0 <= 16384.0:
            if _arg0_dim1 <= 3584.0:
                return 0
            else:
                if _arg0_dim0 <= 2048.0:
                    return 1
                else:
                    return 0
        else:
            if _arg0_dim1 <= 384.0:
                return 2
            else:
                return 0
    else:
        return 1


def autotune_rms_norm(*args) -> dict:
    """Select the optimal config for the given arguments."""
    _C = [
        {'block_sizes': [4], 'reduction_loops': [None], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['last', '', '', 'last', 'last'], 'num_warps': 8, 'num_stages': 5, 'indexing': ['pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [1], 'reduction_loops': [None], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['', '', '', 'last', 'last'], 'num_warps': 16, 'num_stages': 1, 'indexing': ['pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [16], 'reduction_loops': [None], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['', '', '', 'last', 'first'], 'num_warps': 4, 'num_stages': 2, 'indexing': ['pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
    ]
    return _C[key_rms_norm(*args)]
