"""
Auto-generated heuristic for kernel: silu_mul_fp8
Backend: decision_tree

Provides:
- key_silu_mul_fp8(*args): Returns config index (cache key)
- autotune_silu_mul_fp8(*args): Returns config dict for the given arguments
"""

import torch


def key_silu_mul_fp8(*args) -> int:
    """Select config index for the given arguments (also serves as cache key)."""
    _arg0_dim0 = int(args[0].shape[0]) if len(args) > 0 and isinstance(args[0], torch.Tensor) and args[0].ndim > 0 else 0
    _arg0_dim1 = int(args[0].shape[1]) if len(args) > 0 and isinstance(args[0], torch.Tensor) and args[0].ndim > 1 else 0
    if _arg0_dim0 <= 1.0:
        return 0
    else:
        if _arg0_dim1 <= 16384.0:
            return 1
        else:
            if _arg0_dim1 <= 22016.0:
                if _arg0_dim0 <= 8.0:
                    return 1
                else:
                    if _arg0_dim0 <= 128.0:
                        return 0
                    else:
                        return 1
            else:
                return 1


def autotune_silu_mul_fp8(*args) -> dict:
    """Select the optimal config for the given arguments."""
    _C = [
        {'block_sizes': [1, 512], 'loop_orders': [[0, 1]], 'flatten_loops': [False], 'l2_groupings': [1], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['', '', ''], 'num_warps': 4, 'num_stages': 1, 'indexing': ['pointer', 'pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [2, 128], 'loop_orders': [[0, 1]], 'flatten_loops': [True], 'l2_groupings': [4], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['', '', 'first'], 'num_warps': 2, 'num_stages': 1, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    ]
    return _C[key_silu_mul_fp8(*args)]
