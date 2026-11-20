"""
Auto-generated heuristic for kernel: rms_norm_per_block_quant
Backend: decision_tree

Provides:
- key_rms_norm_per_block_quant(*args): Returns config index (cache key)
- autotune_rms_norm_per_block_quant(*args): Returns config dict for the given arguments
"""

import torch


def key_rms_norm_per_block_quant(*args) -> int:
    """Select config index for the given arguments (also serves as cache key)."""
    _arg0_dim0 = int(args[0].shape[0]) if len(args) > 0 and isinstance(args[0], torch.Tensor) and args[0].ndim > 0 else 0
    _arg0_dim1 = int(args[0].shape[1]) if len(args) > 0 and isinstance(args[0], torch.Tensor) and args[0].ndim > 1 else 0
    if _arg0_dim1 <= 2048.0:
        return 0
    else:
        if _arg0_dim1 <= 4096.0:
            return 2
        else:
            if _arg0_dim0 <= 64.0:
                return 1
            else:
                return 0


def autotune_rms_norm_per_block_quant(*args) -> dict:
    """Select the optimal config for the given arguments."""
    _C = [
        {'block_sizes': [1024, 16], 'loop_orders': [[1, 0]], 'range_unroll_factors': [0, 0, 3, 0], 'range_warp_specializes': [], 'range_num_stages': [0, 3, 0, 0], 'range_multi_buffers': [None, None, False, None], 'range_flattens': [None, False, False, None], 'static_ranges': [True], 'load_eviction_policies': ['', '', '', 'first', '', 'last'], 'num_warps': 8, 'num_stages': 5, 'indexing': ['pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [8192, 64], 'loop_orders': [[0, 1]], 'range_unroll_factors': [0, 1, 1, 0], 'range_warp_specializes': [], 'range_num_stages': [0, 2, 0, 0], 'range_multi_buffers': [None, True, True, None], 'range_flattens': [None, False, False, None], 'static_ranges': [True], 'load_eviction_policies': ['first', 'first', 'first', 'first', 'first', ''], 'num_warps': 32, 'num_stages': 5, 'indexing': ['pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [2048, 32], 'loop_orders': [[0, 1]], 'range_unroll_factors': [0, 4, 1, 0], 'range_warp_specializes': [], 'range_num_stages': [0, 4, 0, 0], 'range_multi_buffers': [None, None, True, None], 'range_flattens': [None, True, True, None], 'static_ranges': [True], 'load_eviction_policies': ['first', '', 'last', 'last', 'last', ''], 'num_warps': 16, 'num_stages': 2, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
    ]
    return _C[key_rms_norm_per_block_quant(*args)]
