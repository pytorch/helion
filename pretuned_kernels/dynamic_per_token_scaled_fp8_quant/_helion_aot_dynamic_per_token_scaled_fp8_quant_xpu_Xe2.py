"""
Auto-generated heuristic for kernel: dynamic_per_token_scaled_fp8_quant
Backend: decision_tree

Provides:
- key_dynamic_per_token_scaled_fp8_quant(*args): Returns config index (cache key)
- autotune_dynamic_per_token_scaled_fp8_quant(*args): Returns config dict for the given arguments
"""

import torch


def key_dynamic_per_token_scaled_fp8_quant(*args) -> int:
    """Select config index for the given arguments (also serves as cache key)."""
    _arg0_dim0 = int(args[0].shape[0]) if len(args) > 0 and isinstance(args[0], torch.Tensor) and args[0].ndim > 0 else 0
    _arg0_dim1 = int(args[0].shape[1]) if len(args) > 0 and isinstance(args[0], torch.Tensor) and args[0].ndim > 1 else 0
    if _arg0_dim1 <= 4096.0:
        if _arg0_dim1 <= 2048.0:
            return 0
        else:
            if _arg0_dim0 <= 1024.0:
                return 2
            else:
                if _arg0_dim0 <= 2048.0:
                    return 0
                else:
                    return 2
    else:
        if _arg0_dim0 <= 1024.0:
            return 1
        else:
            return 0


def autotune_dynamic_per_token_scaled_fp8_quant(*args) -> dict:
    """Select the optimal config for the given arguments."""
    _C = [
        {'block_sizes': [2048, 2048], 'range_unroll_factors': [0, 1, 4], 'range_warp_specializes': [], 'range_num_stages': [0, 2, 1], 'range_multi_buffers': [None, None, None], 'range_flattens': [None, None, None], 'load_eviction_policies': ['first', ''], 'num_warps': 8, 'num_stages': 3, 'indexing': ['tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [8192, 2048], 'range_unroll_factors': [0, 4, 4], 'range_warp_specializes': [], 'range_num_stages': [0, 0, 0], 'range_multi_buffers': [None, True, True], 'range_flattens': [None, True, False], 'load_eviction_policies': ['', ''], 'num_warps': 16, 'num_stages': 5, 'indexing': ['pointer', 'tensor_descriptor', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [4096, 4096], 'range_unroll_factors': [4, 3, 2], 'range_warp_specializes': [], 'range_num_stages': [4, 4, 3], 'range_multi_buffers': [True, True, None], 'range_flattens': [False, False, False], 'load_eviction_policies': ['last', 'last'], 'num_warps': 16, 'num_stages': 7, 'indexing': ['tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'persistent_blocked', 'num_sm_multiplier': 4},
    ]
    return _C[key_dynamic_per_token_scaled_fp8_quant(*args)]
