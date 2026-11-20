"""
Auto-generated heuristic for kernel: rms_norm_dynamic_per_token_quant
Backend: decision_tree

Provides:
- key_rms_norm_dynamic_per_token_quant(*args): Returns config index (cache key)
- autotune_rms_norm_dynamic_per_token_quant(*args): Returns config dict for the given arguments
"""

import torch


def key_rms_norm_dynamic_per_token_quant(*args) -> int:
    """Select config index for the given arguments (also serves as cache key)."""
    _arg0_dim0 = int(args[0].shape[0]) if len(args) > 0 and isinstance(args[0], torch.Tensor) and args[0].ndim > 0 else 0
    _arg0_dim1 = int(args[0].shape[1]) if len(args) > 0 and isinstance(args[0], torch.Tensor) and args[0].ndim > 1 else 0
    _arg0_numel = int(args[0].numel()) if len(args) > 0 and isinstance(args[0], torch.Tensor) else 0
    if _arg0_dim0 <= 64.0:
        if _arg0_dim1 <= 4096.0:
            if _arg0_dim0 <= 32.0:
                return 0
            else:
                if _arg0_dim1 <= 2048.0:
                    return 4
                else:
                    return 5
        else:
            return 2
    else:
        if _arg0_numel <= 1310720.0:
            return 1
        else:
            if _arg0_dim1 <= 2048.0:
                if _arg0_dim0 <= 1024.0:
                    return 4
                else:
                    return 3
            else:
                if _arg0_dim1 <= 4096.0:
                    if _arg0_dim0 <= 512.0:
                        return 3
                    else:
                        if _arg0_dim0 <= 1024.0:
                            return 0
                        else:
                            return 3
                else:
                    return 3


def autotune_rms_norm_dynamic_per_token_quant(*args) -> dict:
    """Select the optimal config for the given arguments."""
    _C = [
        {'block_sizes': [4096, 4096, 4096], 'range_unroll_factors': [0, 2, 1, 4], 'range_warp_specializes': [], 'range_num_stages': [2, 3], 'range_multi_buffers': [None, True, True, True], 'range_flattens': [None, None, True, None], 'load_eviction_policies': ['last', '', '', '', 'last'], 'num_warps': 32, 'num_stages': 3, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [1024, 8192, 1024], 'range_unroll_factors': [0, 0, 2, 4], 'range_warp_specializes': [], 'range_num_stages': [3, 3], 'range_multi_buffers': [None, True, True, True], 'range_flattens': [None, True, None, None], 'load_eviction_policies': ['', 'last', 'last', 'last', 'last'], 'num_warps': 8, 'num_stages': 8, 'indexing': ['pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [8192, 8192, 2048], 'range_unroll_factors': [0, 2, 1, 0], 'range_warp_specializes': [], 'range_num_stages': [1, 0], 'range_multi_buffers': [None, None, None, None], 'range_flattens': [None, False, False, False], 'load_eviction_policies': ['last', 'last', 'first', 'last', 'last'], 'num_warps': 32, 'num_stages': 5, 'indexing': ['tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [16, 512, 128], 'range_unroll_factors': [0, 3, 1, 4], 'range_warp_specializes': [], 'range_num_stages': [3, 4], 'range_multi_buffers': [None, False, True, None], 'range_flattens': [None, True, None, True], 'load_eviction_policies': ['last', 'last', 'last', 'last', 'last'], 'num_warps': 1, 'num_stages': 8, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [2048, 2048, 2048], 'range_unroll_factors': [0, 0, 4, 4], 'range_warp_specializes': [], 'range_num_stages': [1, 3], 'range_multi_buffers': [False, None, None, None], 'range_flattens': [False, True, True, False], 'load_eviction_policies': ['', '', 'first', 'last', 'last'], 'num_warps': 8, 'num_stages': 8, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'persistent_interleaved', 'num_sm_multiplier': 128},
        {'block_sizes': [16, 1024, 512], 'range_unroll_factors': [0, 0, 0, 0], 'range_warp_specializes': [], 'range_num_stages': [0, 1], 'range_multi_buffers': [None, None, None, None], 'range_flattens': [None, None, None, None], 'load_eviction_policies': ['last', '', '', '', ''], 'num_warps': 4, 'num_stages': 1, 'indexing': ['pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
    ]
    return _C[key_rms_norm_dynamic_per_token_quant(*args)]
