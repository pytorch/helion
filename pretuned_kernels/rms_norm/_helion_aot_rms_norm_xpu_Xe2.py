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
    _arg0_numel = int(args[0].numel()) if len(args) > 0 and isinstance(args[0], torch.Tensor) else 0
    if _arg0_dim1 <= 6144.0:
        if _arg0_dim1 <= 256.0:
            if _arg0_dim1 <= 96.0:
                return 1
            else:
                if _arg0_dim0 <= 2048.0:
                    return 3
                else:
                    return 1
        else:
            if _arg0_numel <= 4192256.0:
                if _arg0_dim1 <= 1536.0:
                    if _arg0_dim1 <= 1023.0:
                        if _arg0_dim1 <= 768.0:
                            return 0
                        else:
                            return 3
                    else:
                        return 0
                else:
                    return 3
            else:
                return 0
    else:
        return 2


def autotune_rms_norm(*args) -> dict:
    """Select the optimal config for the given arguments."""
    _C = [
        {'block_sizes': [1], 'reduction_loops': [None], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['last', '', 'first', '', 'last'], 'num_warps': 4, 'num_stages': 2, 'indexing': ['pointer', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [32], 'reduction_loops': [128], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['', 'last', '', '', ''], 'num_warps': 32, 'num_stages': 3, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [1], 'reduction_loops': [4096], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['', '', '', '', ''], 'num_warps': 32, 'num_stages': 7, 'indexing': ['pointer', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [4], 'reduction_loops': [32], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['', 'last', '', 'first', ''], 'num_warps': 2, 'num_stages': 4, 'indexing': ['pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    ]
    return _C[key_rms_norm(*args)]
