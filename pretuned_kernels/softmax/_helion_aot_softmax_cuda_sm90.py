"""
Auto-generated heuristic for kernel: softmax
Backend: decision_tree

Provides:
- key_softmax(*args): Returns config index (cache key)
- autotune_softmax(*args): Returns config dict for the given arguments
"""

import torch


def key_softmax(*args) -> int:
    """Select config index for the given arguments (also serves as cache key)."""
    _arg0_dim1 = int(args[0].shape[1]) if len(args) > 0 and isinstance(args[0], torch.Tensor) and args[0].ndim > 1 else 0
    if _arg0_dim1 <= 2048.0:
        return 1
    else:
        if _arg0_dim1 <= 7552.0:
            return 0
        else:
            if _arg0_dim1 <= 9984.0:
                if _arg0_dim1 <= 8192.0:
                    if _arg0_dim1 <= 7680.0:
                        return 2
                    else:
                        return 0
                else:
                    if _arg0_dim1 <= 8960.0:
                        if _arg0_dim1 <= 8832.0:
                            return 2
                        else:
                            return 0
                    else:
                        return 2
            else:
                if _arg0_dim1 <= 12672.0:
                    if _arg0_dim1 <= 11136.0:
                        return 0
                    else:
                        if _arg0_dim1 <= 11264.0:
                            return 2
                        else:
                            return 0
                else:
                    return 2


def autotune_softmax(*args) -> dict:
    """Select the optimal config for the given arguments."""
    _C = [
        {'block_sizes': [1], 'reduction_loops': [None], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['first', 'first', 'first', ''], 'num_warps': 8, 'num_stages': 2, 'indexing': ['tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [1], 'reduction_loops': [None], 'range_unroll_factors': [2], 'range_warp_specializes': [], 'range_num_stages': [4], 'range_multi_buffers': [True], 'range_flattens': [True], 'load_eviction_policies': ['last', 'first', 'first', 'last'], 'num_warps': 1, 'num_stages': 3, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'persistent_interleaved', 'num_sm_multiplier': 32},
        {'block_sizes': [1], 'reduction_loops': [None], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['first', 'first', 'first', ''], 'num_warps': 16, 'num_stages': 2, 'indexing': ['pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    ]
    return _C[key_softmax(*args)]
