"""
Auto-generated heuristic for kernel: scaled_mm
Backend: decision_tree

Provides:
- key_scaled_mm(*args): Returns config index (cache key)
- autotune_scaled_mm(*args): Returns config dict for the given arguments
"""

import torch


def key_scaled_mm(*args) -> int:
    """Select config index for the given arguments (also serves as cache key)."""
    _arg0_dim0 = int(args[0].shape[0]) if len(args) > 0 and isinstance(args[0], torch.Tensor) and args[0].ndim > 0 else 0
    _arg0_dim1 = int(args[0].shape[1]) if len(args) > 0 and isinstance(args[0], torch.Tensor) and args[0].ndim > 1 else 0
    _arg0_numel = int(args[0].numel()) if len(args) > 0 and isinstance(args[0], torch.Tensor) else 0
    if _arg0_numel <= 196608.0:
        if _arg0_dim1 <= 4096.0:
            if _arg0_dim1 <= 2048.0:
                if _arg0_dim0 <= 16.0:
                    return 6
                else:
                    return 3
            else:
                return 0
        else:
            return 5
    else:
        if _arg0_dim1 <= 6144.0:
            if _arg0_dim1 <= 4096.0:
                return 2
            else:
                return 4
        else:
            if _arg0_dim0 <= 16.0:
                if _arg0_dim1 <= 24576.0:
                    return 7
                else:
                    return 8
            else:
                return 1


def autotune_scaled_mm(*args) -> dict:
    """Select the optimal config for the given arguments."""
    _C = [
        {'block_sizes': [16, 64, 128], 'loop_orders': [[1, 0]], 'l2_groupings': [16], 'range_unroll_factors': [0, 3], 'range_warp_specializes': [], 'range_num_stages': [0, 3], 'range_multi_buffers': [None, True], 'range_flattens': [None, True], 'load_eviction_policies': ['', '', '', '', 'last', '', ''], 'num_warps': 16, 'num_stages': 3, 'indexing': ['pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [64, 256, 64], 'loop_orders': [[1, 0]], 'l2_groupings': [32], 'range_unroll_factors': [4, 2], 'range_warp_specializes': [], 'range_num_stages': [2, 1], 'range_multi_buffers': [False, False], 'range_flattens': [None, None], 'load_eviction_policies': ['first', 'last', 'last', 'last', 'first', '', ''], 'num_warps': 32, 'num_stages': 6, 'indexing': ['pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'persistent_blocked', 'num_sm_multiplier': 128},
        {'block_sizes': [64, 128, 256], 'loop_orders': [[0, 1]], 'l2_groupings': [16], 'range_unroll_factors': [0, 2], 'range_warp_specializes': [], 'range_num_stages': [0, 3], 'range_multi_buffers': [None, True], 'range_flattens': [None, False], 'load_eviction_policies': ['first', '', '', '', '', 'last', ''], 'num_warps': 32, 'num_stages': 2, 'indexing': ['pointer', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [64, 64, 128], 'loop_orders': [[0, 1]], 'l2_groupings': [1], 'range_unroll_factors': [0, 1], 'range_warp_specializes': [], 'range_num_stages': [0, 0], 'range_multi_buffers': [None, True], 'range_flattens': [None, False], 'load_eviction_policies': ['first', 'first', 'first', '', 'first', 'first', ''], 'num_warps': 32, 'num_stages': 3, 'indexing': ['pointer', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [64, 32, 64], 'loop_orders': [[0, 1]], 'l2_groupings': [8], 'range_unroll_factors': [0, 2], 'range_warp_specializes': [], 'range_num_stages': [0, 4], 'range_multi_buffers': [None, False], 'range_flattens': [None, True], 'load_eviction_policies': ['', 'last', 'first', '', '', 'last', 'first'], 'num_warps': 8, 'num_stages': 1, 'indexing': ['tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [16, 64, 128], 'loop_orders': [[0, 1]], 'l2_groupings': [2], 'range_unroll_factors': [4, 3], 'range_warp_specializes': [], 'range_num_stages': [2, 0], 'range_multi_buffers': [None, None], 'range_flattens': [None, True], 'load_eviction_policies': ['', 'first', '', 'first', 'last', 'first', ''], 'num_warps': 8, 'num_stages': 2, 'indexing': ['tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'persistent_interleaved', 'num_sm_multiplier': 64},
        {'block_sizes': [16, 64, 512], 'loop_orders': [[1, 0]], 'l2_groupings': [1], 'range_unroll_factors': [0, 4], 'range_warp_specializes': [], 'range_num_stages': [0, 0], 'range_multi_buffers': [None, None], 'range_flattens': [None, None], 'load_eviction_policies': ['', '', '', '', '', '', ''], 'num_warps': 32, 'num_stages': 8, 'indexing': ['pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [16, 256, 64], 'loop_orders': [[1, 0]], 'l2_groupings': [64], 'range_unroll_factors': [0, 4], 'range_warp_specializes': [], 'range_num_stages': [0, 0], 'range_multi_buffers': [None, None], 'range_flattens': [None, None], 'load_eviction_policies': ['', '', '', 'last', '', '', ''], 'num_warps': 16, 'num_stages': 3, 'indexing': ['pointer', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [16, 128, 32], 'loop_orders': [[1, 0]], 'l2_groupings': [32], 'range_unroll_factors': [0, 0], 'range_warp_specializes': [], 'range_num_stages': [0, 4], 'range_multi_buffers': [None, None], 'range_flattens': [None, False], 'load_eviction_policies': ['last', 'first', 'last', 'first', 'last', '', 'last'], 'num_warps': 4, 'num_stages': 1, 'indexing': ['pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
    ]
    return _C[key_scaled_mm(*args)]
