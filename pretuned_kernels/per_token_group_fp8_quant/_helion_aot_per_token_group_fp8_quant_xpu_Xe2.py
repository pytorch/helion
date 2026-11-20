"""
Auto-generated heuristic for kernel: per_token_group_fp8_quant
Backend: decision_tree

Provides:
- key_per_token_group_fp8_quant(*args): Returns config index (cache key)
- autotune_per_token_group_fp8_quant(*args): Returns config dict for the given arguments
"""

import torch


def key_per_token_group_fp8_quant(*args) -> int:
    """Select config index for the given arguments (also serves as cache key)."""
    _arg0_numel = int(args[0].numel()) if len(args) > 0 and isinstance(args[0], torch.Tensor) else 0
    if _arg0_numel <= 262144.0:
        return 2
    else:
        if _arg0_numel <= 524288.0:
            return 0
        else:
            return 1


def autotune_per_token_group_fp8_quant(*args) -> dict:
    """Select the optimal config for the given arguments."""
    _C = [
        {'block_sizes': [4], 'loop_orders': [[1, 2, 0]], 'l2_groupings': [16], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['first'], 'num_warps': 2, 'num_stages': 2, 'indexing': ['tensor_descriptor', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [8], 'loop_orders': [[2, 1, 0]], 'l2_groupings': [32], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': [''], 'num_warps': 2, 'num_stages': 7, 'indexing': ['pointer', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [8], 'loop_orders': [[2, 1, 0]], 'l2_groupings': [16], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['last'], 'num_warps': 8, 'num_stages': 2, 'indexing': ['pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    ]
    return _C[key_per_token_group_fp8_quant(*args)]
